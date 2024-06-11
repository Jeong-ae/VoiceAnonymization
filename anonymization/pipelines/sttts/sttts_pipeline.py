from pathlib import Path
from datetime import datetime
import time

from .. import Pipeline, get_anon_level_from_config
from utils import read_kaldi_format, copy_data_dir, save_kaldi_format, check_dependencies, setup_logger

from ...modules.sttts.tts import SpeechSynthesis
from ...modules.sttts.text import SpeechRecognition
from ...modules.sttts.prosody import ProsodyExtraction, ProsodyAnonymization
from ...modules.sttts.speaker_embeddings.speaker_extractions import SpeakerExtraction
from ...modules.sttts.speaker_embeddings.speaker_anonymization import SpeakerAnonymization
import typing

logger = setup_logger(__name__)

class STTTSPipeline(Pipeline):
    def __init__(self, config: dict, force_compute: bool = False, devices: list = [0]):
        """
        Instantiates a STTTSPipeline with the complete feature extraction,
        modification and resynthesis.

        This pipeline consists of:
              - ASR -> phone sequence                    -
        input - (prosody extr. -> prosody anon.)         - TTS -> output
              - speaker embedding extr. -> speaker anon. -

        Args:
            config (dict): a configuration dictionary, e.g., see anon_ims_sttts_pc.yaml
            force_compute (bool): if True, forces re-computation of
                all steps. otherwise uses saved results.
            devices (list): a list of torch-interpretable devices
        """
        self.total_start_time = time.time()
        self.config = config
        model_dir = Path(config.get("models_dir", "models"))
        vectors_dir = Path(config.get("vectors_dir", "original_speaker_embeddings"))
        self.results_dir = Path(config.get("results_dir", "results"))
        self.intermediate_dir = Path(config.get("intermediate_dir", "exp/intermediate_dir_my"))
        self.data_dir = Path(config["data_dir"]) if "data_dir" in config else None
        self.anon_suffix = config.get("anon_suffix", "ims_sttts")
        save_intermediate = config.get("save_intermediate", True)

        modules_config = config["modules"]
        self.modules_config = config["modules"]

        # Text Extractor
        self.speech_recognition = SpeechRecognition(
            devices=devices,
            save_intermediate=save_intermediate,
            settings=modules_config["asr"],
            #force_compute=force_compute,
        )

        # Speaker component
        self.speaker_extraction = {}
        self.speaker_anonymization = {}
        self.speaker_extraction = SpeakerExtraction(
            devices=devices,
            save_intermediate=save_intermediate,
            settings=modules_config["speaker_embeddings"],
            #force_compute=force_compute,
        )
        # self.speaker_anonymization = SpeakerAnonymization(
        #     vectors_dir=vectors_dir,
        #     device=devices[0],
        #     save_intermediate=save_intermediate,
        #     settings=modules_config["speaker_embeddings"],
        #     force_compute=force_compute,
        # )

        # TTS component
        self.speech_synthesis = SpeechSynthesis(
            devices=devices,
            settings=modules_config["tts"],
            model_dir=model_dir,
            save_output=config.get("save_output", True),
            force_compute=force_compute,
        )

    def run_anonymization_pipeline(
        self,
        datasets: typing.Dict[str, Path],
    ):
        """
            Runs the anonymization algorithm on the given datasets. Optionally
            prepares the results such that the evaluation pipeline
            can interpret them.

            Args:
                datasets (dict of str -> Path): The datasets on which the
                    anonymization pipeline should be runned on. These dataset
                    will be processed sequentially.
        """
        anon_wav_scps = {}

        for i, (dataset_name, dataset_path) in enumerate(datasets.items()):
            logger.info(f"{i + 1}/{len(datasets)}: Processing {dataset_name}...")
            # Step 1: Extract Text, F0 Features
            start_time = time.time()

            texts = self.speech_recognition.recognize_speech(dataset_path=dataset_path, dataset_name=dataset_name)
            logger.info("--- Speech recognition (Text extraction) time: %f min ---" % (float(time.time() - start_time) / 60))

            start_time = time.time()
            f0_features = self.speaker_extraction.extract_f0_features(dataset_path=dataset_path,
                                                                      dataset_name=dataset_name)
            logger.info(f"--- F0 extraction time: {(float(time.time() - start_time) / 60)} min ---")

            # Step 2: Synthesize
            start_time = time.time()
            wav_scp = self.speech_synthesis.synthesize_speech(dataset_name=dataset_name, texts=texts,
                                                              speaker_embeddings=f0_features,
                                                              emb_level='utt')
            logger.info("--- Synthesis time: %f min ---" % (float(time.time() - start_time) / 60))
            
            output_path = Path(str(dataset_path) + self.anon_suffix)
            copy_data_dir(dataset_path, output_path)

            # Overwrite wav.scp with the paths to the anonymized wavs
            save_kaldi_format(wav_scp, output_path / 'wav.scp')

        logger.info("--- Total computation time: %f min ---" % (float(time.time() - self.total_start_time) / 60))

        return anon_wav_scps
