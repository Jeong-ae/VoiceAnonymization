import shutil

from tqdm import tqdm
from pathlib import Path
import torch
import torchaudio
from tqdm.contrib.concurrent import process_map
import time
from torch.multiprocessing import set_start_method
from itertools import repeat

from .speechbrain_vectors import SpeechBrainVectors
from .utils import normalize_wave
from .speaker_embedding import SpeakerEmbeddings
from utils import read_kaldi_format, remove_contents_in_dir, setup_logger
import librosa
from multiprocessing import Pool


set_start_method('spawn', force=True)
logger = setup_logger(__name__)

def extract_f0(device, y, sr=16000):
    y = y.cpu().detach().numpy()
    f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 =torch.from_numpy(f0).to(device)
    return f0
    
def extraction_job(params):
    # utt_info, speaker_extractors, sleep, device, vec_type, out_dir, save_intermediate, job_id = params
    utt_info, sleep, device, vec_type, out_dir, save_intermediate, job_id = params
    time.sleep(sleep)

    speaker_embeddings = SpeakerEmbeddings(vec_type=vec_type, emb_level='utt', device=device)

    add_suffix = f'_{job_id}' if job_id is not None else ''
    out_dir = out_dir / f'temp{add_suffix}'

    vectors = []
    utts = []
    speakers = []
    genders = []
    i = 0
    
    for utt, info in tqdm(utt_info.items(), desc=f'Job {job_id or 0}', leave=True):
        wav_path = info['path']
        if isinstance(wav_path, list):
            wav_path = wav_path[4]
        signal, fs = torchaudio.load(wav_path) #signal [1,31129]
        # if len(signal.shape) == 2:
        #     signal = signal.squeeze(0)
        norm_wave = normalize_wave(signal, fs, device=device) #norm_wave [31129]

        try:
            f0 = extract_f0(device, norm_wave, fs)
        except RuntimeError as e:
            logger.warning(f'Runtime error: {utt}, {signal.shape}, {norm_wave.shape}')
            raise e

        if len(f0.shape) == 1:
            vector = f0 #f0[0]
        else:
            vector = torch.cat(f0, dim=0)
        vectors.append(vector)
        utts.append(utt)
        speakers.append(info['spk'])
        genders.append(info['gender'])

        i += 1
        if i % 100 == 0:
            speaker_embeddings.add_vectors(vectors=vectors, identifiers=utts, speakers=speakers, genders=genders)
            vectors, utts, speakers, genders = [], [], [], []
            if save_intermediate:
                speaker_embeddings.save_vectors(out_dir)

    if vectors:
        speaker_embeddings.add_vectors(vectors=vectors, identifiers=utts, speakers=speakers, genders=genders)
        if save_intermediate:
            speaker_embeddings.save_vectors(out_dir)

    return speaker_embeddings


class SpeakerExtraction:

    def __init__(self, devices: list, settings: dict, results_dir: Path = None, model_dir: Path = None,
                 save_intermediate=True, force_compute=False):
        self.devices = devices
        self.n_processes = len(self.devices)
        self.save_intermediate = save_intermediate
        self.force_compute = force_compute if force_compute else settings.get('force_compute_extraction', False)

        # self.emb_model_path = settings['emb_model_path']
        self.vec_type = 'f0'
        self.emb_level = 'utt'

        if results_dir:
            self.results_dir = results_dir
        elif 'extraction_results_path' in settings:
            self.results_dir = settings['extraction_results_path']
        elif 'results_dir' in settings:
            self.results_dir = settings['results_dir']
        else:
            if self.save_intermediate:
                raise ValueError('Results dir must be specified in parameters or settings!')

        # model_dir = Path(model_dir or self.emb_model_path)
        # if not model_dir.exists():
        #     raise FileNotFoundError(f'ASV model {model_dir} does not exist!')

        # if not (model_dir / "hyperparams.yaml").exists():
        #     raise FileNotFoundError(f'ASV model {model_dir / "hyperparams.yaml"} does not exist!')

        # self.model_hparams = {
        #     'vec_type': self.vec_type,
        #     'model_path': model_dir or self.emb_model_path,
        # }

        self.extractors = [self._create_extractors(device=device) for device in devices]

    def extract_f0_features(self, dataset_path, dataset_name=None, emb_level=None):
        dataset_name = dataset_name if dataset_name is not None else dataset_path.name
        dataset_results_dir = self.results_dir / dataset_name if self.save_intermediate else Path('')
        emb_level = 'utt'
        utt2spk = read_kaldi_format(dataset_path / 'utt2spk')

        # we have to extract on utt level first and convert it to something else later
        if emb_level == 'spk':
            final_results_dir = dataset_results_dir / 'spk-level'
            utt_level_results_dir = dataset_results_dir / 'utt-level'
        else:
            final_results_dir = dataset_results_dir / 'utt-level'
            utt_level_results_dir = dataset_results_dir / 'utt-level'

        if self.force_compute:
            speaker_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, emb_level='utt', device=self.devices[0])
            missing_utterances = list(utt2spk.keys())
        else:
            speaker_embeddings, missing_utterances = self._get_already_extracted_speakers(
                utterances=list(utt2spk.keys()), emb_level=emb_level, final_results_dir=final_results_dir,
                utt_level_results_dir=utt_level_results_dir)

        if len(missing_utterances) > 0:
            logger.info(f'Extract embeddings of {len(missing_utterances)} utterances')
            speaker_embeddings.new = True
            wav_scp = read_kaldi_format(dataset_path / 'wav.scp')
            spk2gender = read_kaldi_format(dataset_path / 'spk2gender')
            # sometimes an utterance is skipped during synthesis
            previous_num_missing_utterances = len(missing_utterances)
            missing_utterances = [utt for utt in missing_utterances if utt in wav_scp.keys()]
            if len(missing_utterances) < previous_num_missing_utterances:
                logger.info(f'Skip {previous_num_missing_utterances - len(missing_utterances)} utterances because they do not have corresponding audios')
            if len(missing_utterances) == 0:
                return speaker_embeddings

            utt_info = {utt: {'path': wav_scp[utt], 'spk': utt2spk[utt], 'gender': spk2gender[utt2spk[utt]]}
                        for utt in missing_utterances}

            if self.n_processes > 1:
                sleeps = [10 * i for i in range(self.n_processes)]
                utt_info_jobs = [{k: v for k, v in list(utt_info.items())[i::self.n_processes]}
                                 for i in range(self.n_processes)]
                # multiprocessing
                params = zip(utt_info_jobs,  # utterances to extract speaker emb from
                            #  self.extractors, # extractors to use for extraction
                             sleeps, # avoid starting all processes at same time
                             self.devices, # device for each process
                             repeat(self.vec_type),  # which vec_type to use for extraction
                             repeat(utt_level_results_dir),  # where to store utt level results
                             repeat(self.save_intermediate), # whether to save intermediate results
                             list(range(self.n_processes)))  # job_id
                with Pool(processes=self.n_processes) as pool:
                    job_spk_embeddings = pool.map(extraction_job, params)
                # job_spk_embeddings = process_map(extraction_job, params, max_workers=self.n_processes)
            else:
                params = [utt_info, 0, self.devices[0], self.vec_type, utt_level_results_dir,
                          self.save_intermediate, None]
                job_spk_embeddings = [extraction_job(params)]

            speaker_embeddings = self._combine_speaker_embeddings(main_emb_instance=speaker_embeddings,
                                                                  additional_emb_instances=job_spk_embeddings)

            if self.save_intermediate:
                speaker_embeddings.save_vectors(out_dir=utt_level_results_dir)
                self._remove_temp_files(out_dir=utt_level_results_dir)
            if emb_level == 'spk':
                speaker_embeddings = speaker_embeddings.convert_to_spk_level()
                speaker_embeddings.save_vectors(out_dir=final_results_dir)

        return speaker_embeddings

    def _get_already_extracted_speakers(self, utterances, emb_level, final_results_dir, utt_level_results_dir):
        missing_utterances = []
        intermediate_results_dirs = list(utt_level_results_dir.glob('temp*'))
        device = self.devices[0]
        if (final_results_dir / 'speaker_vectors.pt').exists():
            speaker_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, emb_level=emb_level, device=device)
            speaker_embeddings.load_vectors(final_results_dir)
            # if the extraction is something else than utt-level and a final results dir exists, we assume that it is complete
            # if the extraction is on utt-level, we have to check that we extracted embeddings of all utterances
            if emb_level == 'utt':
                missing_utterances = list(set(utterances) - set(speaker_embeddings.identifiers2idx.keys()))
        elif (utt_level_results_dir / 'speaker_vectors.pt').exists():  # only possible if utt_level_results_dir != final_results_dir
            speaker_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, emb_level='utt', device=device)
            speaker_embeddings.load_vectors(utt_level_results_dir)
            missing_utterances = list(set(utterances) - set(speaker_embeddings.identifiers2idx.keys()))
            if len(missing_utterances) == 0 and emb_level == 'spk':
                speaker_embeddings = speaker_embeddings.convert_to_spk_level()
                speaker_embeddings.save_vectors(final_results_dir)
        elif len(intermediate_results_dirs) > 0:
            int_speaker_embeddings = []
            for intermediate_results_dir in intermediate_results_dirs:
                if (intermediate_results_dir / 'speaker_vectors.pt').exists():
                    emb = SpeakerEmbeddings(vec_type=self.vec_type, emb_level='utt', device=device)
                    emb.load_vectors(intermediate_results_dir)
                    int_speaker_embeddings.append(emb)
            if len(int_speaker_embeddings) > 0:
                speaker_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, emb_level='utt', device=device)
                speaker_embeddings = self._combine_speaker_embeddings(main_emb_instance=speaker_embeddings,
                                                                      additional_emb_instances=int_speaker_embeddings)
                missing_utterances = list(set(utterances) - set(speaker_embeddings.identifiers2idx.keys()))
                if len(missing_utterances) == 0:
                    speaker_embeddings.save_vectors(utt_level_results_dir)
            else:
                speaker_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, emb_level='utt', device=device)
                missing_utterances = utterances
        else:
            speaker_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, emb_level='utt', device=device)
            missing_utterances = utterances

        if len(missing_utterances) == 0:
            logger.info('No speaker extraction necessary; load existing embeddings instead...')
        elif len(missing_utterances) < len(utterances):
            logger.info( f'No speaker extraction necessary for {len(utterances) - len(missing_utterances)} of {len(utterances)} utterances...')

        return speaker_embeddings, missing_utterances
    


    def _create_extractors(self, device):
        extractors = []
        def extract_f0(y, sr=16000):
            y = y.cpu().detach().numpy()
            f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0 =torch.from_numpy(f0).to(device)
            return f0
        if self.vec_type =='f0':
            extractors.append(lambda y, sr=16000: extract_f0(y, sr))
        else:
            raise ValueError(f'Invalid vector type {self.vec_type}!')

        return extractors

    def _combine_speaker_embeddings(self, main_emb_instance, additional_emb_instances):
        for add_emb_instance in additional_emb_instances:
            if len(add_emb_instance) == 0:
                continue
            identifiers = [add_emb_instance.idx2identifiers[i] for i in range(len(add_emb_instance))]
            main_emb_instance.add_vectors(identifiers=identifiers, vectors=add_emb_instance.vectors,
                                          speakers=add_emb_instance.original_speakers, genders=add_emb_instance.genders)
        return main_emb_instance

    def _remove_temp_files(self, out_dir):
        for temp_dir in out_dir.glob('temp*'):
            remove_contents_in_dir(temp_dir)
            shutil.rmtree(temp_dir)

