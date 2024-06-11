from tqdm import tqdm
import soundfile
import time
from torch.multiprocessing import Pool, set_start_method
from itertools import repeat

from utils import create_clean_dir, setup_logger
from ..vits import utils as vit_utils
from ..vits.models import SynthesizerTrn
from ..vits.text.symbols import symbols
from ..vits.text import text_to_sequence
from ..vits import commons as vit_commons
import torch
import os

set_start_method('spawn', force=True)
logger = setup_logger(__name__)

class SpeechSynthesis:

    def __init__(self, devices, settings, model_dir=None, results_dir=None, save_output=True, force_compute=False):
        self.devices = devices


        self.output_sr = settings.get('output_sr', 16000)
        self.save_output = save_output
        self.force_compute = False
        self.hps = vit_utils.get_hparams_from_file("/user/ljs_base.json")
        # self.sr = 16000

        synthesizer_type = settings.get('synthesizer', 'ims')
        if synthesizer_type == 'vit':
            self.tts_models = []

            for device in self.devices:
                net_g = SynthesizerTrn(len(symbols),
                            self.hps.data.filter_length // 2 + 1,
                            self.hps.train.segment_size // self.hps.data.hop_length,
                                **self.hps.model)
                _ = vit_utils.load_checkpoint("/user/pretrained_ljs.pth", net_g, None)
                net_g.eval()  
                self.tts_models.append(net_g)

        if results_dir:
            self.results_dir = results_dir
        elif 'results_path' in settings:
            self.results_dir = settings['results_path']
        elif 'results_dir' in settings:
            self.results_dir = settings['results_dir']
        else:
            if self.save_output:
                raise ValueError('Results dir must be specified in parameters or settings!')

    def synthesize_speech(self, dataset_name, texts, speaker_embeddings, prosody=None, emb_level='utt'):
        # depending on whether we save the generated audios to disk or not, we either return a dict of paths to the
        # saved wavs (wav.scp) or the wavs themselves
        dataset_results_dir = self.results_dir / dataset_name if self.save_output else ''
        wavs = {}

        if dataset_results_dir.exists() and not self.force_compute:
            already_synthesized_utts = {wav_file.stem: str(wav_file.absolute())
                                        for wav_file in dataset_results_dir.glob('*.wav')
                                        if wav_file.stem in texts.utterances}

            if len(already_synthesized_utts):
                logger.info(f'No synthesis necessary for {len(already_synthesized_utts)} of {len(texts)} utterances...')
                texts.remove_instances(list(already_synthesized_utts.keys()))
                if self.save_output:
                    wavs = already_synthesized_utts
                else:
                    wavs = {}
                    for utt, wav_file in already_synthesized_utts.items():
                        if len(wav_file)==6:
                            wav, sr = soundfile.read(wav_file[4])
                        else:
                            wav, sr = soundfile.read(wav_file)
                        # wav, sr = soundfile.read(wav_file)
                        wavs[utt] = wav
                        self.sr = sr

        if texts:
            logger.info(f'Synthesize {len(texts)} utterances...')
            if self.force_compute or not dataset_results_dir.exists():
                create_clean_dir(dataset_results_dir)

            text_is_phones = texts.is_phones

            if len(self.tts_models) == 1:
                instances = []
                for text, utt, speaker in texts:
                    try:
                        if emb_level == 'spk':
                            speaker_embedding = speaker_embeddings.get_embedding_for_identifier(speaker)
                        else:
                            speaker_embedding = speaker_embeddings.get_embedding_for_identifier(utt)

                        if prosody:
                            utt_prosody_dict = prosody.get_instance(utt)
                        else:
                            utt_prosody_dict = {}
                        instances.append((text, utt, speaker_embedding, utt_prosody_dict))
                    except KeyError:
                        logger.warn(f'Key error at {utt}')
                        continue
                wavs.update(synthesis_job(instances=instances, tts_model=self.tts_models[0],
                                          out_dir=dataset_results_dir, sleep=0, text_is_phones=text_is_phones,
                                          save_output=self.save_output))

            else:
                num_processes = len(self.tts_models)
                sleeps = [10 * i for i in range(num_processes)]
                text_iterators = texts.get_iterators(n=num_processes)

                instances = []
                for iterator in text_iterators:
                    job_instances = []
                    for text, utt, speaker in iterator:
                        try:
                            if emb_level == 'spk':
                                speaker_embedding = speaker_embeddings.get_embedding_for_identifier(speaker)
                            else:
                                speaker_embedding = speaker_embeddings.get_embedding_for_identifier(utt)
                                speaker_embedding = speaker_embedding.detach()

                            job_instances.append((text, utt, speaker_embedding))
                        except KeyError:
                            logger.warn(f'Key error at {utt}')
                            continue
                    instances.append(job_instances)

                # multiprocessing
                with Pool(processes=num_processes) as pool:
                    job_params = zip(instances, range(len(self.tts_models)), self.devices, repeat(dataset_results_dir), sleeps,
                        repeat(text_is_phones), repeat(self.save_output), repeat(self.hps))
                    new_wavs = tqdm(pool.starmap(synthesis_job, job_params))

                for new_wav_dict in new_wavs:
                    wavs.update(new_wav_dict)
        return wavs

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = vit_commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def synthesis_job(instances, model_index, device, out_dir, sleep, text_is_phones=False, save_output=False, hps = None):
    net_g = SynthesizerTrn(len(symbols),
                            hps.data.filter_length // 2 + 1,
                            hps.train.segment_size // hps.data.hop_length,
                                **hps.model)
    _ = vit_utils.load_checkpoint("/user/pretrained_ljs.pth", net_g, None)
    net_g = net_g.to(device)
    _ = net_g.eval()  

    wavs = {}
    for text, utt, speaker_embedding in tqdm(instances):
        
        stn_tst = get_text(text, hps)
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        speaker_embedding[torch.isnan(speaker_embedding)] = 0
        wav = net_g.infer(device, x_tst, x_tst_lengths, speaker_embedding, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
   
        if save_output:
            out_file = str((out_dir / f'{utt}.wav').absolute())
            soundfile.write(file=out_file, data=wav, samplerate=16000)
            wavs[utt] = out_file
        else:
            wavs[utt] = wav
    return wavs