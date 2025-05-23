from collections import defaultdict
from pathlib import Path
import torch

from utils import read_kaldi_format, save_kaldi_format, create_clean_dir


class SpeakerEmbeddings:

    def __init__(self, vec_type='xvector', emb_level='spk', device=torch.device('cpu')):
        self.vec_type = vec_type
        self.emb_level = emb_level
        self.device = device

        self.identifiers2idx = {}
        self.idx2identifiers = {}
        self.vectors = None
        self.original_speakers = []
        self.genders = []

        self.new = True

    def __iter__(self):
        assert self.identifiers2idx and self.vectors is not None, \
            'Speaker vectors need to be extracted or loaded before they can be iterated!'

        for identifier, idx in sorted(self.identifiers2idx.items(), key=lambda x: x[1]):
            yield identifier, self.vectors[idx]

    def __len__(self):
        return len(self.identifiers2idx)

    def __getitem__(self, item):
        assert (self.identifiers2idx is not None) and (self.vectors is not None), \
            'Speaker vectors need to be extracted or loaded before they can be accessed!'
        assert item <= len(self), 'Index needs to be smaller or equal the number of speakers!'
        return self.idx2identifiers[item], self.vectors[item]

    def add_vector(self, identifier, vector, speaker, gender):
        idx = len(self)
        if not self.vectors:
            self.vectors = torch.tensor(vector)
        else:
            self.vectors = torch.cat((self.vectors, vector), 0)
        self.identifiers2idx[identifier] = idx
        self.idx2identifiers[idx] = identifier
        self.original_speakers.append(speaker)
        self.genders.append(gender)

    def set_vectors(self, identifiers, vectors, speakers, genders):
        if not isinstance(identifiers, dict):
            self.identifiers2idx = {identifier: idx for idx, identifier in enumerate(identifiers)}
        else:
            self.identifiers2idx = identifiers
        self.vectors = torch.tensor(vectors) if not isinstance(vectors, torch.Tensor) else vectors
        self.genders = genders
        self.original_speakers = speakers
        self.idx2identifiers = {idx: identifier for identifier, idx in self.identifiers2idx.items()}

    def f0_add_vectors(self, identifiers, vectors, speakers, genders):
        if not isinstance(identifiers, dict):
            identifiers = {identifier: idx for idx, identifier in enumerate(identifiers)}

        new_identifiers = list(identifiers.keys() - self.identifiers2idx.keys())
        indices = [identifiers[iden] for iden in new_identifiers]
        last_known_index = len(self)

        new_iden_dict = {iden: last_known_index + i for i, iden in enumerate(new_identifiers)}
        self.identifiers2idx.update(new_iden_dict)
        self.idx2identifiers.update({idx: iden for iden, idx in new_iden_dict.items()})
        # if isinstance(vectors, list):
        #     vectors = torch.cat([v for v in vectors], dim=0).to(self.device)
        if self.vectors is None:
            self.vectors = [vectors[i] for i in indices]
            # self.vectors = torch.index_select(vectors.clone().detach(),
            #                        0,
            #                        torch.LongTensor(indices).clone().detach().to(self.device)).clone().detach()
        else:
            self.vectors.extend([vectors[i] for i in indices])

    def f0_load_vectors(self, in_dir: Path):
        assert (in_dir / f'id2idx').exists() and (in_dir / f'f0.pt').exists(), \
            f'speaker_vectors.pt and id2idx must exist in {in_dir}!'

        idx2spk = read_kaldi_format(in_dir / 'idx2spk')

        self.vectors = torch.load(in_dir / f'f0.pt', map_location=self.device)

        self.identifiers2idx = {id: int(idx) for id, idx in read_kaldi_format(in_dir / f'id2idx').items()}
        self.idx2identifiers = {idx: identifier for identifier, idx in self.identifiers2idx.items()}

        self.new = False

    def f0_save_vectors(self, out_dir: Path):
        assert (self.identifiers2idx is not None) and (self.vectors is not None), \
            'Speaker vectors need to be extracted or loaded before they can be stored!'
        create_clean_dir(out_dir)

        torch.save(self.vectors, out_dir / f'f0.pt')

    def get_embedding_for_identifier(self, identifier):
        idx = self.identifiers2idx[identifier]
        return self.vectors[int(idx)]


    def get_utt_list(self):
        return [identifier for identifier, idx in sorted(self.identifiers2idx.items(), key=lambda x: x[1])]

    def get_spk2gender(self):
        return {speaker: gender for speaker, gender in zip(self.original_speakers, self.genders)}

    def convert_to_spk_level(self, method='average'):
        assert self.emb_level == 'utt', \
            'Speaker embeddings must be on utterance level to be able to convert them to speaker level!'

        if method == 'average':
            spk2idx = defaultdict(list)
            for i, speaker in enumerate(self.original_speakers):
                spk2idx[speaker].append(i)

            spk_level_embeddings = SpeakerEmbeddings(vec_type=self.vec_type, emb_level='spk', device=self.device)
            spk_vectors, speakers, genders = [], [], []
            if not isinstance(self.vectors, torch.Tensor):
                self.vectors = torch.tensor(self.vectors)
            for speaker, idx_list in spk2idx.items():
                spk_vectors.append(torch.mean(self.vectors[idx_list], dim=0))
                speakers.append(speaker)
                genders.append(self.genders[idx_list[0]])
            spk_level_embeddings.set_vectors(identifiers=speakers, vectors=torch.stack(spk_vectors, dim=0),
                                             speakers=speakers, genders=genders)

            return spk_level_embeddings
        else:
            return self
