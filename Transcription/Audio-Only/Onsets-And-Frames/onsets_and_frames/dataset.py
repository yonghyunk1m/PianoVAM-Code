# [Source] https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/dataset.py

import json
import os
from abc import abstractmethod
from glob import glob

import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py

from .config import *
from .midi import parse_midi

# PIANOVAM_AUDIO
class PianoRollAudioDataset(Dataset):
    def __init__(self, path, groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)

        self.data = []
        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]
        result = dict(path=data['path'])
        audio_length = len(data['audio'])

        if self.sequence_length is not None:
            step_begin = self.random.randint(audio_length - self.sequence_length) // HOP_LENGTH
            n_steps = self.sequence_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length

        else: # evaluate.py (Entire Song)
            begin = 0
            end = audio_length
            step_begin = 0
            step_end = end // HOP_LENGTH
            
        audio_filtered = torch.tensor(data['audio'][begin:end], dtype=torch.float32)
        
        result['audio'] = torch.tensor(audio_filtered, dtype=torch.float32)
        result['label'] = data['label'][step_begin:step_end, :].clone().detach().float()
        result['velocity'] = data['velocity'][step_begin:step_end, :].clone().detach().float()

        result['audio'] = result['audio'].float().div_(32768.0)
        result['onset'] = (result['label'] == 3).float()
        result['frame'] = ((result['label'] == 3) | (result['label'] == 2)).float()
        result['offset'] = (result['label'] == 1).float()
        result['velocity'] = result['velocity'].float().div_(128.0)
        
        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        raise NotImplementedError

    @abstractmethod 
    def files(self, group):
        raise NotImplementedError

    def load(self, hdf5_path, tsv_path):
        with h5py.File(hdf5_path, "r") as hf:
            audio = hf["waveform"][:]
        
        audio_length = len(audio)
        
        n_keys = MAX_MIDI - MIN_MIDI + 1 
        n_steps = (audio_length - 1) // HOP_LENGTH + 1

        label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
        velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

        midi_data = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
        for onset, key_offset, ped_offset, note, vel in midi_data:

            onset_start_index = int(round((onset * SAMPLE_RATE) / HOP_LENGTH))
            onset_end_index = min(onset_start_index + HOPS_IN_ONSET, n_steps)
            
            ped_offset_start_index = int(round((ped_offset * SAMPLE_RATE) / HOP_LENGTH))
            ped_offset_end_index = min(ped_offset_start_index + HOPS_IN_OFFSET, n_steps)
            
            f = int(note) - MIN_MIDI  # Convert MIDI to index
            label[onset_start_index:onset_end_index, f] = 3 # ONSET
            label[onset_end_index: ped_offset_start_index, f] = 2 # FRAME - ONSET
            label[ped_offset_start_index:ped_offset_end_index, f] = 1 # OFFSET
            velocity[onset_start_index:ped_offset_start_index, f] = vel # Velocity (Onset~Before Frame Offset)
        
        # Revised Audio and Label will be sent.
        return dict(
            path=hdf5_path,
            audio=audio,
            label=label,
            velocity=velocity
        )


# MAESTRO v.3
class MAESTRO(PianoRollAudioDataset):

    def __init__(self, path='/media/backup_SSD/Yonghyun/RobustAMT/data', groups=None, sequence_length=None, seed=42, device=DEFAULT_DEVICE):
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, device)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def read_maestro_metadata(self, json_path):
        with open(json_path, "r") as f:
            metadata = json.load(f)

        split_dict = {group: [] for group in ['train', 'validation', 'test']}

        for idx, split in metadata['split'].items():
            hdf5_filename = metadata['audio_filename'][idx].replace('.wav', '.h5').replace('.flac', '.h5')
            hdf5_path = os.path.join(self.path, 'maestro/workspace/hdf5s/maestro-v3.0.0', hdf5_filename)
            midi_path = os.path.join(self.path, 'maestro/maestro-v3.0.0', metadata['midi_filename'][idx])
            tsv_path = midi_path.replace('maestro-v3.0.0', 'tsv').replace('.midi', '.tsv').replace('.mid', '.tsv')

            split_dict[split].append((hdf5_path, tsv_path))

        return split_dict

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        metadata_path = os.path.join(self.path, 'maestro/maestro-v3.0.0', 'maestro-v3.0.0.json')
        split_dict = self.read_maestro_metadata(metadata_path)

        if group not in split_dict:
            raise ValueError(f"Group {group} is invalid.")

        files = split_dict[group]
        if not files:
            raise ValueError(f"No files found for group '{group}' in metadata.json")
        
        return files



class PIANOVAM_AUDIO(PianoRollAudioDataset):
    @classmethod
    def available_groups(cls): return ['train', 'valid', 'test']

    def read_pianovam_metadata(self, json_path):
        with open(json_path, "r") as f:
            metadata = json.load(f)

        split_dict = {"train": [], "valid": [], "test": []}

        for entry in metadata.values():
            record_time = entry["record_time"]  # e.g., "2024-02-14_19-10-09"
            split = entry["split"]  # train/valid/test

            hdf5_path = os.path.join("/media/backup_SSD/Yonghyun/RobustAMT/data/pianovam/workspace/hdf5s/pianovam/30FPS", f"{record_time}.h5")
            tsv_path = os.path.join("/media/backup_SSD/Yonghyun/RobustAMT/data/pianovam/tsv", f"{record_time}.tsv")

            if os.path.exists(hdf5_path) and os.path.exists(tsv_path):
                split_dict[split].append((hdf5_path, tsv_path))

        return split_dict

    def files(self, group):
        metadata_path = os.path.join(self.path, "pianovam/metadata.json")
        split_dict = self.read_pianovam_metadata(metadata_path)

        if group not in split_dict:
            raise ValueError(f"Invalid group '{group}'. Must be one of {list(split_dict.keys())}")

        file_list = split_dict[group]
        if not file_list:
            raise ValueError(f"No files found for group '{group}' in metadata.json")

        return file_list
