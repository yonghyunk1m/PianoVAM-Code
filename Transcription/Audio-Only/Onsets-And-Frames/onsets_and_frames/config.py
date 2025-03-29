# [Source 1] https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/constants.py
# [Source 2] https://github.com/jongwook/onsets-and-frames/blob/master/train.py

SAMPLE_RATE = 16000
HOP_LENGTH = SAMPLE_RATE * 32 // 1000
ONSET_LENGTH = SAMPLE_RATE * 32 // 1000
OFFSET_LENGTH = SAMPLE_RATE * 32 // 1000
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH
MIN_MIDI = 21
MAX_MIDI = 108

N_MELS = 229
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048

ACCUMULATION_STEPS = 1

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
ex = Experiment('train_transcriber')

data_path = '/media/backup_SSD/Yonghyun/RobustAMT/data'

import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import wandb

@ex.config
def config():
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 150000 * ACCUMULATION_STEPS
    resume_iteration = None
    checkpoint_interval = 10000 * ACCUMULATION_STEPS
    train_on = 'MAESTRO' # PIANOVAM_AUDIO | MAESTRO

    batch_size = 8
    sequence_length = 327680
    model_complexity = 48

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000 * ACCUMULATION_STEPS
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 100 * ACCUMULATION_STEPS

    ex.observers.append(FileStorageObserver.create(logdir))
    
    wandb.init(project="Onsets-And-Frames", 
        name=f"{os.path.basename(logdir)}", 
        config={
        "iterations": iterations,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "sequence_length (validation_length)": sequence_length,
        "hop_length": HOP_LENGTH,
        "window_size": WINDOW_LENGTH,
        "n_mels": N_MELS,
        "model_complexity": model_complexity,
        "train_on": train_on,
        "validation_interval": validation_interval,
        "checkpoint_interval": checkpoint_interval,
        "accumulation_steps": ACCUMULATION_STEPS,
        "data_augmentation": "None" # "None" | "White Noise Injection"
    })
