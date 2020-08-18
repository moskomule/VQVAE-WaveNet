import random
from typing import Tuple, Callable

import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchaudio import datasets
from torchaudio import functional as AF

torchaudio.set_audio_backend("sox_io")


def get_dataloader(name: str,
                   batch_size: int,
                   train_portion: float,
                   sample_rate: int,
                   top_db: int,
                   length: int,
                   download: bool = False
                   ) -> Tuple[DataLoader, DataLoader]:
    if name.lower() == "vctk":
        dataset = datasets.VCTK("~/.torch/data/vctk", download=download,
                                transform=get_transform(sample_rate, top_db, length))
    else:
        raise NotImplementedError
    train_size = int(len(dataset) * train_portion)
    # for
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    return train_loader, val_loader


def digitize(x: Tensor,
             bins: Tensor,
             out_int32: bool
             ) -> Tensor:
    # PyTorch version of digitize
    return torch.searchsorted(bins, x, out_int32=out_int32)


def get_transform(sample_rate,
                  top_db,
                  length,
                  ) -> Callable[[Tensor], Tensor]:
    def transform(waveform: Tensor
                  ) -> Tensor:
        # waveform is already normalized
        waveform = AF.vad(waveform, sample_rate=sample_rate, trigger_level=top_db)
        wave_length = waveform.size(0)
        if wave_length < length:
            waveform = F.pad(waveform, [0, length - wave_length])
        else:
            start = random.randrange(wave_length - length - 1)
            waveform = waveform[start: start + length]
        return waveform

    return transform
