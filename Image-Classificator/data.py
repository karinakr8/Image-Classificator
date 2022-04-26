import os
import random

import torch
import torchaudio
from torchaudio import transforms


class AudioDataset:

    def __init__(self, annotations_file, audio_dir, target_sample_rate, num_samples, duration, shift_limit, channel):
        with open(annotations_file) as f:
            self.annotations = f.readlines()
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.channel = channel
        self.duration = duration
        self.shift_limit = shift_limit

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_path = self._get_audio_path(index)
        label = self._get_audio_labelID(index)
        signal, sample_rate = torchaudio.load(audio_path)
        signal = self._rechannel(signal)
        signal = self._resample(signal, sample_rate)
        signal = self._truncate(signal, self.target_sample_rate)
        signal = self._time_shift(signal)  # testavimo duomenim nereikia
        signal = self._spectro_gram(signal, self.target_sample_rate)
        signal = self._spectro_augment(signal)  # testavimo duomenim nereikia
        return signal, label

    def _get_audio_path(self, index):
        index_slash = self.annotations[index].index('/')
        fold = self.annotations[index][:index_slash]
        path = os.path.join(self.audio_dir, fold,
                            self.annotations[index][index_slash + 1:-1])
        # print(torchaudio.info(path))
        return path

    def _get_audio_labelID(self, index):
        index_slash = self.annotations[index].index('/')
        for i, lbl in enumerate(self.unique()):
            if lbl == self.annotations[index][:index_slash]:
                return i
        return None

    def _rechannel(self, signal):
        if signal.shape[0] == self.channel:
            return signal
        if self.channel == 1:
            signal = signal[:1, :]
        else:
            signal = torch.cat([signal, signal])
        return signal

    def _resample(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _truncate(self, signal, sample_rate):
        num_rows, signal_len = signal.shape
        max_len = sample_rate // 1000 * self.duration

        if signal_len > max_len:
            signal = signal[:, :max_len]

        elif signal_len < max_len:
            pad_begin_len = random.randint(0, max_len - signal_len)
            pad_end_len = max_len - signal_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            signal = torch.cat((pad_begin, signal, pad_end), 1)
        return signal

    def _time_shift(self, signal):
        _, signal_len = signal.shape
        shift_amt = int(random.random() * self.shift_limit * signal_len)
        return signal.roll(shift_amt)

    def _spectro_gram(self, signal, sample_rate, n_mels=64, n_fft=1024, hop_len=None):
        spec = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(signal)
        spec = transforms.AmplitudeToDB(top_db=80)(spec)
        return spec

    def _spectro_augment(self, spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec

    def unique(self):
        classes = [
                    "bed",
                    "bird",
                    "cat",
                    "dog",
                    "down",
                    "eight",
                    "five",
                    "four",
                    "go",
                    "happy",
                    "house",
                    "left",
                    "marvin",
                    "nine",
                    "no",
                    "off",
                    "on",
                    "one",
                    "right",
                    "seven",
                    "sheila",
                    "six",
                    "stop",
                    "three",
                    "tree",
                    "two",
                    "up",
                    "wow",
                    "yes",
                    "zero"
                ]
        return classes
