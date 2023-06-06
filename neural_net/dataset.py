import torch
import pandas as pd
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torch.nn.functional as F
from torchvision.transforms import Resize
import torchaudio
import random

class AudioTransforms():
      
    @staticmethod
    def open(audio_file):
        signal, sample_rate = torchaudio.load(audio_file)
        return signal, sample_rate

    @staticmethod
    def rechannel(signal, num_channels):

        if signal.shape[0] == num_channels:
            return signal

        if num_channels == 1:
            signal = signal[:1, :]
        else:
            signal = torch.cat([signal, signal])

        return signal

    @staticmethod
    def resample(signal, sample_rate, new_sample_rate):

        if sample_rate == new_sample_rate:
            return signal
        
        num_channels = signal.shape[0]

        if num_channels == 1:
            signal = T.Resample(sample_rate, new_sample_rate)(signal)
            
            return signal

        if num_channels == 2:
            channel_one = T.Resample(sample_rate, new_sample_rate)(signal[0, :])
            channel_two = T.Resample(sample_rate, new_sample_rate)(signal[1, :])
            signal = torch.cat([channel_one, channel_two])

            return signal

    @staticmethod
    def resize(signal, num_samples, pad_mode='constant'):
        signal_length = signal.shape[1]
        if signal_length > num_samples:
            signal = signal[:, :num_samples]
            return signal
   
        if signal_length < num_samples:
            num_missing_samples = num_samples - signal_length
            last_dim_padding = (0, num_missing_samples)
            signal = F.pad(signal, last_dim_padding, mode=pad_mode)
        
        return signal

    @staticmethod
    def time_shift(signal, shift_limit):
        signal_length = signal.shape[1]
        shift = int(random.random() * shift_limit * signal_length)
        signal = signal.roll(shift)

        return signal
    
    @staticmethod
    def mel_spectogram(signal, sample_rate, n_mels=64, n_fft=1024, hop_length=512, top_db=80):
        spectogram = T.MelSpectrogram(sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length)(signal)
        spectogram = T.AmplitudeToDB(top_db=top_db)(spectogram)

        return spectogram
    
    @staticmethod
    def spectogram_augmentation(spectogram, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spectogram.shape
        mask_value = spectogram.mean()

        freq_mask_param = max_mask_pct * n_mels
        for _ in range (n_freq_masks):
            spectogram = T.FrequencyMasking(freq_mask_param)(spectogram, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            spectogram = T.TimeMasking(time_mask_param)(spectogram, mask_value)

        return spectogram
    
    @staticmethod
    def reshape(spectogram, shape):
        spectogram = Resize(shape, antialias=True)(spectogram)

        return spectogram




class WakeWordData(Dataset):
    def __init__(self, data_json, target_sample_rate, target_num_samples, validation=False, target_shape=(64, 64), num_channels=1):
        self.data = pd.read_json(data_json, lines=True)
        self.target_sample_rate = target_sample_rate
        self.target_num_samples = target_num_samples
        self.num_channels = num_channels
        self.target_shape = target_shape
        self.validation = validation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        try:    
            path = self.data.key.iloc[idx]
            label = self.data.label.iloc[idx]
            signal, sample_rate = AudioTransforms.open(path)
            signal = AudioTransforms.resample(signal, sample_rate, self.target_sample_rate)
            signal = AudioTransforms.rechannel(signal, self.num_channels)
            signal = AudioTransforms.resize(signal, self.target_num_samples)
            spectogram = AudioTransforms.mel_spectogram(signal, sample_rate)
            spectogram = AudioTransforms.reshape(spectogram, self.target_shape)
            if self.validation == False:
                spectogram = AudioTransforms.spectogram_augmentation(spectogram)
            return spectogram, label

        except Exception as e:
            print(str(e), path)
            return self.__getitem__(torch.randint(0, len(self), (1,)))

    

    

