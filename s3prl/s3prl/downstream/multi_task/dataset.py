import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F

from ..vocalset_singer_id.dataset import SingerDataset
from ..fluent_commands.dataset import FluentCommandsDataset
#from ..vocalset_technique_id.dataset import TechniqueDataset


class MultiTaskDataset(Dataset):
    """
    Dataset class that combines singer identification and fluent commands tasks
    with mixed audio inputs
    """

    def __init__(self, singer_root_dir, singer_meta_data, fluent_df, fluent_base_path, Sy_intent, split, upstream, features_path):
        self.singer_dataset = SingerDataset(
            singer_root_dir,
            singer_meta_data,
            split,
            upstream=upstream,
            features_path=features_path,
        )
        self.fluent_dataset = FluentCommandsDataset(
            fluent_df,
            fluent_base_path,
            Sy_intent,
            upstream=upstream,
            features_path=features_path,
        )
        
        # Store dataset sizes
        self.singer_size = len(self.singer_dataset)
        self.fluent_size = len(self.fluent_dataset)
        
        # Audio processing parameters
        self.sample_rate = 16000  # Standard sample rate
        self.sample_duration = 3  # 3 seconds of audio
        self.sample_length = self.sample_rate * self.sample_duration

    def __len__(self):
        # Return the size of the larger dataset
        return max(self.singer_size, self.fluent_size)

    def load_and_process_audio(self, audio_path, is_singer=True):
        """Load and process audio file with resampling and duration adjustment"""
        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        audio = wav.squeeze()

        # Handle audio duration
        if audio.shape[0] <= self.sample_length:
            # Pad if shorter
            audio = F.pad(audio, (0, self.sample_length - audio.shape[0]), 'constant', 0)
        else:
            # Random crop if longer
            random_start = np.random.randint(0, audio.shape[0] - self.sample_length)
            audio = audio[random_start:random_start + self.sample_length]

        return audio

    def mix_audio(self, singer_audio, fluent_audio):
        """Mix two audio signals with emphasis on fluent speech data"""
        # Normalize both signals
        singer_audio = singer_audio / torch.max(torch.abs(singer_audio))
        fluent_audio = fluent_audio / torch.max(torch.abs(fluent_audio))
        
        # Calculate RMS energy
        singer_rms = torch.sqrt(torch.mean(singer_audio ** 2))
        fluent_rms = torch.sqrt(torch.mean(fluent_audio ** 2))
        
        # Target speech-to-singing ratio (in dB) - increased to emphasize speech more
        target_ratio_db = np.random.uniform(4, 7)  # Speech 6-9dB louder than singing
        target_ratio = 10 ** (target_ratio_db / 20)
        
        # Calculate scaling factors to achieve target ratio
        current_ratio = fluent_rms / (singer_rms + 1e-6)
        if current_ratio < target_ratio:
            # Need to increase speech energy
            fluent_scale = target_ratio / current_ratio
            singer_scale = 0.5  # Reduce singer audio energy
        else:
            # Need to decrease singing energy
            singer_scale = 0.5  # Keep singer audio low
            fluent_scale = 1.0
        
        # Apply scaling with some random variation
        fluent_scale *= np.random.uniform(0.95, 1.05)  # Less variation for speech
        singer_scale *= np.random.uniform(0.8, 1.2)    # More variation for singing
        
        # Mix the signals
        mixed_audio = singer_audio * singer_scale + fluent_audio * fluent_scale
        
        # Normalize the mixed signal to prevent clipping
        max_val = torch.max(torch.abs(mixed_audio))
        if max_val > 0.95:  # If close to clipping
            mixed_audio = mixed_audio * (0.95 / max_val)
        
        return mixed_audio

    def __getitem__(self, idx):
        # Get indices for both datasets
        singer_idx = idx % self.singer_size
        fluent_idx = idx % self.fluent_size
        
        # Get data from both datasets
        singer_data = self.singer_dataset[singer_idx]
        fluent_data = self.fluent_dataset[fluent_idx]
        
        # Load audio files
        singer_audio_path = os.path.join(self.singer_dataset.audio_dir, "audio", singer_data[2])
        fluent_audio_path = os.path.join(self.fluent_dataset.base_path, self.fluent_dataset.df.loc[fluent_idx].path)
        
        singer_audio = self.load_and_process_audio(singer_audio_path, is_singer=True)
        fluent_audio = self.load_and_process_audio(fluent_audio_path, is_singer=False)
        
        # Mix the audio signals
        mixed_audio = self.mix_audio(singer_audio, fluent_audio)
        
        # Combine the data
        return {
            'features': mixed_audio,  # Mixed audio features
            'labels': {
                'singer': singer_data[1],  # Singer label
                'fluent': fluent_data[1]   # Fluent commands label
            },
            'filename': f"{singer_data[2]}_{fluent_data[2]}"  # Combined filename
        }

    def collate_fn(self, samples):
        features = [sample['features'] for sample in samples]
        labels = {
            'singer': [sample['labels']['singer'] for sample in samples],
            'fluent': [sample['labels']['fluent'] for sample in samples]
        }
        filenames = [sample['filename'] for sample in samples]
        
        return features, labels, filenames 
