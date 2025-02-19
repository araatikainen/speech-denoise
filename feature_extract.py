import pathlib
import os
import numpy as np
import librosa
from typing import Optional
from utils import get_audio_files, get_audio_data


def extract_mel_spectrogram(audio_signal: np.ndarray,
                        sr: int = 16000,
                        n_mels: int = 13,
                        n_fft: Optional[int] = 2048,
                        hop_length: Optional[int] = 1024,
                        window: Optional[str] = 'hamm') \
        -> np.ndarray:
    """Extracts and returns the mel spectrogram from the `audio_signal` signal."""

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_signal,
                                                    sr=sr,
                                                    n_mels=n_mels,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    window=window)

    # Return the Mel spectrogram
    return mel_spectrogram


def main(dataset_root: pathlib.Path):
    splits = ["trainset_28spk_wav", "testset_wav"]
    prefixes = ["clean_", "noisy_"]
    
    # Extract features for each split
    for prefix in prefixes:
        for split in splits:
            # Get the dataset path
            dataset_path = dataset_root / f"{prefix}{split}"

            # Check if the dataset path exists and is a directory
            if dataset_path.exists() and dataset_path.is_dir():
                audio_paths = get_audio_files(dataset_path)
                output_dir = dataset_root / f"{prefix}{split}_features"
                # Overwrite existing feature files
                if output_dir.exists():
                    for file in output_dir.iterdir():
                        file.unlink()  # Remove existing feature files
                else:
                    os.mkdir(output_dir)
                
                # Extract features for each audio file
                print(f'Extracting features from {dataset_path} to {output_dir}')
                for audio_file in audio_paths:
                    y, sr = get_audio_data(audio_file)
                    mel_spec = extract_mel_spectrogram(y, sr)
                    np.save(output_dir / f"{audio_file.stem}.npy", mel_spec)
            else:
                print(f"Skipping missing directory: {dataset_path}")

if __name__ == '__main__':
    dataset_root_path = pathlib.Path() 
    main(dataset_root_path)