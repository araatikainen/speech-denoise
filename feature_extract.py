
from typing import List, Optional

import pathlib
import os
import numpy as np
import librosa

from utils import get_audio_files_from_subdirs, get_audio_file_data


def extract_mel_spectrogram(audio_signal: np.ndarray,
                        sr: int = 16000,
                        n_mels: int = 13,
                        n_fft: Optional[int] = 2048,
                        hop_length: Optional[int] = 1024,
                        window: Optional[str] = 'hamm') \
        -> np.ndarray:
    """Extracts and returns the mel spectrogram from the `audio_signal` signal.

    :param audio_signal: Audio signal.
    :type audio_signal: numpy.ndarray
    :param n_fft: STFT window length (in samples), defaults to 2048.
    :type n_fft: Optional[int]
    :param hop_length: Hop length (in samples), defaults to 1024.
    :type hop_length: Optional[int]
    :param window: Window type, defaults 'hamm'.
    :type window: Optional[str]
    :return: Mel spectrogram.
    :rtype: numpy.ndarray
    """

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_signal,
                                                    sr=sr,
                                                    n_mels=n_mels,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    window=window)

    # Return the Mel spectrogram
    return mel_spectrogram


def split_into_sequences(spec: np.ndarray, seq_len: int) \
        -> List[np.ndarray]:
    """Splits the spectrum `spec` into sequences of length `seq_len`.

    :param spec: Spectrum to be split into sequences.
    :type spec: numpy.ndarray
    :param seq_len: Length of the sequences.
    :type seq_len: int
    :return: List of sequences.
    :rtype: list[numpy.ndarray]
    """
    # split into sequences of length `seq_len`
    sequences = []
    for i in range(0, spec.shape[0] - seq_len + 1, seq_len):
        sequences.append(spec[i:i + seq_len])
    return sequences


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
                audio_paths = get_audio_files_from_subdirs(dataset_path)
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
                    y, sr = get_audio_file_data(audio_file)
                    mel_spec = extract_mel_spectrogram(y, sr)
                    np.save(output_dir / f"{audio_file.stem}.npy", mel_spec)
            else:
                print(f"Skipping missing directory: {dataset_path}")

if __name__ == '__main__':
    dataset_root_path = pathlib.Path() 
    main(dataset_root_path)