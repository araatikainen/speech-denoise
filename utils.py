from typing import List, Union, Tuple
from pathlib import Path
import pathlib
import os
import numpy as np
import librosa


__docformat__ = 'reStructuredText'
__all__ = [ 'get_files_from_dir_with_pathlib',
            'get_audio_files_from_subdirs',
            'get_audio_file_data',
           ]


def get_files_from_dir_with_pathlib(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the files in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return list(pathlib.Path(dir_name).iterdir())


def get_audio_files_from_subdirs(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the audio files in the subdirectories of `dir_name`.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the audio files in the subdirectories `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return [Path(dirpath) / Path(filename) for dirpath, _, filenames in os.walk(dir_name)
                                           for filename in filenames
                                           if filename[-4:] == '.wav']


def get_audio_file_data(audio_file: Union[str, pathlib.Path]) \
        -> Tuple[np.ndarray, float]:
    """Loads and returns the audio data from the `audio_file`.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str
    :return: Data of the `audio_file` audio file.
    :rtype: Tuple[numpy.ndarray, float]
    """
    # TODO add data normalization

    return librosa.core.load(path=audio_file, sr=None, mono=True)


def test_utils():
    # Test get_files_from_dir_with_pathlib
    test_dir = "noisy_testset_wav"
    files = get_files_from_dir_with_pathlib(test_dir)
    print(f"Files in {test_dir}: {files}")
    
    # Test get_audio_files_from_subdirs
    audio_files = get_audio_files_from_subdirs(test_dir)
    print(f"Audio files in {test_dir} subdirectories: {audio_files}")
    
    # Test get_audio_file_data
    if audio_files:
        audio_data, sr = get_audio_file_data(audio_files[0])
        print(f"Audio data for {audio_files[0]}: {audio_data.shape}, Sampling rate: {sr}")

if __name__ == "__main__":
    test_utils()

