from typing import Union
from torch.utils.data import Dataset
from pathlib import Path

from utils import get_files_from_dir, get_audio_data


class SpeechTrainDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path],
                 split: str = 'noisy') \
            -> None:
        """Pytorch Dataset class for training samples."""

        self.root_dir = Path(root_dir)
        self.split = split
        self.load_data()

    def load_data(self) -> None:
        """Loads the data into memory."""

        noisy_train_dir = self.root_dir / (self.split + '_trainset_28spk_wav')
        clean_train_dir = self.root_dir / (self.split + '_trainset_28spk_wav')

        noisy_train_files = get_files_from_dir(noisy_train_dir)
        clean_train_files = get_files_from_dir(clean_train_dir)

        self.noisy_train = [get_audio_data(f) for f in noisy_train_files]
        self.clean_train = [get_audio_data(f) for f in clean_train_files]


    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.noisy_train)

    def __getitem__(self, idx):
        """Returns the item at index `idx`."""
        return self.noisy_train[idx], self.clean_train[idx]


class SpeechTestDataset(Dataset):
    def __init__(self, root_dir: Union[str, Path],
                 split: str = 'noisy') \
            -> None:
        """Pytorch Dataset class for testing samples. """
        self.root_dir = Path(root_dir)
        self.split = split
        self.load_data()

    def load_data(self) -> None:
        """Loads the data into memory."""

        noisy_test_dir = self.root_dir / (self.split + '_testset_wav')
        clean_test_dir = self.root_dir / (self.split + '_testset_wav')

        noisy_test_files = get_files_from_dir(noisy_test_dir)
        clean_test_files = get_files_from_dir(clean_test_dir)

        self.noisy_test = [get_audio_data(f) for f in noisy_test_files]
        self.clean_test = [get_audio_data(f) for f in clean_test_files]

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.noisy_test)
    
    def __getitem__(self, idx):
        """Returns the item at index `idx`."""
        return self.noisy_test[idx], self.clean_test[idx]
    

def test_data_handling():
    root_dir = "."
    
    # Test SpeechTrainDataset
    train_dataset = SpeechTrainDataset(root_dir, split="noisy")
    print(f"Train dataset length: {len(train_dataset)}")
    noisy_data, clean_data = train_dataset[0]
    print(f"First training sample noisy shape: {noisy_data[0].shape}, clean shape: {clean_data[0].shape}")
    
    # Test SpeechTestDataset
    test_dataset = SpeechTestDataset(root_dir, split="noisy")
    print(f"Test dataset length: {len(test_dataset)}")
    noisy_data, clean_data = test_dataset[0]
    print(f"First test sample noisy shape: {noisy_data[0].shape}, clean shape: {clean_data[0].shape}")

if __name__ == "__main__":
    test_data_handling()

