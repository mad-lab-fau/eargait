"""Dataset and DataModule classes for handling HAR data with PyTorch Lightning."""
import numpy as np
import pytorch_lightning as pl
import torch.utils.data
from torch.utils.data import DataLoader, Sampler


class HARDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict):
        # reshape data in to form (samples, 1, sensor channels, sequence)
        expanded_data = np.expand_dims(data["data"], axis=1)
        self.sensor_data = np.swapaxes(expanded_data, 2, 3)
        self.labels = data["labels"]

    def __len__(self):
        return len(self.sensor_data)

    def __getitem__(self, index: int):
        data = torch.Tensor(self.sensor_data[index])
        label = torch.tensor(self.labels[index])
        return {"data": data, "label": label}


class HARDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data: dict,
        validation_data: dict,
        test_data: dict,
        batch_size: int,
        shuffle: bool = False,
        sampler: Sampler = None,
        num_workers: int = 0,
    ):  # changes 1 ->0 bc of "RuntimeError: DataLoader worker (pid(s) 3260821) exited unexpectedly"
        super().__init__()
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_dataset = HARDataset(self.train_data)
        self.test_dataset = HARDataset(self.test_data)
        self.validation_dataset = HARDataset(self.validation_data)
        self.sampler = sampler
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
