import os

import lightning as L

from datasets import (
    load_dataset,
)

import torch
from torch.utils.data import (
    DataLoader,
)

import fairseq2


class DataModule(L.LightningDataModule):
    """
    Known issues:
    1. Due to the large sample size, we do not use streaming.
       For context, it is not practical to fetch e.g. 8192 samples
       each of size 1024 and dtype float64. This would become
       a bottle neck (from 2 to 7 seconds) and consume too much bandwidth
       (85 GB per epoch)
    2. Shuffling creates a bottleneck
    """

    def __init__(
        self,
        batch_size: int,
    ):
        super().__init__()
        fairseq2.setup_fairseq2()

        self.dataset_path = "jasonrichdarmawan/nllb-200-6M-sample-embedding"
        self.batch_size = batch_size

        # Shuffling hyperparameters
        self.num_workers = os.cpu_count() // 2
        self.prefetch_factor = self.num_workers * 2

    def setup(self, stage=None):
        self.train_data = load_dataset(
            path=self.dataset_path,
            split="train",
            streaming=False,
        )
        self.train_data = self.train_data.with_format(type="torch")
        # self.train_data = self.train_data.to_iterable_dataset(
        #     num_shards=self.num_workers,
        # )

    def train_dataloader(self):
        # self.train_data = self.train_data.shuffle(
        #     buffer_size=self.batch_size * self.prefetch_factor * 2,
        # )
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            # shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def collate_fn(self, batch):
        text1 = [item["text1"] for item in batch]
        lang1 = [item["lang1"] for item in batch]
        embedding1 = [item["embedding1"] for item in batch]
        embedding1 = torch.stack(embedding1, dim=0)
        return {
            "text1": text1,
            "embedding1": embedding1,
            "lang1": lang1,
        }
