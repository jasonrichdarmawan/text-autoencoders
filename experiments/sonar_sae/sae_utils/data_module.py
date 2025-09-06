import os

import lightning as L

from lightning.pytorch.utilities.combined_loader import CombinedLoader

from datasets import (
    load_dataset,
)

import torch
from torch.utils.data import (
    DataLoader,
)

import fairseq2

import random


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

        self.batch_size = batch_size

        # Shuffling hyperparameters
        self.num_workers = 2
        self.prefetch_factor = 64

    def setup(self, stage=None):
        self.train_nllb_200_6m_sample_embedding = load_dataset(
            path="jasonrichdarmawan/nllb-200-6M-sample-embedding",
            split="train",
            streaming=False,
        )
        self.train_nllb_200_6m_sample_embedding = (
            self.train_nllb_200_6m_sample_embedding.with_format(type="torch")
        )

        self.train_nllb_primary_datasets_embedding = load_dataset(
            path="jasonrichdarmawan/nllb-primary-datasets-embedding",
            split="train",
            streaming=False,
        )
        self.train_nllb_primary_datasets_embedding = (
            self.train_nllb_primary_datasets_embedding.with_format(type="torch")
        )
        # self.train_data = self.train_data.to_iterable_dataset(
        #     num_shards=self.num_workers,
        # )

    def train_dataloader(self):
        # self.train_data = self.train_data.shuffle(
        #     buffer_size=self.batch_size * self.prefetch_factor * 2,
        # )
        return CombinedLoader(
            iterables={
                "nllb_200_6m_sample_embedding": DataLoader(
                    dataset=self.train_nllb_200_6m_sample_embedding,
                    batch_size=self.batch_size // 2,
                    # shuffle=True,
                    collate_fn=self.collate_nllb_200_6m_sample_embedding,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    pin_memory=True,
                    persistent_workers=True,
                ),
                "nllb_primary_datasets_embedding": DataLoader(
                    dataset=self.train_nllb_primary_datasets_embedding,
                    batch_size=self.batch_size // 2,
                    # shuffle=True,
                    collate_fn=self.collate_nllb_primary_datasets_embedding,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    pin_memory=True,
                    persistent_workers=True,
                ),
            }
        )

    def collate_nllb_200_6m_sample_embedding(self, batch):
        text = []
        lang = []
        embedding = []
        for item in batch:
            if random.random() < 0.5:
                text.append(item["text1"])
                lang.append(item["lang1"])
                embedding.append(item["embedding1"])
            else:
                text.append(item["text2"])
                lang.append(item["lang2"])
                embedding.append(item["embedding2"])
        embedding = torch.stack(embedding, dim=0)
        return {
            "text1": text,
            "lang1": lang,
            "embedding1": embedding,
        }

    def collate_nllb_primary_datasets_embedding(self, batch):
        text = []
        lang = []
        embedding = []
        for item in batch:
            if random.random() < 0.5:
                text.append(item["text_1"])
                lang.append(item["lang_1"])
                embedding.append(item["embedding_1"])
            else:
                text.append(item["text_2"])
                lang.append(item["lang_2"])
                embedding.append(item["embedding_2"])
        embedding = torch.stack(embedding, dim=0)
        return {
            "text1": text,
            "lang1": lang,
            "embedding1": embedding,
        }
