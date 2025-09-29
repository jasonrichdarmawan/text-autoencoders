import lightning as L

from datasets import (
    load_dataset,
)

from pytorch_lightning.utilities.combined_loader import CombinedLoader

import torch
from torch.utils.data import (
    Dataset,
    IterableDataset,
    ConcatDataset,
    DataLoader,
    Subset,
)

import fairseq2

import random

from itertools import cycle


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
        self.num_workers = 16
        self.prefetch_factor = 64

    def setup(self, stage=None):
        # self.train_nllb_200_6m_sample_embedding = load_dataset(
        #     path="jasonrichdarmawan/nllb-200-6M-sample-embedding",
        #     split="train",
        #     streaming=False,
        # ).with_format(type="torch")
        # self.train_nllb_primary_datasets_embedding = (
        #     load_dataset(
        #         path="jasonrichdarmawan/nllb-primary-datasets-embedding",
        #         split="train",
        #         streaming=False,
        #     )
        #     .rename_columns(
        #         {
        #             "text_1": "text1",
        #             "lang_1": "lang1",
        #             "embedding_1": "embedding1",
        #             "text_2": "text2",
        #             "lang_2": "lang2",
        #             "embedding_2": "embedding2",
        #         }
        #     )
        #     .with_format(type="torch")
        # )
        ds1 = load_dataset(
            path="jasonrichdarmawan/nllb-200-6M-sample-embedding",
            split="train",
            streaming=False,
            num_proc=8,
        ).with_format(type="torch")

        ds2 = (
            load_dataset(
                path="jasonrichdarmawan/nllb-primary-datasets-public-data-embedding",
                split="train",
                streaming=False,
                num_proc=8,
            )
            .rename_columns(
                {
                    "text_1": "text1",
                    "lang_1": "lang1",
                    "embedding_1": "embedding1",
                    "text_2": "text2",
                    "lang_2": "lang2",
                    "embedding_2": "embedding2",
                }
            )
            .with_format(type="torch")
        )

        # ds1_unified = UnifiedDataset(
        #     ds=ds1,
        #     mapping={
        #         "text1": "text1",
        #         "lang1": "lang1",
        #         "embedding1": "embedding1",
        #         "text2": "text2",
        #         "lang2": "lang2",
        #         "embedding2": "embedding2",
        #     },
        # )
        # ds2_unified = UnifiedDataset(
        #     ds=ds2,
        #     mapping={
        #         "text1": "text_1",
        #         "lang1": "lang_1",
        #         "embedding1": "embedding_1",
        #         "text2": "text_2",
        #         "lang2": "lang_2",
        #         "embedding2": "embedding_2",
        #     },
        # )
        # self.train_data = self.train_data.to_iterable_dataset(
        #     num_shards=self.num_workers,
        # )

        # combined_dataset = ConcatDataset(datasets=[ds1, ds2])

        train_split = 0.9999
        train_len_ds1 = int(len(ds1) * train_split)
        train_len_ds2 = int(len(ds2) * train_split)

        train_indices_ds1 = list(range(train_len_ds1))
        val_indices_ds1 = list(range(train_len_ds1, len(ds1)))
        # 6,369,073 samples * (1 - 0,9999) / 2 batch_size = 636 batches
        # -> 636 batches / 8 iterations/second = 39 seconds
        # -> 39 seconds * 9 epochs * 4 eval per epoch = 23 minutes

        train_indices_ds2 = list(range(train_len_ds2))
        val_indices_ds2 = list(range(train_len_ds2, len(ds2)))

        # train_indices = train_indices_ds1 + train_indices_ds2
        # val_indices = val_indices_ds1 + val_indices_ds2
        self.train_nllb_200_6m_sample_embedding = Subset(
            dataset=ds1, indices=train_indices_ds1
        )
        self.val_nllb_200_6m_sample_embedding = Subset(
            dataset=ds1, indices=val_indices_ds1
        )
        self.train_nllb_primary_datasets_embedding = Subset(
            dataset=ds2, indices=train_indices_ds2
        )
        self.val_nllb_primary_datasets_embedding = Subset(
            dataset=ds2, indices=val_indices_ds2
        )
        # self.train_set = IntervalSubset(
        #     dataset=ds1,
        #     start=train_indices_ds1[0],
        #     end=train_indices_ds1[-1] + 1,
        # ) + IntervalSubset(
        #     dataset=ds2,
        #     start=train_indices_ds2[0],
        #     end=train_indices_ds2[-1] + 1,
        # )
        # self.val_set = IntervalSubset(
        #     dataset=ds1,
        #     start=val_indices_ds1[0],
        #     end=val_indices_ds1[-1] + 1,
        # ) + IntervalSubset(
        #     dataset=ds2,
        #     start=val_indices_ds2[0],
        #     end=val_indices_ds2[-1] + 1,
        # )
        # self.train_set = [ds1[i] for i in train_indices_ds1] + [
        #     ds2[i] for i in train_indices_ds2
        # ]
        # self.val_set = [ds1[i] for i in val_indices_ds1] + [
        #     ds2[i] for i in val_indices_ds2
        # ]

        # seed = torch.Generator().manual_seed(42)
        # self.train_set, self.val_set = random_split(
        #     dataset=combined_dataset,
        #     lengths=[train_len, val_len],
        #     generator=seed,
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
                    collate_fn=self.collate_fn,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    pin_memory=True,
                    persistent_workers=True,
                ),
                "nllb_primary_datasets_embedding": DataLoader(
                    dataset=self.train_nllb_primary_datasets_embedding,
                    batch_size=self.batch_size // 2,
                    # shuffle=True,
                    collate_fn=self.collate_fn,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    pin_memory=True,
                    persistent_workers=True,
                ),
            },
            mode="max_size_cycle",
        )
        # combined_dataset = CombinedDataset(
        #     ds1=self.ds1,
        #     ds2=self.ds2,
        # )
        # return DataLoader(
        #     dataset=combined_dataset,
        #     batch_size=self.batch_size // 2,
        #     # shuffle=True,
        #     collate_fn=self.collate_fn,
        #     num_workers=self.num_workers,
        #     prefetch_factor=self.prefetch_factor,
        #     pin_memory=True,
        #     persistent_workers=True,
        # )
        # return DataLoader(
        #     dataset=self.train_set,
        #     batch_size=self.batch_size,
        #     # shuffle=True,
        #     collate_fn=self.collate_fn,
        #     num_workers=self.num_workers,
        #     prefetch_factor=self.prefetch_factor,
        #     pin_memory=True,
        #     persistent_workers=True,
        # )

    def val_dataloader(self):
        # return DataLoader(
        #     dataset=self.val_set,
        #     batch_size=8,
        #     # shuffle=False,
        #     collate_fn=self.collate_fn,
        #     num_workers=self.num_workers,
        #     prefetch_factor=self.prefetch_factor,
        #     pin_memory=True,
        #     persistent_workers=True,
        # )
        batch_size = 4
        return CombinedLoader(
            iterables={
                "nllb_200_6m_sample_embedding": DataLoader(
                    dataset=self.val_nllb_200_6m_sample_embedding,
                    batch_size=batch_size // 2,
                    # shuffle=False,
                    collate_fn=self.collate_fn,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    pin_memory=True,
                    persistent_workers=True,
                ),
                "nllb_primary_datasets_embedding": DataLoader(
                    dataset=self.val_nllb_primary_datasets_embedding,
                    batch_size=batch_size // 2,
                    # shuffle=False,
                    collate_fn=self.collate_fn,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    pin_memory=True,
                    persistent_workers=True,
                ),
            },
            mode="max_size_cycle",
        )

    def collate_fn(self, batch):
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
            # if random.random() < 0.5:
            #     text.extend(item["text1"])
            #     lang.extend(item["lang1"])
            #     embedding.extend(item["embedding1"])
            # else:
            #     text.extend(item["text2"])
            #     lang.extend(item["lang2"])
            #     embedding.extend(item["embedding2"])
        embedding = torch.stack(embedding, dim=0)
        return {
            "text1": text,
            "lang1": lang,
            "embedding1": embedding,
        }

    # def collate_nllb_200_6m_sample_embedding(self, batch):
    #     text = []
    #     lang = []
    #     embedding = []
    #     for item in batch:
    #         if random.random() < 0.5:
    #             text.append(item["text1"])
    #             lang.append(item["lang1"])
    #             embedding.append(item["embedding1"])
    #         else:
    #             text.append(item["text2"])
    #             lang.append(item["lang2"])
    #             embedding.append(item["embedding2"])
    #     embedding = torch.stack(embedding, dim=0)
    #     return {
    #         "text1": text,
    #         "lang1": lang,
    #         "embedding1": embedding,
    #     }

    # def collate_nllb_primary_datasets_embedding(self, batch):
    #     text = []
    #     lang = []
    #     embedding = []
    #     for item in batch:
    #         if random.random() < 0.5:
    #             text.append(item["text_1"])
    #             lang.append(item["lang_1"])
    #             embedding.append(item["embedding_1"])
    #         else:
    #             text.append(item["text_2"])
    #             lang.append(item["lang_2"])
    #             embedding.append(item["embedding_2"])
    #     embedding = torch.stack(embedding, dim=0)
    #     return {
    #         "text1": text,
    #         "lang1": lang,
    #         "embedding1": embedding,
    #     }


# class UnifiedDataset(Dataset):
#     def __init__(self, ds, mapping):
#         self.ds = ds
#         self.mapping = mapping

#     def __len__(self):
#         return len(self.ds)

#     def __getitem__(self, idx):
#         item = self.ds[idx]
#         return {
#             "text1": item[self.mapping["text1"]],
#             "lang1": item[self.mapping["lang1"]],
#             "embedding1": item[self.mapping["embedding1"]],
#             "text2": item[self.mapping["text2"]],
#             "lang2": item[self.mapping["lang2"]],
#             "embedding2": item[self.mapping["embedding2"]],
#         }


# class CombinedDataset(IterableDataset):
#     def __init__(self, ds1, ds2):
#         self.ds1 = ds1
#         self.ds2 = ds2

#         assert len(self.ds1) > len(self.ds2)

#     def __iter__(self):
#         ds1_iter = iter(self.ds1)
#         ds2_iter = cycle(self.ds2)
#         for item_long, item_short in zip(ds1_iter, ds2_iter):
#             yield {
#                 "text1": [item_long["text1"], item_short["text_1"]],
#                 "lang1": [item_long["lang1"], item_short["lang_1"]],
#                 "embedding1": [item_long["embedding1"], item_short["embedding_1"]],
#                 "text2": [item_long["text2"], item_short["text_2"]],
#                 "lang2": [item_long["lang2"], item_short["lang_2"]],
#                 "embedding2": [item_long["embedding2"], item_short["embedding_2"]],
#             }


class IntervalSubset(Dataset):
    def __init__(self, dataset, start, end):
        self.dataset = dataset
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        return self.dataset[self.start + idx]
