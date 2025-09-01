# %% [markdown]
"""
Preprocess [slone/nllb-200-10M-sample](https://huggingface.co/datasets/slone/nllb-200-10M-sample)
to generate text embeddings using SONAR's text encoder

Notes:
1. `all_ds` needs to be shuffled before pushing to hub to destroy any ordering.
   Consequently, the `all_ds.upload_to_hub` takes 3 hours instead of 1 hour.
   So, why not do shuffle each shard before saving to disk?
   Suppose the original dataset has a bias, then shuffling each shard would
   be insufficient to break the bias

Known Issues:
1. `datasets==3.6.0` will throws this error when `load_dataset(...)`. The solution is to upgrade to `datasets==4.0.0`.

    ```
    Feature type 'List' not found. Available feature types: ['Value', 'ClassLabel', 'Translation', 'TranslationVariableLanguages', 'LargeList', 'Sequence', 'Array2D', 'Array3D', 'Array4D', 'Array5D', 'Audio', 'Image', 'Video', 'Pdf']
    ```
"""

# %%
# 4Auto-reload modules when code changes (for Jupyter notebooks)

from IPython import get_ipython

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except Exception:
    pass

# %%
# Use modified libraries

import sys

sys.path.insert(0, "/workspace/ALGOVERSE/UJR/jason/SONAR")
sys.path.insert(0, "/workspace/ALGOVERSE/UJR/jason/SAELens")

# %%

from tabulate import tabulate

import pprint

from sae_utils import is_notebook

import sys

import argparse

import pprint

from typing import (
    Any,
)

from datasets import (
    load_dataset,
    Dataset,
    Features,
    Value,
    Sequence,
    load_from_disk,
    concatenate_datasets,
)

import torch
from torch.utils.data import DataLoader

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

from tqdm import tqdm

# %%
# Arguments

if is_notebook():
    workspace = "/workspace/ALGOVERSE/UJR/jason"
    split = "train"
    num_shards = 48
    mode = "verify"  # "save_to_disk" | "push_to_hub" | "verify"
    sys.argv = [
        "preprocess.py",
        "--workspace",
        workspace,
        "--num_shards",
        str(num_shards),
        "--split",
        split,
        "--cudaId",
        "2",
        "--mode",
        mode,
    ]

    # save to disk
    if mode == "save_to_disk":
        sys.argv += ["--shard_idx", "0", "--debug"]

    # push to hub
    elif mode == "push_to_hub":
        shard_paths = [
            f"{workspace}/data/nllb-200-6M-sample-embedding/{split}-{i:05d}-of-{num_shards:05d}"
            for i in range(num_shards)
        ]
        sys.argv += ["--shard_paths"] + shard_paths

    # verify
    elif mode == "verify":
        sys.argv += ["--mode", "verify"]

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--workspace",
    type=str,
    help="Workspace directory",
)
argparser.add_argument(
    "--num_shards",
    type=int,
    help="Number of Shards to use",
)
argparser.add_argument(
    "--shard_idx",
    type=int,
    help="Shard index (0-based)",
)
argparser.add_argument(
    "--cudaId",
    type=int,
    help="CUDA device ID to use",
)
argparser.add_argument(
    "--split",
    type=str,
    help="Save as which split",
)
argparser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode (run only a few batches)",
)
argparser.add_argument(
    "--mode",
    type=str,
    choices=["save_to_disk", "push_to_hub", "verify"],
)
argparser.add_argument(
    "--shard_paths",
    type=str,
    nargs="*",
    help="Paths to load shards from",
)
args = argparser.parse_args().__dict__

data_loader_batch_size = 128
num_proc = 8
predict_batch_size = 8
workspace = args["workspace"]
split = args["split"]

print("Arguments:")
pprint.pprint(args)

# %%
# Load dataset

if args["mode"] == "save_to_disk":
    dataset = load_dataset(
        path="slone/nllb-200-10M-sample",
        split="train",
    )

    # Shard the dataset to support multi GPUs processing
    dataset = dataset.shard(num_shards=args["num_shards"], index=args["shard_idx"])

    # Filter out low similarity pairs
    # https://github.com/facebookresearch/SONAR/issues/75#issuecomment-3221061689
    dataset = dataset.filter(function=lambda example: example["blaser_sim"] >= 3.5)

    # Sort dataset by text1_len descending
    # for better time estimation, GPU utilization (less padding)
    dataset = dataset.map(
        function=lambda x: {"text1_len": [len(t) for t in x["text1"]]},
        batched=True,
        num_proc=num_proc,
    )
    dataset = dataset.sort(column_names="text1_len", reverse=True)

    print("Dataset length:", len(dataset))

# %%
# DataLoader

if args["mode"] == "save_to_disk":
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=data_loader_batch_size,
        num_workers=2,
        prefetch_factor=2,
    )

    if args["debug"]:
        samples = next(iter(dataloader))

        print("Samples keys:")
        pprint.pprint(samples.keys())

        print("\nSamples:")
        pprint.pprint({k: v[:2] for k, v in samples.items()})

# %%
# Load models

if args["mode"] == "save_to_disk":
    text2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(f"cuda:{args['cudaId']}"),
    )

# %%
# Test model

if args["mode"] == "save_to_disk" and args["debug"]:
    embeddings = text2vec_model.predict(
        input=samples["text1"],
        source_lang=samples["lang1"],
    )

    print("Embeddings shape:", embeddings.shape)

# %%
# Generate embeddings

if args["mode"] == "save_to_disk":
    features = Features(
        {
            "laser_score": Value(dtype="float64"),
            "lang1": Value(dtype="string"),
            "text1": Value(dtype="string"),
            "embedding1": Sequence(feature=Value("float64"), length=1024),
            "lang2": Value(dtype="string"),
            "text2": Value(dtype="string"),
            "embedding2": Sequence(feature=Value("float64"), length=1024),
            "blaser_sim": Value(dtype="float64"),
        }
    )

    data: list[dict[str, Any]] = []

    target_device = torch.device("cpu")
    count = 0
    for batch in tqdm(dataloader):
        texts = batch["text1"] + batch["text2"]
        langs = batch["lang1"] + batch["lang2"]

        embeddings = text2vec_model.predict(
            input=texts,
            source_lang=langs,
            batch_size=predict_batch_size,
            target_device=target_device,
        )
        batch_size = len(batch["text1"])
        for i in range(batch_size):
            data.append(
                {
                    "laser_score": batch["laser_score"][i],
                    "text1": batch["text1"][i],
                    "lang1": batch["lang1"][i],
                    "embedding1": embeddings[i],
                    "text2": batch["text2"][i],
                    "lang2": batch["lang2"][i],
                    "embedding2": embeddings[i + batch_size],
                    "blaser_sim": batch["blaser_sim"][i],
                }
            )
        count += 1
        if count % 10 == 0 and split == "dev":
            break

# %%
# Load to Dataset

if args["mode"] == "save_to_disk":
    ds = Dataset.from_list(
        mapping=data,
        features=features,
        split=split,
    )

# %%
# Save to disk

if args["mode"] == "save_to_disk":
    dataset_path = f"{workspace}/data/nllb-200-6M-sample-embedding/{split}-{args['shard_idx']:05d}-of-{args['num_shards']:05d}"
    print(f"Saving to {dataset_path}...")
    ds.save_to_disk(
        dataset_path=dataset_path,
    )

# %%
# Save to HuggingFace Hub

if args["mode"] == "push_to_hub":
    # Load all shards
    shards = [load_from_disk(path) for path in args["shard_paths"]]

    all_ds = concatenate_datasets(dsets=shards)

    # shuffle to destroy any ordering
    all_ds = all_ds.shuffle(seed=42)

    all_ds.push_to_hub(
        repo_id="jasonrichdarmawan/nllb-200-6M-sample-embedding",
        split=split,
    )

# %%
# Verify

if args["mode"] == "verify":
    all_ds = load_dataset(
        path="jasonrichdarmawan/nllb-200-6M-sample-embedding",
        split=split,
        streaming=True,
    )
    all_ds = all_ds.with_format(type="torch")
    print(all_ds)
    dataloader = DataLoader(
        dataset=all_ds,
        batch_size=2,
    )
    iterable = iter(dataloader)
    samples = next(iterable)
    print("samples embedding1' shape:", samples["embedding1"].shape)
    print("samples text1 shape:", len(samples["text1"]))

# %%
# Load embedding2text model

if args["mode"] == "verify":
    vec2text_model = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_decoder",
        device=torch.device(f"cuda:{args['cudaId']}"),
    )

# %%
# Reconstruct embeddings

if args["mode"] == "verify":
    samples = next(iterable)
    batch_size = samples["embedding1"].shape[0]

    reconstructed = vec2text_model.predict(
        inputs=torch.concat(
            [
                samples["embedding1"],
                samples["embedding2"],
                samples["embedding1"],
                samples["embedding2"],
            ],
            dim=0,
        ),
        target_lang=(
            samples["lang1"] + samples["lang1"] + samples["lang2"] + samples["lang2"]
        ),
        batch_size=8,
    )

    print(
        tabulate(
            tabular_data=[
                ["lang1"] + samples["lang1"],
                ["lang2"] + samples["lang2"],
                ["text1"] + samples["text1"],
                ["text1 -> text1"] + reconstructed[:batch_size],
                ["text2 -> text1"] + reconstructed[batch_size : 2 * batch_size],
                ["text2"] + samples["text2"],
                ["text1 -> text2"] + reconstructed[2 * batch_size : 3 * batch_size],
                ["text2 -> text2"] + reconstructed[3 * batch_size :],
            ]
        )
    )

# %%
