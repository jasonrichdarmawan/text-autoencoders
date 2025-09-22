# %% [markdown]
"""
Download the primary datasets from [here](https://github.com/facebookresearch/fairseq/blob/nllb/examples/nllb/data/README.md)

Specifically:
1. Public data by running `python download_parallel_corpora.py --directory $DOWNLOAD_DIRECTORY`
2. NLLB-Seed data

Note:
1. At the time of writing, the `download_parallel_corpora.py` script unable to download the `Xhosa Navy` and `Bianet` datasets

Please remove the following datasets:
1. because SONAR does not support the languages:
   1. `aau` because of `orm` lang
   2. `hornmt` because of `aar` lang
   3. `mburisano` because of `nde` lang
   4. `tico` because of `orm` lang
2. because the datasets are too large to process:
   1. `indic_nlp`
   2. `til`
"""

# %%
# Auto-reload modules when code changes (for Jupyter notebooks)

from IPython import get_ipython

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except Exception:
    pass


# %%
# Use modified libraries

import sys

sys.path.insert(0, "/home/test/jason/SONAR")
sys.path.insert(0, "/home/test/jason/SAELens")


# %%
# Imports

from sae_utils import is_notebook

from pnpd_utils import (
    preprocess_akuapem,
    preprocess_cmu_hatian,
    preprocess_ffr,
    preprocess_french_ewe,
    preprocess_french_fongbe,
    preprocess_giossa,
    preprocess_kinya_smt,
    preprocess_lingala_songs,
    preprocess_menyo20k,
    preprocess_minangnlp,
    preprocess_mukiibi,
    preprocess_umsuka,
    preprocess_aau,

    preprocess_nllb_seed,

    preprocess_nynorsk_memories,
    preprocess_tico,
    preprocess_indic_nlp,
)

import pprint

from argparse import ArgumentParser

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

from datasets import (
    Dataset,
    Features,
    Value,
    Sequence,
    load_from_disk,
    concatenate_datasets,
    load_dataset,
)

from torch.utils.data import DataLoader

import torch

from tqdm import tqdm

import os

from tabulate import tabulate


# %%
# Parse arguments


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["save_to_disk", "push_to_hub", "verify"],
        required=True,
        help="Whether to save the processed dataset to disk or push to the Hugging Face Hub",
    )

    # Arguments for "save_to_disk" mode
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the directory containing the downloaded datasets",
    )

    parser.add_argument(
        "--cudaId",
        type=int,
        help="CUDA device ID to use for model inference",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[
            "akuapem",
            "cmu_hatian",
            "ffr",
            "french_ewe",
            "french_fongbe",
            "giossa",
            "kinya_smt",
            "lingala_songs",
            "menyo20k",
            "minangnlp",
            "mukiibi",
            "umsuka",
            "aau",

            "NLLB-Seed",
            "nynorsk_memories",
            "tico",
            "indic_nlp",
        ],
        help="Name of the dataset to process",
    )
    parser.add_argument(
        "--data_loader_batch_size",
        type=int,
        help="Batch size for the DataLoader",
    )
    parser.add_argument(
        "--predict_batch_size",
        type=int,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        help="Number of shards to split the dataset into when saving to disk",
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        help="Shard index (0-based) to save when saving to disk",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Path to save the processed dataset to when saving to disk",
    )

    # Arguments for "push_to_hub" mode
    parser.add_argument(
        "--processed_dir",
        type=str,
        help="Path to the directory containing the processed dataset shards to push to the Hugging Face Hub",
    )

    return parser.parse_args().__dict__


if is_notebook():
    WORKSPACE = "/workspace/ALGOVERSE/UJR/jason"
    MODE = "verify"  # "save_to_disk" | "push_to_hub" | "verify"
    sys.argv = [
        "preprocess_nllb_primary_datasets.py",
        "--mode",
        MODE,
        "--cudaId",
        "0",
    ]

    if MODE == "save_to_disk":
        sys.argv += [
            "--data_dir",
            # f"{WORKSPACE}/data/nllb/primary_datasets/public_data",
            f"{WORKSPACE}/data/nllb/primary_datasets",
            "--cudaId",
            "3",
            "--dataset_name",
            "NLLB-Seed",
            "--data_loader_batch_size",
            "128",
            "--predict_batch_size",
            "8",
            "--num_shards",
            "1",
            "--shard_idx",
            "0",
            "--save_dir",
            f"{WORKSPACE}/data/nllb/primary_datasets/processed",
        ]

args = parse_args()

print("Arguments:")
pprint.pprint(args)

# %% Load model

if args["mode"] == "save_to_disk":
    text2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(f"cuda:{args['cudaId']}"),
    )

# %%
# Define dataset features

features = Features(
    {
        "text_1": Value("string"),
        "lang_1": Value("string"),
        "text_2": Value("string"),
        "lang_2": Value("string"),
    }
)

preprocessed_features = Features(
    {
        "text_1": Value("string"),
        "lang_1": Value("string"),
        "embedding_1": Sequence(Value("float32"), length=1024),
        "text_2": Value("string"),
        "lang_2": Value("string"),
        "embedding_2": Sequence(Value("float32"), length=1024),
    }
)


# %%
# Load dataset

if args["mode"] == "save_to_disk":
    if args["dataset_name"] == "akuapem":
        data = preprocess_akuapem(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "cmu_hatian":
        data = preprocess_cmu_hatian(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "ffr":
        data = preprocess_ffr(directory=f"{args['data_dir']}/{args['dataset_name']}")
    elif args["dataset_name"] == "french_ewe":
        data = preprocess_french_ewe(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "french_fongbe":
        data = preprocess_french_fongbe(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "giossa":
        data = preprocess_giossa(directory=f"{args['data_dir']}/{args['dataset_name']}")
    elif args["dataset_name"] == "kinya_smt":
        data = preprocess_kinya_smt(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "lingala_songs":
        data = preprocess_lingala_songs(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "menyo20k":
        data = preprocess_menyo20k(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "minangnlp":
        data = preprocess_minangnlp(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "mukiibi":
        data = preprocess_mukiibi(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "nynorsk_memories":
        data = preprocess_nynorsk_memories(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "umsuka":
        data = preprocess_umsuka(directory=f"{args['data_dir']}/{args['dataset_name']}")
    elif args["dataset_name"] == "aau":
        data = preprocess_aau(directory=f"{args['data_dir']}/{args['dataset_name']}")
    elif args["dataset_name"] == "NLLB-Seed":
        data = preprocess_nllb_seed(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "tico":
        data = preprocess_tico(
            directory=f"{args['data_dir']}/{args['dataset_name']}"
        )
    elif args["dataset_name"] == "indic_nlp":
        data = preprocess_indic_nlp(
            directory=f"{args['data_dir']}/{args['dataset_name']}/finalrepo",
            split="train",
        )
    else:
        raise ValueError(f"Unsupported dataset: {args['dataset_name']}")
    dataset = Dataset.from_list(mapping=data, features=features)
    dataset = dataset.shard(num_shards=args["num_shards"], index=args["shard_idx"])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args["data_loader_batch_size"],
        num_workers=2,
        prefetch_factor=2,
    )

# %%
# Preprocess dataset

if args["mode"] == "save_to_disk":
    processed_data = []
    target_device = torch.device("cpu")
    for batch in tqdm(dataloader):
        texts = batch["text_1"] + batch["text_2"]
        langs = batch["lang_1"] + batch["lang_2"]

        embeddings = text2vec_model.predict(
            input=texts,
            source_lang=langs,
            batch_size=args["predict_batch_size"],
            target_device=torch.device("cpu"),
        )

        batch_size = len(batch["text_1"])
        for i in range(len(batch["text_1"])):
            processed_data.append(
                {
                    "text_1": batch["text_1"][i],
                    "lang_1": batch["lang_1"][i],
                    "embedding_1": embeddings[i],
                    "text_2": batch["text_2"][i],
                    "lang_2": batch["lang_2"][i],
                    "embedding_2": embeddings[i + batch_size],
                }
            )

# %%
# Save to disk

if args["mode"] == "save_to_disk":
    dataset_path = f"{args['save_dir']}/{args['dataset_name']}-{args['shard_idx']:05d}-of-{args['num_shards']:05d}"
    processed_dataset = Dataset.from_list(
        mapping=processed_data, features=preprocessed_features
    )
    processed_dataset.save_to_disk(
        dataset_path=dataset_path,
    )

# %%
# Save to HuggingFace Hub

if args["mode"] == "push_to_hub":

    datasets = [
        load_from_disk(f"{args['processed_dir']}/{path}")
        for path in os.listdir(args["processed_dir"])
    ]

    all_ds = concatenate_datasets(dsets=datasets)

    # shuffle to destroy any ordering
    all_ds = all_ds.shuffle(seed=42)

    all_ds.push_to_hub(
        repo_id="jasonrichdarmawan/nllb-primary-datasets-embedding",
        split="train",
    )

# %%
# Verify

if args["mode"] == "verify":
    ds = load_dataset(
        path="jasonrichdarmawan/nllb-primary-datasets-embedding",
        split="train",
    )
    ds = ds.with_format(type="torch")
    print(ds)
    dataloader = DataLoader(
        dataset=ds,
        batch_size=2,
    )
    iterable = iter(dataloader)
    samples = next(iterable)
    print("samples embedding1' shape:", samples["embedding_1"].shape)
    print("samples text1 shape:", len(samples["text_1"]))


# %5
# Load vec2text model

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
    batch_size = samples["embedding_1"].shape[0]

    reconstructed = vec2text_model.predict(
        inputs=torch.concat(
            [
                samples["embedding_1"],
                samples["embedding_2"],
                samples["embedding_1"],
                samples["embedding_2"],
            ],
            dim=0,
        ),
        target_lang=(
            samples["lang_1"]
            + samples["lang_1"]
            + samples["lang_2"]
            + samples["lang_2"]
        ),
        batch_size=8,
    )

    print(
        tabulate(
            tabular_data=[
                ["lang_1"] + samples["lang_1"],
                ["lang_2"] + samples["lang_2"],
                ["text_1"] + samples["text_1"],
                ["text_1 -> text1"] + reconstructed[:batch_size],
                ["text_2 -> text1"] + reconstructed[batch_size : 2 * batch_size],
                ["text_2"] + samples["text_2"],
                ["text_1 -> text_2"] + reconstructed[2 * batch_size : 3 * batch_size],
                ["text_2 -> text_2"] + reconstructed[3 * batch_size :],
            ]
        )
    )

# %%

