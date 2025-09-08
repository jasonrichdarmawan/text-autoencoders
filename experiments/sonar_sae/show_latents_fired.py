# %%
# Auto-reload modules when code changes (for Jupyter Notebook)

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
# Library imports

from sae_utils import (
    is_notebook,
    DataModule,
    LitModel,
)

from autointerp_utils import (
    print_autointerp_results,
)

import sys

from argparse import ArgumentParser

import pprint

from sae_lens import (
    GatedTrainingSAEConfig,
    TrainingSAE,
    TrainingSAEConfig,
)

import torch as t
from torch import Tensor

import json

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

from tabulate import tabulate

from jaxtyping import Float

# %%
# Parse arguments

if is_notebook():
    WORKSPACE = "/workspace/ALGOVERSE/UJR/jason"
    LOGGER_ID = "tk4tyu7f"
    CHECKPOINT_NAME = "epoch=9-step=30000"
    sys.argv = [
        "show_latents_fired.py",
        "--checkpoint_filename",
        f"{WORKSPACE}/experiments/sonar_sae/checkpoints/{LOGGER_ID}/{CHECKPOINT_NAME}.ckpt",
        "--d_sae",
        "16384",
        "--cudaId",
        "2",
        f"--autointerp_results_filename",
        f"{WORKSPACE}/experiments/sonar_sae/autointerp_results/{LOGGER_ID}/{CHECKPOINT_NAME}-nllb-200-6M-sample-embedding.json",
    ]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        required=True,
        help="Path to the model checkpoint file.",
    )
    parser.add_argument(
        "--d_sae",
        type=int,
        required=True,
        help="Dimensionality of the SAE latent space.",
    )
    parser.add_argument(
        "--cudaId",
        type=str,
    )
    parser.add_argument(
        "--autointerp_results_filename",
        type=str,
        required=True,
        help="Path to the autointerp results JSON file.",
    )
    return parser.parse_args().__dict__


args = parse_args()
print("Arguments:", args)
pprint.pprint(args)

# %%
# Load dataset

data_module = DataModule(
    batch_size=2,
)
data_module.setup("fit")

# %%
# Set up SAE model

sae_cfg = GatedTrainingSAEConfig(
    d_in=1024,
    d_sae=args["d_sae"],
    apply_b_dec_to_input=True,
    normalize_activations="none",
    device=f"cuda:{args['cudaId']}",
)

sae = TrainingSAE.from_dict(
    config_dict=TrainingSAEConfig.from_dict(sae_cfg.to_dict()).to_dict(),
)

# %%
# Load SAE model

model = LitModel.load_from_checkpoint(
    checkpoint_path=args["checkpoint_filename"],
    map_location=f"cuda:{args['cudaId']}",
    sae=sae,
)

# Disable randomness, dropout, etc
model.eval()

# %%
# Load SONAR embedding to text model

text2vec_model = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=t.device(f"cuda:{args['cudaId']}"),
)

vec2text_model = EmbeddingToTextModelPipeline(
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_decoder",
    device=t.device(f"cuda:{args['cudaId']}"),
)

# %%
# Load autointerp results

try:
    with open(args["autointerp_results_filename"], "r") as f:
        autointerp_results = json.load(f)
except FileNotFoundError as e:
    print("Autointerp results file not found:", e)
    autointerp_results = {}

# %%
# Inference with seen data

iterable = iter(data_module.train_dataloader())

data = next(iterable)[0]
texts = (
    data["nllb_200_6m_sample_embedding"]["text1"]
    + data["nllb_primary_datasets_embedding"]["text1"]
)
source_langs = (
    data["nllb_200_6m_sample_embedding"]["lang1"]
    + data["nllb_primary_datasets_embedding"]["lang1"]
)
embedding = t.concat(
    (
        data["nllb_200_6m_sample_embedding"]["embedding1"].to(model.sae.device),
        data["nllb_primary_datasets_embedding"]["embedding1"].to(model.sae.device),
    ),
    dim=0,
)


@t.no_grad()
def process_embedding_with_sae(
    embedding: Float[Tensor, "batch d_in"], source_langs: list[str]
):
    with t.autocast(device_type="cuda", dtype=t.bfloat16):
        feature_acts, _ = model.sae.encode_with_hidden_pre(x=embedding)
        fired = (feature_acts > 0).float()  # shape [batch_size, d_sae]

        fired_indices = t.nonzero(fired)  # shape [num_fired, 2]
        # Group by sample
        fired_latents_per_sample = [
            fired_indices[fired_indices[:, 0] == i][:, 1].tolist()
            for i in range(fired.shape[0])
        ]

        sae_out = model.sae.decode(feature_acts)
        reconstructed_texts = vec2text_model.predict(
            inputs=t.concat(
                [sae_out, sae_out],
                dim=0,
            ),
            target_lang=["eng_Latn"] * sae_out.shape[0] + source_langs,
            batch_size=8,
        )

        return {
            "feature_acts": feature_acts,
            "fired_latents_per_sample": fired_latents_per_sample,
            "reconstructed_texts": reconstructed_texts,
        }


results = process_embedding_with_sae(embedding=embedding, source_langs=source_langs)
feature_acts = results["feature_acts"]
fired_latents_per_sample = results["fired_latents_per_sample"]
reconstructed_texts = results["reconstructed_texts"]


# %%
# Inference with simple texts

texts = [
    "cat",
    "dog",
    "and",
    "cat and dog",
    "dog and cat",
    "the cat",
    "the dog",
    "the cat and the dog",
    "the dog and the cat",
]
source_langs = ["eng_Latn"] * len(texts)
embedding = text2vec_model.predict(
    input=texts,
    source_lang=source_langs,
    target_device=t.device(f"cuda:{args['cudaId']}"),
)

results = process_embedding_with_sae(embedding=embedding, source_langs=source_langs)
fired_latents_per_sample = results["fired_latents_per_sample"]
reconstructed_texts = results["reconstructed_texts"]


# %%
# Show results for specific samples

batch_index = 0

text = texts[batch_index]
reconstructed_text_eng_Latn = reconstructed_texts[batch_index]
reconstructed_text_source_lang = reconstructed_texts[
    len(reconstructed_texts) // 2 + batch_index
]
fired_latents = fired_latents_per_sample[batch_index]

print(
    tabulate(
        tabular_data=[
            ["Text"] + [text],
            ["Reconstructed with sae (source lang)"] + [reconstructed_text_source_lang],
            ["Reconstructed with sae (eng_Latn)"] + [reconstructed_text_eng_Latn],
        ]
    )
)
print("Latent dimensions fired:", len(fired_latents))
print("Fired latent dimensions:", " ".join(map(str, fired_latents)))
print_autointerp_results(
    results_dict=autointerp_results,
    latents=fired_latents,
    acts=feature_acts[batch_index].tolist(),
)

# %%
# Show details for specific latents

fired_latent_detail_indices = [11508]

print(
    json.dumps(
        {
            k: v
            for k, v in autointerp_results.items()
            if int(k) in fired_latent_detail_indices
        },
        indent=2,
        ensure_ascii=False,
    )
)

# %%
