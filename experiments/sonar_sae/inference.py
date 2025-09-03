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

sys.path.insert(0, "/workspace/ALGOVERSE/UJR/jason/SONAR")
sys.path.insert(0, "/workspace/ALGOVERSE/UJR/jason/SAELens")


# %%
# Library imports

from sae_utils import (
    is_notebook,
    LitModel,
)

import pprint
from tabulate import tabulate

from argparse import ArgumentParser

import torch

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

from sae_lens import (
    TrainingSAE,
    TrainingSAEConfig,
    GatedTrainingSAEConfig,
)

# %%
# Parse arguments

if is_notebook():
    WORKSPACE = "/workspace/ALGOVERSE/UJR/jason"
    LOGGER_NAME = "crkg5s4v"
    sys.argv = [
        "main.py",
        # Hyperparameters
        "--d_sae",
        "16384",
        # Checkpoint
        "--checkpoint_filename",
        f"{WORKSPACE}/experiments/sonar_sae/checkpoints/{LOGGER_NAME}/epoch=19-step=30000.ckpt",
        # Misc
        "--device",
        "cuda:1",
    ]


def parse_args():
    parser = ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--d_sae",
        type=int,
        required=True,
        help="Dimensionality of the SAE latent space",
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the model on",
    )

    return parser.parse_args().__dict__


args = parse_args()
print("Arguments:")
pprint.pprint(args)


# %%
# Load text2vec model

text2vec_model = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=torch.device(f"{args['device']}"),
)

# %%
# Load vec2text model

vec2text_model = EmbeddingToTextModelPipeline(
    decoder="text_sonar_basic_decoder",
    tokenizer="text_sonar_basic_decoder",
    device=torch.device(f"{args['device']}"),
)


# %%
# Set up SAE model

cfg = GatedTrainingSAEConfig(
    d_in=1024,
    d_sae=args["d_sae"],
    apply_b_dec_to_input=True,
    normalize_activations="none",  # TODO: implementation
)


sae = TrainingSAE.from_dict(
    config_dict=TrainingSAEConfig.from_dict(cfg.to_dict()).to_dict()
)


# %%

model = LitModel.load_from_checkpoint(
    checkpoint_path=args["checkpoint_filename"],
    map_location=args["device"],
    sae=sae,
)


# Disable randomness, dropout, etc
model.eval()


# %%
# Test

texts = [
    "Despite the algorithm's polynomial time complexity, pathological edge cases can induce exponential slowdowns.",
    "In the midst of existential uncertainty, the philosopher pondered the ineffability of consciousness.",
    "Heisenberg's uncertainty principle imposes fundamental limits on simultaneous measurements of position and momentum.",
    "Interpretasi multisemesta menimbulkan pertanyaan ontologis tentang hakikat kenyataan itu sendiri.",
    "Paradoks Schrödinger menyoroti ketidakpastian eksistensi melalui eksperimen kucing yang terkenal itu.",
]

lang = [
    "eng_Latn",
    "eng_Latn",
    "eng_Latn",
    "ind_Latn",
    "ind_Latn",
]

batch_size = len(texts)

embeddings = text2vec_model.predict(
    input=texts,
    source_lang=lang,
)

reconstructions = model.sae(embeddings)

# MSE difference
mse_with_sae = torch.mean((embeddings - reconstructions) ** 2).item()
print(f"MSE with SAE: {mse_with_sae:.6f}")

reconstructed_texts = vec2text_model.predict(
    inputs=torch.cat([embeddings, reconstructions], dim=0),
    target_lang=lang + lang,
)

for i in range(batch_size):
    print(
        tabulate(
            tabular_data=[
                ["text"] + [texts[i]],
                ["reconstructed_without_sae"] + [reconstructed_texts[i]],
                ["reconstructed_with_sae"] + [reconstructed_texts[i + batch_size]],
            ]
        )
    )

# %%
