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

import torch
from torch import Tensor

import json

import fairseq2
fairseq2.setup_fairseq2()

from fairseq2.data.text.tokenizers import TextTokenizer, get_text_tokenizer_hub

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

from sonar.models.sonar_translation import SonarEncoderDecoderModel

from sonar.models.sonar_text import (
    get_sonar_text_decoder_hub,
    get_sonar_text_encoder_hub,
)

from tabulate import tabulate

from jaxtyping import Float

from sae_lens import (
    TrainingSAE,
    TrainingSAEConfig,
    GatedTrainingSAEConfig,
    BatchTopKTrainingSAEConfig,
    JumpReLUTrainingSAEConfig,
    LanguageModelSAERunnerConfig,
)

# %%
# Parse arguments

if is_notebook():
    WORKSPACE = "/workspace/jason/jason-ujr-1"
    LOGGER_ID = "g97mb3sb"
    CHECKPOINT_NAME = "epoch=45-step=240991"
    SAE_TYPE = "batchtopk"
    sys.argv = [
        "show_latents_fired.py",
        "--checkpoint_filename",
        f"{WORKSPACE}/experiments/sonar_sae/checkpoints/{LOGGER_ID}/{CHECKPOINT_NAME}.ckpt",
        "--d_sae",
        str(2**17),
        "--sae_type",
        SAE_TYPE,
        "--cudaId",
        0,
        f"--autointerp_results_filename",
        f"{WORKSPACE}/experiments/sonar_sae/autointerp_results/{LOGGER_ID}/{CHECKPOINT_NAME}.json",
    ]

    if SAE_TYPE == "batchtopk":
        sys.argv += [
            "--k",
            "64",
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
        "--sae_type",
        type=str,
        required=True,
        choices=["gated", "batchtopk", "jumprelu"],
        help="Type of the SAE model.",
    )

    parser.add_argument(
        "--k",
        type=int,
        help="Top-k for BatchTopK SAE",
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
# Load model

tokenizer_hub = get_text_tokenizer_hub()
encoder_hub = get_sonar_text_encoder_hub()
encoder = encoder_hub.load(
    "text_sonar_basic_encoder", device=torch.device(f"cuda:{args['cudaId']}")
)
encoder_tokenizer = tokenizer_hub.load("text_sonar_basic_encoder")
decoder_hub = get_sonar_text_decoder_hub()
decoder = decoder_hub.load(
    "text_sonar_basic_decoder", device=torch.device(f"cuda:{args['cudaId']}")
)
decoder_tokenizer = tokenizer_hub.load("text_sonar_basic_decoder")

model = SonarEncoderDecoderModel(encoder=encoder, decoder=decoder)

# %%
# Set up SAE model

if args["sae_type"] == "gated":
    sae_cfg = GatedTrainingSAEConfig(
        d_in=1024,
        d_sae=args["d_sae"],
        apply_b_dec_to_input=True,
        normalize_activations="none",  # TODO: implementation
        device=f"cuda:{args['cudaId']}",
    )
elif args["sae_type"] == "batchtopk":
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=1024,
        d_sae=args["d_sae"],
        apply_b_dec_to_input=True,
        normalize_activations="none",  # TODO: implementation
        k=args["k"],
        device=f"cuda:{args['cudaId']}",
    )
elif args["sae_type"] == "jumprelu":
    sae_cfg = JumpReLUTrainingSAEConfig(
        d_in=1024,
        d_sae=args["d_sae"],
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",  # TODO: implementation
        device=f"cuda:{args['cudaId']}",
    )

cfg = LanguageModelSAERunnerConfig(
    sae=sae_cfg,
)

sae = TrainingSAE.from_dict(
    config_dict=TrainingSAEConfig.from_dict(cfg.get_training_sae_cfg_dict()).to_dict()
)

# %%
# Load SAE model

lit_model = LitModel.load_from_checkpoint(
    checkpoint_path=args["checkpoint_filename"],
    map_location=f"cuda:{args['cudaId']}",
    model=model,
    encoder_tokenizer=encoder_tokenizer,
    decoder_tokenizer=decoder_tokenizer,
    sae=sae,
)

# Disable randomness, dropout, etc
lit_model.eval()

# %%
# Load SONAR embedding to text model

# text2vec_model = TextToEmbeddingModelPipeline(
#     encoder="text_sonar_basic_encoder",
#     tokenizer="text_sonar_basic_encoder",
#     device=t.device(f"cuda:{args['cudaId']}"),
# )

# vec2text_model = EmbeddingToTextModelPipeline(
#     decoder="text_sonar_basic_decoder",
#     tokenizer="text_sonar_basic_decoder",
#     device=t.device(f"cuda:{args['cudaId']}"),
# )

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

@torch.no_grad()
def process_embedding_with_sae(
    lit_model: LitModel,
    embedding: Float[Tensor, "batch d_in"], 
    source_langs: list[str]
):
    feature_acts, _ = lit_model.sae.encode_with_hidden_pre(x=embedding)
    fired = (feature_acts > 0).float()  # shape [batch_size, d_sae]

    fired_indices = torch.nonzero(fired)  # shape [num_fired, 2]
    # Group by sample
    fired_latents_per_sample = [
        fired_indices[fired_indices[:, 0] == i][:, 1].tolist()
        for i in range(fired.shape[0])
    ]

    sae_out = lit_model.sae.decode(feature_acts)
    reconstructed_texts = lit_model.decode_embedding(
        embeddings=torch.concat(
            [embedding, embedding, sae_out, sae_out],
            dim=0,
        ),
        target_lang=(
            source_langs 
            + ["eng_Latn"] * embedding.shape[0] 
            + source_langs 
            + ["eng_Latn"] * sae_out.shape[0]
        ),
    )

    return {
        "feature_acts": feature_acts,
        "fired_latents_per_sample": fired_latents_per_sample,
        "reconstructed_texts": reconstructed_texts,
    }

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
embedding = torch.concat(
    (
        data["nllb_200_6m_sample_embedding"]["embedding1"].to(lit_model.sae.device),
        data["nllb_primary_datasets_embedding"]["embedding1"].to(lit_model.sae.device),
    ),
    dim=0,
)

results = process_embedding_with_sae(lit_model=lit_model, embedding=embedding, source_langs=source_langs)
feature_acts = results["feature_acts"]
fired_latents_per_sample = results["fired_latents_per_sample"]
reconstructed_texts = results["reconstructed_texts"]


# %%
# Inference with simple texts

texts = [
    "Mice chase cats."
]
source_langs = ["eng_Latn"] * len(texts)
tokens, padding_mask = lit_model.tokenize_text(texts=texts, langs=source_langs)
embedding = lit_model.encode_text(
    seqs=tokens,
    padding_mask=padding_mask,
)

results = process_embedding_with_sae(lit_model=lit_model, embedding=embedding, source_langs=source_langs)
feature_acts = results["feature_acts"]
fired_latents_per_sample = results["fired_latents_per_sample"]
reconstructed_texts = results["reconstructed_texts"]


# %%
# Show results for specific samples

batch_index = 0

text = texts[batch_index]
reconstructed_without_sae_source_lang = reconstructed_texts[batch_index]
reconstructed_without_sae_eng_Latn = reconstructed_texts[len(texts) + batch_index]
reconstructed_with_sae_source_lang = reconstructed_texts[len(texts) * 2 + batch_index]
reconstructed_with_sae_eng_Latn = reconstructed_texts[len(texts) * 3 + batch_index]
fired_latents = fired_latents_per_sample[batch_index]

print(
    tabulate(
        tabular_data=[
            ["Text"] + [text],
            ["Reconstructed without sae (source lang)"] + [reconstructed_without_sae_source_lang],
            ["Reconstructed without sae (eng_Latn)"] + [reconstructed_without_sae_eng_Latn],
            ["Reconstructed with sae (source lang)"] + [reconstructed_with_sae_source_lang],
            ["Reconstructed with sae (eng_Latn)"] + [reconstructed_with_sae_eng_Latn],
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

fired_latent_detail_indices = [1]

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
