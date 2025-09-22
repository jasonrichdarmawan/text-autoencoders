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
    WORKSPACE = "/workspace/ALGOVERSE/UJR/jason"
    LOGGER_ID = "mffcsqri"
    SAE_TYPE = "jumprelu"
    # CHECKPOINT_NAME = "last"
    CHECKPOINT_NAME = "epoch=9-step=29544"
    sys.argv = [
        "test_trained_sae.py",
        "--sae_type",
        SAE_TYPE,
        # Hyperparameters
        "--d_sae",
        "16384",
        # Checkpoint
        "--checkpoint_filename",
        f"{WORKSPACE}/experiments/sonar_sae/checkpoints/{LOGGER_ID}/{CHECKPOINT_NAME}.ckpt",
        # Misc
        "--device",
        "cuda:1",
        # "cpu",
    ]

    if SAE_TYPE == "batchtopk":
        sys.argv += [
            "--k",
            "96",
        ]


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--sae_type",
        type=str,
        help="Type of SAE to use",
    )

    # Hyperparameters
    parser.add_argument(
        "--d_sae",
        type=int,
        required=True,
        help="Dimensionality of the SAE latent space",
    )

    # Architecture-specific hyperparameters
    parser.add_argument(
        "--k",
        type=int,
        help="Top-k for BatchTopK SAE",
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
# Load model

tokenizer_hub = get_text_tokenizer_hub()
encoder_hub = get_sonar_text_encoder_hub()
encoder = encoder_hub.load(
    "text_sonar_basic_encoder", device=torch.device(args["device"])
)
encoder_tokenizer = tokenizer_hub.load("text_sonar_basic_encoder")
decoder_hub = get_sonar_text_decoder_hub()
decoder = decoder_hub.load(
    "text_sonar_basic_decoder", device=torch.device(args["device"])
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
        device=args["device"],
    )
elif args["sae_type"] == "batchtopk":
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=1024,
        d_sae=args["d_sae"],
        apply_b_dec_to_input=True,
        normalize_activations="none",  # TODO: implementation
        k=args["k"],
        device=args["device"],
    )
elif args["sae_type"] == "jumprelu":
    sae_cfg = JumpReLUTrainingSAEConfig(
        d_in=1024,
        d_sae=args["d_sae"],
        apply_b_dec_to_input=True,
        normalize_activations="expected_average_only_in",  # TODO: implementation
        device=args["device"],
    )

cfg = LanguageModelSAERunnerConfig(
    sae=sae_cfg,
)

sae = TrainingSAE.from_dict(
    config_dict=TrainingSAEConfig.from_dict(cfg.get_training_sae_cfg_dict()).to_dict()
)


# %%
# Load trained model

lit_model = LitModel.load_from_checkpoint(
    checkpoint_path=args["checkpoint_filename"],
    map_location=args["device"],
    model=model,
    encoder_tokenizer=encoder_tokenizer,
    decoder_tokenizer=decoder_tokenizer,
    sae=sae,
)

# TODO: remove
# lit_model = LitModel(
#     cfg=cfg,
#     model=model,
#     encoder_tokenizer=encoder_tokenizer,
#     decoder_tokenizer=decoder_tokenizer,
#     sae=sae,
# )
# lit_model.to(torch.device(args["device"]))


# Disable randomness, dropout, etc
lit_model.eval()

# %%
# TODO: remove this later

# texts = [
#     "halo dunia",
#     "halo dunia",
# ]
# langs = ["ind_Latn", "ind_Latn"]
texts = [
    "Mice chase cats.",
    "hello world",
    "There are two methods to proceed.",
]
langs = ["eng_Latn", "eng_Latn", "eng_Latn"]

tokens, padding_mask = lit_model.tokenize_text(texts=texts, langs=langs)
embeddings = lit_model.encode_text(seqs=tokens, padding_mask=padding_mask)
embeddings = lit_model.sae(embeddings)

target_seqs = lit_model.get_target_seqs(
    # texts=["halo dunia", "halo dunia"],
    # langs=["ind_Latn", "ind_Latn"],
    texts=texts,
    langs=langs,
)

logits, padded = lit_model.get_logits(embeddings=embeddings, target_seqs=target_seqs)

losses, n_toks = lit_model.get_decoder_loss(
    logits=logits,
    padded=padded,
)
print("losses:", losses)
print("n_toks:", n_toks)
avg_loss = losses.sum() / n_toks.sum()
print(f"Avg loss: {avg_loss.item():.6f}")

reconstructed_texts = lit_model.decode_embedding(
    embeddings=embeddings,
    target_lang=langs,
)
print("reconstructed_texts:", reconstructed_texts)

# %%
# Test

texts = [
    # "Despite the algorithm's polynomial time complexity, pathological edge cases can induce exponential slowdowns.",
    # "In the midst of existential uncertainty, the philosopher pondered the ineffability of consciousness.",
    # "Heisenberg's uncertainty principle imposes fundamental limits on simultaneous measurements of position and momentum.",
    # "Interpretasi multisemesta menimbulkan pertanyaan ontologis tentang hakikat kenyataan itu sendiri.",
    # "Paradoks Schrödinger menyoroti ketidakpastian eksistensi melalui eksperimen kucing yang terkenal itu.",
    "Time flies.",
    "Flying time.",
    "Cats chase mice.",
    "Mice chase cats.",
    "Quantum leaps.",
]

langs = [
    "eng_Latn",
    "eng_Latn",
    "eng_Latn",
    "eng_Latn",
    "eng_Latn",
]

batch_size = len(texts)

tokens, padding_mask = lit_model.tokenize_text(
    texts=texts,
    langs=langs,
)

embeddings = lit_model.encode_text(seqs=tokens, padding_mask=padding_mask)

reconstructed_embeddings = lit_model.sae(embeddings)

# MSE
mse_with_sae = torch.mean((embeddings - reconstructed_embeddings) ** 2).item()
print(f"MSE with SAE: {mse_with_sae:.6f}")

reconstructed_texts = lit_model.decode_embedding(
    embeddings=torch.cat([embeddings, reconstructed_embeddings], dim=0),
    target_lang=langs + langs,
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
