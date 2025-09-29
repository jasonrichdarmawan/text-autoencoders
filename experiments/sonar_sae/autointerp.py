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
    AutoInterpConfig,
    AutoInterp,
    print_autointerp_results,
)

import os

from argparse import ArgumentParser
import pprint

from sae_lens import (
    GatedTrainingSAEConfig,
    TrainingSAE,
    TrainingSAEConfig,
)

import torch

import fairseq2
fairseq2.setup_fairseq2()

from fairseq2.data.text.tokenizers import TextTokenizer, get_text_tokenizer_hub

from sonar.models.sonar_translation import SonarEncoderDecoderModel

from sonar.models.sonar_text import (
    get_sonar_text_decoder_hub,
    get_sonar_text_encoder_hub,
)

import json

from dotenv import load_dotenv

load_dotenv()

import asyncio

from tabulate import tabulate

import portalocker

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
    LOGGER_ID = "dnjrmfqk"
    CHECKPOINT_NAME = "epoch=9-step=30000"
    mode = "verify"  # "autointerp" or "verify"
    sys.argv = [
        "main.py",
        "--debug",
        "--mode",
        mode,
        # Results
        "--result_filename",
        f"{WORKSPACE}/experiments/sonar_sae/autointerp_results/{LOGGER_ID}/{CHECKPOINT_NAME}-nllb-200-6M-sample-embedding.json",
    ]

    if mode == "autointerp":
        SAE_TYPE = "batchtopk"
        sys.argv += [
            # Hyperparameters
            "--d_sae",
            "16384",
            "--sae_type",
            SAE_TYPE,
            "--batch_size",
            "128",
            "--latents",
            "8",
            "9",
            "10",
            "11",
            # Checkpoint
            "--checkpoint_filename",
            f"{WORKSPACE}/experiments/sonar_sae/checkpoints/{LOGGER_ID}/{CHECKPOINT_NAME}.ckpt",
            # Misc
            "--device",
            "cuda:0",
            "--max_concurrent",
            "10",
        ]

        if SAE_TYPE == "batchtopk":
            sys.argv += [
                "--k",
                "96",
            ]

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["autointerp", "verify"],
        help="Mode to run: autointerp or analysis",
    )

    # Hyperparameters
    parser.add_argument(
        "--d_sae",
        type=int,
        help="Dimensionality of the SAE bottleneck",
    )
    parser.add_argument(
        "--sae_type",
        type=str,
        help="Type of SAE to use",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Top-k for BatchTopK SAE",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for data loading",
    )
    parser.add_argument(
        "--latents",
        type=int,
        nargs="+",
        help="Latent dimensions to use",
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        help="Path to the trained SAE checkpoint",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the inference on",
    )

    parser.add_argument(
        "--max_concurrent",
        type=int,
        help="Maximum number of concurrent OpenAI API requests",
    )

    # Results
    parser.add_argument(
        "--result_filename",
        type=str,
        help="Path to save the results to",
    )

    args = parser.parse_args().__dict__
    return args


args = parse_args()
print("Arguments:")
pprint.pprint(args)


# %%
# Load dataset

if args["mode"] == "autointerp":
    data_module = DataModule(
        batch_size=args["batch_size"],
    )
    data_module.setup("fit")

# %%
# % Load model

if args["mode"] == "autointerp":
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

if args["mode"] == "autointerp":
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
    else:
        raise ValueError(f"Unknown sae_type: {args['sae_type']}")

    cfg = LanguageModelSAERunnerConfig(
        sae=sae_cfg,
    )

    sae = TrainingSAE.from_dict(
        config_dict=TrainingSAEConfig.from_dict(cfg.get_training_sae_cfg_dict()).to_dict()
    )

# %%
# Set up model

if args["mode"] == "autointerp":
    lit_model = LitModel.load_from_checkpoint(
        checkpoint_path=args["checkpoint_filename"],
        map_location=args["device"],
        model=model,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        sae=sae,
    )

    # Disable randomness, dropout, etc
    lit_model.eval()


# %%
# Inference

if args["mode"] == "autointerp":
    cfg = AutoInterpConfig(
        # latents=list[range(0, args["d_sae"], args["d_sae"] // 100)],
        latents=args["latents"],
        total_tokens=6369073 + 788368,  # Entire datasets
        max_tokens=65536,
    )

    autointerp = AutoInterp(
        cfg=cfg,
        data_module=data_module,
        model=lit_model,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Jupyter (and VS Code Interactive) already runs
    # an event loop
    # results = await autointerp.run(debug=args["debug"]) # type: ignore
    results = asyncio.run(
        autointerp.run(debug=args["debug"], max_concurrent=args["max_concurrent"])
    )

    if args["debug"]:
        pprint.pprint(results)


# %%
# Combine results with previous results

if args["mode"] == "autointerp":
    os.makedirs(os.path.dirname(args["result_filename"]), exist_ok=True)
    # Open for reading and writing, create if not exists
    try:
        with portalocker.Lock(args["result_filename"], "r+", timeout=60) as f:
            loaded_results = json.load(f)

            # moves the file pointer to the beginning, so we overwrite from the start
            f.seek(0)

            combined_results = {**loaded_results, **results}
            json.dump(combined_results, f, indent=2)

            # cuts off any remaining old data after the new data, so
            # only the new content remains
            f.truncate()
    except FileNotFoundError:
        with portalocker.Lock(args["result_filename"], "w+", timeout=60) as f:
            json.dump(results, f, indent=2)

    print(f"Results saved to {args['result_filename']}")


# %%
# Load results


if args["mode"] == "verify":
    with open(args["result_filename"], "r") as f:
        loaded_results = json.load(f)

    print("Loaded results:")
    print_autointerp_results(
        results_dict=loaded_results,
    )

# %%
# Inspect specific latents

if args["mode"] == "verify":
    # Check latent
    print(json.dumps(loaded_results["5248"], indent=2, ensure_ascii=False))


# %%
