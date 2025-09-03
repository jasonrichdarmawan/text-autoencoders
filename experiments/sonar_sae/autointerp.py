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
)

import os

from argparse import ArgumentParser
import pprint

from sae_lens import (
    GatedTrainingSAEConfig,
    TrainingSAE,
    TrainingSAEConfig,
)

import json

from dotenv import load_dotenv

load_dotenv()

import asyncio

from tabulate import tabulate

# %%
# Parse arguments

if is_notebook():
    WORKSPACE = "/workspace/ALGOVERSE/UJR/jason"
    LOGGER_NAME = "crkg5s4v"
    CHECKPOINT_NAME = "epoch=19-step=30000"
    mode = "verify"  # "autointerp" or "verify"
    sys.argv = [
        "main.py",
        "--debug",
        "--mode",
        mode,
        # Results
        "--result_filename",
        f"{WORKSPACE}/experiments/sonar_sae/autointerp_results/{LOGGER_NAME}/{CHECKPOINT_NAME}-nllb-200-6M-sample-embedding.json",
    ]

    if mode == "autointerp":
        sys.argv += [
            # Hyperparameters
            "--d_sae",
            "16384",
            "--batch_size",
            "128",
            "--latents",
            "8",
            "9",
            "10",
            "11",
            # Checkpoint
            "--checkpoint_filename",
            f"{WORKSPACE}/experiments/sonar_sae/checkpoints/{LOGGER_NAME}/{CHECKPOINT_NAME}.ckpt",
            # Misc
            "--device",
            "cuda:2",
            "--max_concurrent",
            "10",
        ]


# %%


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


# %%
# Set up SAE model

if args["mode"] == "autointerp":
    sae_cfg = GatedTrainingSAEConfig(
        d_in=1024,
        d_sae=args["d_sae"],
        apply_b_dec_to_input=True,
        normalize_activations="none",
        device=args["device"],
    )

    sae = TrainingSAE.from_dict(
        config_dict=TrainingSAEConfig.from_dict(sae_cfg.to_dict()).to_dict(),
    )

# %%
# Set up model

if args["mode"] == "autointerp":
    model = LitModel.load_from_checkpoint(
        checkpoint_path=args["checkpoint_filename"],
        map_location=args["device"],
        sae=sae,
    )

    # Disable randomness, dropout, etc
    model.eval()


# %%
# Prepare data loader

if args["mode"] == "autointerp":
    data_module.setup("fit")


# %%
# Inference

if args["mode"] == "autointerp":
    cfg = AutoInterpConfig(
        # latents=list[range(0, args["d_sae"], args["d_sae"] // 100)],
        latents=args["latents"],
        max_tokens=65536,
    )

    autointerp = AutoInterp(
        cfg=cfg,
        data_module=data_module,
        model=model,
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
    try:
        with open(args["result_filename"], "r") as f:
            loaded_results = json.load(f)
    except FileNotFoundError:
        loaded_results = {}

    combined_results = {**loaded_results, **results}


# %%
# Save results

if args["mode"] == "autointerp":
    os.makedirs(os.path.dirname(args["result_filename"]), exist_ok=True)

    with open(args["result_filename"], "w") as f:
        json.dump(combined_results, f, indent=2)

    print(f"Results saved to {args['result_filename']}")


# %%
# Load results

if args["mode"] == "verify":
    with open(args["result_filename"], "r") as f:
        loaded_results = json.load(f)

    print("Loaded results:")
    print(
        tabulate(
            [
                (k, v.get("explanation", ""), v.get("score", ""))
                for k, v in sorted(
                    loaded_results.items(),
                    key=lambda item: int(item[0]),
                )
            ],
            headers=["Latent", "Explanation", "Score"],
            tablefmt="github",
        )
    )

    # Check latent 7
    pprint.pprint(loaded_results["7"])


# %%
