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

import time

from sae_utils import (
    is_notebook,
    DataModule,
    LitModel,
)

from typing import (
    TypedDict,
    Literal,
)

import sys
import pprint

import argparse

import fairseq2

fairseq2.setup_fairseq2()

from sae_lens import (
    LanguageModelSAERunnerConfig,
    TrainingSAE,
    TrainingSAEConfig,
    GatedTrainingSAEConfig,
    LoggingConfig,
)

from lightning.pytorch.loggers import WandbLogger

import lightning as L

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
)

# %%
# Setup for notebook or script execution
if is_notebook():
    WORKSPACE_DIR = "/workspace/ALGOVERSE/UJR/jason"
    LOGGER_NAME = "gated-16384-lr_coef=2-e2"

    mode = "load_from_dict"  # "load_from_dict"

    sys.argv = [
        "main.py",
        "--workspace",
        WORKSPACE_DIR,
        "--debug",
        "--mode",
        mode,
    ]

    # mode=training
    if mode == "load_from_dict":
        sys.argv += [
            # Training hyperparameters
            "--d_sae",
            "16384",
            "--l1_coefficient",
            "0.05",
            "--l1_warm_up_steps",
            "3_000",
            "--total_training_batches",
            "30_000",
            "--lr_warm_up_steps",
            "3_000",
            "--lr_decay_steps",
            "6_000",
            "--batch_size",
            "4096",
            "--accumulate_grad_batches",
            "1",
            "--devices",
            "3",
            # WANDB
            "--logger_dir",
            f"{WORKSPACE_DIR}/experiments/sonar_sae",
            "--logger_name",
            LOGGER_NAME,
            # Checkpoints
            "--checkpoints_dir",
            f"{WORKSPACE_DIR}/experiments/sonar_sae/checkpoints",
        ]


# Argument parsing
class ArgsConfig(TypedDict):
    workspace: str

    debug: bool

    mode: Literal["load_from_dict"]

    # Training hyperparameters
    d_sae: int
    """
    e.g.: if SAE input dim is 1024, and you want a 16x overcomplete SAE,
    then d_sae = 16 * 1024 = 16384
    """

    total_training_batches: int
    """
    take into account the accumulate_grad_batches
    """

    l1_coefficient: float

    l1_warm_up_steps: int
    """
    e.g.: if total_training_steps = 30_000, and you want to warm up
    L1 regularization over the 5% of training steps,
    then l1_warm_up_steps = 0.05 * 30_000 = 1_500
    """

    lr_warm_up_steps: int
    """
    gradually increase learning rate over this many steps.
    """

    lr_decay_steps: int
    """
    gradually decrease learning rate to zero over this many steps.
    """

    batch_size: int
    """
    if you want effective batch size of 4096,
    make sure to set the batch_size where
    the GPU utilization is high (e.g. 16), 
    then set the accumulate_grad_batches 
    accordingly (e.g. 256).
    In other words, you should maximize
    GPU utilization and not VRAM usage
    """
    accumulate_grad_batches: int

    # Misc
    devices: list[int]

    # WANDB
    logger_dir: str
    logger_name: str

    # Checkpoints
    checkpoints_dir: str


def parse_args() -> ArgsConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workspace",
        type=str,
        help="Workspace directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (run only a few batches)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["load_from_dict"],
        help="Mode to run the script in",
    )

    parser.add_argument(
        "--d_sae",
        type=int,
        help="Dimensionality of the sparse autoencoder bottleneck",
    )

    parser.add_argument(
        "--l1_coefficient",
        type=float,
        help="Coefficient for L1 regularization",
    )

    parser.add_argument(
        "--l1_warm_up_steps",
        type=int,
        help="Number of warm-up steps for L1 regularization",
    )

    parser.add_argument(
        "--total_training_batches",
        type=int,
        help="Total number of training batches",
    )
    parser.add_argument(
        "--lr_warm_up_steps",
        type=int,
        help="Number of warm-up steps for learning rate",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        help="Number of decay steps for learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size (in number of sequences)",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        help="Number of batches to accumulate gradients over",
    )

    # Misc
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        help="List of device IDs to use",
    )

    # WANDB
    parser.add_argument(
        "--logger_dir",
        type=str,
        help="Directory to save WANDB logs",
    )
    parser.add_argument(
        "--logger_name",
        type=str,
        help="Name for the WANDB logger",
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()

    return ArgsConfig(**vars(args))


args = parse_args()
pprint.pprint(args)

# %%
# Load dataset

data_module = DataModule(
    batch_size=args["batch_size"],
)

if args["debug"]:
    data_module.setup()
    dataloader = data_module.train_dataloader()
    iterable = iter(dataloader)
    start = time.time()
    for i, batch in enumerate(iterable):
        end = time.time()
        print(f"Time taken to load batch {i}: {(end - start) * 1000:.1f} ms")
        start = time.time()
        if i >= 30:
            break
    print(f"Dataset keys: {batch.keys()}")
    print(f"embedding1 shape: {batch['embedding1'].shape}")
    print(f"lang1 shape: {len(batch['lang1'])}")
    print(f"Dataset examples:")
    pprint.pprint({k: v[:2] for k, v in batch.items() if k != "sequence_batch"})

# %%
# Set up SAE model

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distribution)
    # ignored, since we use our own dataset
    # SAE Parameters are in the nested `sae` config
    sae=GatedTrainingSAEConfig(
        d_in=1024,
        d_sae=args["d_sae"],
        apply_b_dec_to_input=True,
        normalize_activations="none",  # TODO: implementation
        l1_coefficient=args["l1_coefficient"],
        l1_warm_up_steps=args["l1_warm_up_steps"],
    ),
    # Training hyperparameters (standard)
    lr=5e-5,
    lr_warm_up_steps=args["lr_warm_up_steps"],
    # lr_warm_up_steps=0,  # TODO: remove
    lr_decay_steps=args["lr_decay_steps"],
    training_tokens=(
        args["total_training_batches"]
        * args["batch_size"]
        * args["accumulate_grad_batches"]
    ),
    train_batch_size_tokens=args["batch_size"],
    # Training hyperparameters (SAE-specific)
    # Activation Store Parameters
    # ignored, since we use our own dataset
    # WANDB
    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="sonar_sae",
        run_name=args["logger_name"],
        wandb_log_frequency=30,
        # wandb_log_frequency=1,  # TODO: remove
        eval_every_n_wandb_logs=20,
    ),
    # Misc
    n_checkpoints=10,
)

sae = TrainingSAE.from_dict(
    config_dict=TrainingSAEConfig.from_dict(cfg.get_training_sae_cfg_dict()).to_dict()
)

# %%
# Set up full model

if args["mode"] == "load_from_dict":
    model = LitModel(
        sae=sae,
        cfg=cfg.to_sae_trainer_config(),
    )

# %%
# Set up logger

print("Setting up logger...")
wandb_logger = WandbLogger(
    name=cfg.logger.run_name,
    save_dir=args["logger_dir"],
    project=cfg.logger.wandb_project,
)
wandb_logger.experiment.config.update(cfg.to_dict())

# %%
# Set up trainer
print("Setting up trainer...")

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{args['checkpoints_dir']}/{wandb_logger.experiment.id}",
    save_top_k=-1,
    every_n_train_steps=(args["total_training_batches"] // cfg.n_checkpoints),
    # every_n_train_steps=1,  # TODO: remove
)

trainer = L.Trainer(
    # Misc
    accelerator="gpu",
    devices=args["devices"],
    precision="bf16-mixed",
    accumulate_grad_batches=args["accumulate_grad_batches"],
    callbacks=[checkpoint_callback],
    logger=wandb_logger,
    log_every_n_steps=cfg.logger.wandb_log_frequency,
    max_steps=args["total_training_batches"],
    # max_steps=100, # TODO: remove
    reload_dataloaders_every_n_epochs=1,
)

trainer.fit(
    model=model,
    datamodule=data_module,
)

# %%
