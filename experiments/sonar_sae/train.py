# %%
# Auto-reload modules when code changes (for Jupyter notebooks)

from IPython import get_ipython

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except Exception:
    pass

# %% [markdown]
"""
Use modified libraries

or add in the `./project_directory/.env`
```
WORKSPACE=/workspace/jason
HF_HOME=$WORKSPACE/.cache/huggingface
PYTHONPATH=$WORKSPACE/SONAR:$WORKSPACE/SAELens
```
"""

# import sys

# sys.path.insert(0, "/workspace/ALGOVERSE/UJR/jason/SONAR")
# sys.path.insert(0, "/workspace/ALGOVERSE/UJR/jason/SAELens")

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
    BatchTopKTrainingSAEConfig,
    JumpReLUTrainingSAEConfig,
    LoggingConfig,
)

from lightning.pytorch.loggers import WandbLogger

import lightning as L

from lightning.pytorch.callbacks import (
    ModelCheckpoint,
)

from fairseq2.data.text.tokenizers import get_text_tokenizer_hub

from sonar.models.sonar_text import (
    get_sonar_text_decoder_hub,
    get_sonar_text_encoder_hub,
)

from sonar.models.sonar_translation import SonarEncoderDecoderModel

import torch

torch.set_float32_matmul_precision("highest")

# %%
# Setup for notebook or script execution
if is_notebook():
    WORKSPACE_DIR = "/workspace/jason"
    
    WIDTH = 2**17
    LR = 3e-4

    L1_COEFFICIENT = 1e-4
    LOGGER_NAME = f"gated-detach-norm-aux-mse-clip-{str(WIDTH)}-lr={str(LR)}-l1_coefficient={str(L1_COEFFICIENT)}"

    mode = "load_from_dict"  # "load_from_dict" | "load_from_checkpoint"

    sys.argv = [
        "main.py",
        "--debug",
        "--mode",
        mode,
    ]

    # mode=training
    SAE_TYPE = "gated"  # "gated" or "batchtopk"

    TOTAL_TRAINING_STEPS = 30_000

    sys.argv += [
        # Training hyperparameters
        "--sae_type",
        SAE_TYPE,
        "--d_sae",
        str(WIDTH),
        "--total_training_batches",
        str(TOTAL_TRAINING_STEPS),
        "--lr",
        str(LR),
        "--batch_size",
        "256",
        "--accumulate_grad_batches",
        "16",
        "--device",
        "0",
        # WANDB
        "--logger_dir",
        f"{WORKSPACE_DIR}/experiments/sonar_sae",
        "--logger_name",
        LOGGER_NAME,
        # Checkpoints
        "--checkpoints_dir",
        f"{WORKSPACE_DIR}/experiments/sonar_sae/checkpoints",
    ]

    if SAE_TYPE == "gated":
        sys.argv += [
            "--lr_warm_up_steps",
            str(TOTAL_TRAINING_STEPS // 10),  # 10% of training
            # "0",
            "--lr_decay_steps",
            str(TOTAL_TRAINING_STEPS // 5), # 20% of training
            # "0",
            "--l1_coefficient",
            str(L1_COEFFICIENT),
            "--l1_warm_up_steps",
            str(TOTAL_TRAINING_STEPS // 10),  # 10% of training
            # "0",
        ]
    elif SAE_TYPE == "batchtopk":
        sys.argv += [
            "--k",
            "96",
        ]
    elif SAE_TYPE == "jump_relu":
        sys.argv += [
            # Anthropic recommends decaying the LR for the final 20% of training
            "--lr_decay_steps",
            str(TOTAL_TRAINING_STEPS // 5),  # 20% of training
            "--l0_coefficient",
            "5",
            # Anthropic recommends using the full training steps for the warm-up
            "--l0_warm_up_steps",
            str(TOTAL_TRAINING_STEPS),  # full training
        ]

    if mode == "load_from_checkpoint":
        sys.argv += [
            "--checkpoint_filename",
            f"{WORKSPACE_DIR}/experiments/sonar_sae/checkpoints/tk4tyu7f/epoch=9-step=30000.ckpt",
        ]


# Argument parsing
class ArgsConfig(TypedDict):
    debug: bool

    mode: Literal["load_from_dict"]

    # Training hyperparameters
    d_sae: int
    """
    e.g.: if SAE input dim is 1024, and you want a 16x overcomplete SAE,
    then d_sae = 16 * 1024 = 16384
    """

    sae_type: Literal["gated", "batchtopk", "jump_relu"]

    total_training_batches: int
    """
    take into account the accumulate_grad_batches
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

    # Training SAE-specific hyperparameters

    ## Gated SAE-specific
    l1_coefficient: float

    l1_warm_up_steps: int
    """
    e.g.: if total_training_steps = 30_000, and you want to warm up
    L1 regularization over the 5% of training steps,
    then l1_warm_up_steps = 0.05 * 30_000 = 1_500
    """

    ## BatchTopK SAE-specific
    k: int

    # Misc
    device: int

    # WANDB
    logger_dir: str
    logger_name: str

    # Checkpoints
    checkpoints_dir: str

    checkpoint_filename: str


def parse_args() -> ArgsConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (run only a few batches)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["load_from_dict", "load_from_checkpoint"],
        help="Mode to run the script in",
    )

    # Training hyperparameters

    parser.add_argument(
        "--sae_type",
        type=str,
        choices=["gated", "batchtopk", "jump_relu"],
        help="Type of sparse autoencoder to use",
    )

    parser.add_argument(
        "--d_sae",
        type=int,
        help="Dimensionality of the sparse autoencoder bottleneck",
    )

    parser.add_argument(
        "--total_training_batches",
        type=int,
        help="Total number of training batches",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
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

    # Training SAE-specific hyperparameters

    ## Gated SAE-specific
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

    ## BatchTopK SAE-specific
    parser.add_argument(
        "--k",
        type=int,
        help="Number of top activations to keep in BatchTopK SAE",
    )

    ## JumpReLU SAE-specific
    parser.add_argument(
        "--l0_coefficient",
        type=float,
        help="Coefficient for L0 regularization",
    )
    parser.add_argument(
        "--l0_warm_up_steps",
        type=int,
        help="Number of warm-up steps for L0 regularization",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=int,
        help="Which GPU to use",
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

    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        help="Path to checkpoint file",
    )

    args, unknown = parser.parse_known_args()

    return ArgsConfig(**vars(args))


args = parse_args()
pprint.pprint(args)

# %% TODO remove

# a = torch.load(
#     "/workspace/ALGOVERSE/UJR/jason/experiments/sonar_sae/checkpoints/gkkzzviw/epoch=9-step=29516 copy.ckpt",
#     weights_only=False,
# )
# a["state_dict"] = {
#     k: v
#     for k, v in a["state_dict"].items()
#     if not k.startswith("model.")
# }
# torch.save(
#     a,
#     "/workspace/ALGOVERSE/UJR/jason/experiments/sonar_sae/checkpoints/gkkzzviw/epoch=9-step=29516-fixed.ckpt",
# )

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

if args["sae_type"] == "gated":
    sae_cfg = GatedTrainingSAEConfig(
        d_in=1024,
        d_sae=args["d_sae"],
        apply_b_dec_to_input=True,
        normalize_activations="none",  # TODO: implementation
        l1_coefficient=args["l1_coefficient"],
        l1_warm_up_steps=args["l1_warm_up_steps"],
        # l1_warm_up_steps=args["total_training_batches"],
        normalize_decoder=True,
        # Misc
        device=f"cuda:{args['device']}",
    )
elif args["sae_type"] == "batchtopk":
    sae_cfg = BatchTopKTrainingSAEConfig(
        d_in=1024,
        d_sae=args["d_sae"],
        apply_b_dec_to_input=True,
        normalize_activations="none",  # TODO: implementation
        k=args["k"],
        # Misc
        device=f"cuda:{args['device']}",
    )
elif args["sae_type"] == "jump_relu":
    sae_cfg = JumpReLUTrainingSAEConfig(
        d_in=1024,
        d_sae=args["d_sae"],
        apply_b_dec_to_input=True,
        l0_coefficient=args["l0_coefficient"],  # Sparsity penalty coefficient
        jumprelu_sparsity_loss_mode="tanh",
        jumprelu_tanh_scale=4.0,  # default value
        jumprelu_bandwidth=2.0,
        jumprelu_init_threshold=0.1,
        pre_act_loss_coefficient=3e-6,
        # Anthropic's settings assume normalized activations
        normalize_activations="expected_average_only_in",
        # Anthropic recommends using the full training steps for the warm-up
        l0_warm_up_steps=args["l0_warm_up_steps"],
        # Misc
        device=f"cuda:{args['device']}",
    )

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distribution)
    # ignored, since we use our own dataset
    # SAE Parameters are in the nested `sae` config
    sae=sae_cfg,
    # Training hyperparameters (standard)
    lr=args["lr"],
    lr_warm_up_steps=args["lr_warm_up_steps"] or 0,
    # lr_warm_up_steps=0,  # TODO: remove
    lr_decay_steps=args["lr_decay_steps"] or 0,
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
        eval_every_n_wandb_logs=(
            args["total_training_batches"] // 10
        ),  # n_checkpoints=10
    ),
    # feature_sampling_window=100,  # TOOD: remove
    # Misc
    n_checkpoints=10,
)

sae = TrainingSAE.from_dict(
    config_dict=TrainingSAEConfig.from_dict(cfg.get_training_sae_cfg_dict()).to_dict()
)

# %%
# Load model

tokenizer_hub = get_text_tokenizer_hub()
encoder_hub = get_sonar_text_encoder_hub()
encoder = encoder_hub.load(name_or_card="text_sonar_basic_encoder")
encoder_tokenizer = tokenizer_hub.load(name_or_card="text_sonar_basic_encoder")
decoder_hub = get_sonar_text_decoder_hub()
decoder = decoder_hub.load(name_or_card="text_sonar_basic_decoder")
decoder_tokenizer = tokenizer_hub.load(name_or_card="text_sonar_basic_decoder")
model = SonarEncoderDecoderModel(encoder=encoder, decoder=decoder)

# %%
# Set up full model

lit_model = LitModel(
    # cfg=cfg.to_sae_trainer_config(),
    cfg=cfg,
    model=model,
    encoder_tokenizer=encoder_tokenizer,
    decoder_tokenizer=decoder_tokenizer,
    sae=sae,
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

if args["sae_type"] == "batchtopk":
    monitor = "model_performance_preservation.ce_loss_score"
    mode = "max"
else:
    monitor = "metrics/l0"
    mode = "min"

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{args['checkpoints_dir']}/{wandb_logger.experiment.id}",
    # monitor="model_performance_preservation.ce_loss_score",
    # mode="max",
    # monitor="metrics/l0",
    # mode="min",
    monitor=monitor,
    mode=mode,
    save_top_k=cfg.n_checkpoints,
    save_last=True,
    # every_n_train_steps=(args["total_training_batches"] // cfg.n_checkpoints),
    # every_n_train_steps=2,  # TODO: remove
)

trainer = L.Trainer(
    # Misc
    accelerator="gpu",
    devices=[args["device"]],
    precision="32-true",
    accumulate_grad_batches=args["accumulate_grad_batches"],
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    callbacks=[checkpoint_callback],
    logger=wandb_logger,
    log_every_n_steps=cfg.logger.wandb_log_frequency,
    val_check_interval=0.25,
    # val_check_interval=1 * args["accumulate_grad_batches"],  # TODO: remove
    max_steps=args["total_training_batches"],
    # max_steps=35000,  # TODO: remove
    # reload_dataloaders_every_n_epochs=1,
)

trainer.fit(
    model=lit_model,
    datamodule=data_module,
    # ckpt_path="/workspace/ALGOVERSE/UJR/jason/experiments/sonar_sae/checkpoints/gkkzzviw/epoch=9-step=29516-fixed.ckpt",
    ckpt_path=args["checkpoint_filename"],
)

# %%
