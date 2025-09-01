# %%
# Auto-reload modules when code changes (for Jupyter notebooks)

from IPython import get_ipython

try:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
except Exception:
    pass

# %%
# Modify SAELens library

import sys

sys.path.insert(0, "/workspace/ALGOVERSE/UJR/jason/SAELens")

# %%

import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    GatedTrainingSAEConfig,
    LoggingConfig,
)

# Define total training steps and batch size
total_training_steps = 30_000
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

# Learning rate and L1 warmup schedules
lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

device = "cuda:3"

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distribution)
    model_name="tiny-stories-1L-21M",
    hook_name="blocks.0.hook_mlp_out",
    dataset_path="apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
    is_dataset_tokenized=True,
    streaming=True,
    # SAE Parameters are in the nested 'sae' config
    sae=GatedTrainingSAEConfig(
        d_in=1024,  # Matches hook_mlp_out for tiny-stories-1L-21M
        d_sae=16 * 1024,
        apply_b_dec_to_input=True,
        normalize_activations="none",
        l1_coefficient=5,
        l1_warm_up_steps=l1_warm_up_steps,
    ),
    # Training Parameters
    lr=5e-5,
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_decay_steps,
    train_batch_size_tokens=batch_size,
    # Activation Store Parameters
    context_size=256,
    n_batches_in_buffer=64,
    training_tokens=total_training_tokens,
    store_batch_size_prompts=16,
    # WANDB
    logger=LoggingConfig(
        log_to_wandb=True,
        wandb_project="sae_lens_tutorial",
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
    ),
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32",
)
sparse_autoencoder = LanguageModelSAETrainingRunner(cfg).run()

# %%
