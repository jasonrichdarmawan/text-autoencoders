# %% [markdown]
"""
Problem 1:
Computing the ROUGE score requires both predicted 
and target texts. However, generating the predicted 
text involves processing embeddings which, after 
passing through an untrained linear transformation, 
can cause the model to produce excessively long 
outputs. For example, instead of the expected 
"hello", the model might generate "It is an 
integral part of the conversation." As a result, 
processing 16 embeddings takes approximately 10 
seconds on average.
"""

# %%

# !pip install "datasets>=3,<4"

# %%

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
  print(f"Adding project root to sys.path: {project_root}")
  sys.path.append(project_root)

# %%

from dp_utils import Task

from argparse import ArgumentParser
from typing import TypedDict

from datasets import load_from_disk
from datasets import load_dataset

import random
from dataclasses import dataclass

from jaxtyping import Float

import torch as t
import torch.nn as nn
from torch import Tensor
from torch import distributed as dist
import torch.nn.functional as F
from torch.utils.data import (
  IterableDataset,
  DataLoader,
  DistributedSampler,
)

from pytorch_lightning import (
  LightningDataModule,
  LightningModule,
  Trainer,
)
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers.modeling_outputs import ModelOutput
from transformers.optimization import get_linear_schedule_with_warmup

from sonar.inference_pipelines.text import (
  TextToEmbeddingModelPipeline,
  EmbeddingToTextModelPipeline,
)

from fairseq2 import setup_fairseq2
from fairseq2.data import Collater
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.data.text.tokenizers import get_text_tokenizer_hub

# from torchmetrics.text.rouge import ROUGEScore

# from time import sleep # TODO: remove this later

# %%

print("Setting float32 matmul precision to 'high'...")  # For better performance
t.set_float32_matmul_precision('high')

# %%

setup_fairseq2()

# %%

class Args(TypedDict):
  data_dir: str
  checkpoint_path: str | None
  default_root_dir: str
  accelerator: str
  strategy: str
  devices: list[int]
  batch_size: int
  lr: float
  num_warmup_steps: int
  max_steps: int
  """
  ```
  max_steps = (
    num_samples 
    // (devices * batch_size) 
    * max_epochs
  )
  ```
  For example, `( 989944 // (3 * 8) + 989944 // (3 * 8)) * 10 = 824940 steps`
  """

def parse_args() -> Args:
  parser = ArgumentParser(description="Data Preprocessing Script")
  parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Path to the workspace directory where data will be processed.",
  )
  parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to the model checkpoint for loading.",
  )
  parser.add_argument(
    "--default_root_dir",
    type=str,
    help="Default root directory for PyTorch Lightning logs.",
  )
  parser.add_argument(
    "--accelerator",
    type=str,
    help="Type of accelerator to use (e.g., 'gpu', 'cpu').",
  )
  parser.add_argument(
    "--strategy",
    type=str,
    help="Distributed training strategy (e.g., 'ddp', 'dp').",
  )
  parser.add_argument(
    "--devices",
    nargs="+",
    type=int,
    required=True,
  )
  parser.add_argument(
    "--batch_size",
    type=int,
    help="Batch size per GPU for training.",
  )
  parser.add_argument(
    "--lr",
    type=float,
    help="Learning rate for the optimizer.",
  )
  parser.add_argument(
    "--num_warmup_steps",
    type=int,
    help="Number of warmup steps for the learning rate scheduler.",
  )
  parser.add_argument(
    "--max_steps",
    type=int,
    help="Total number of training steps.",
  )
  args = parser.parse_args()
  return Args(**vars(args))

if False:
  DATA_DIR = "/workspace/ALGOVERSE/UJR/jason/data"
  sys.argv = [
    'main.py',
    '--data_dir', DATA_DIR,
    '--accelerator', 'gpu',
    '--strategy', 'ddp',
    '--devices', '1', '2', '3',
    '--batch_size', '16',
    '--lr', '1e-4',
    '--num_warmup_steps', '500',
    '--max_steps', '1000',
  ]

args = parse_args()

print("Parsed arguments:")
for key, value in args.items():
  print(f"{key}: {value}")

# %%

class DataCollator:
  def __init__(self, pad_idx: int):
    self.pad_idx = pad_idx

  def __call__(self, batch):
    input_a = [t.tensor(item["input_a"]) for item in batch]
    input_b = [t.tensor(item["input_b"]) for item in batch]
    input_target = [t.tensor(item["input_target"]) for item in batch]

    # max_len_input_a = max(l.size(0) for l in input_a)
    # max_len_input_b = max(l.size(0) for l in input_b)
    # max_len_input_target = max(l.size(0) for l in input_target)
    # max_len = max(max_len_input_a, max_len_input_b, max_len_input_target)
    
    # input_a_padded = []
    # for seq in input_a:
    #   if seq.size(0) < max_len:
    #     pad_amt = max_len - seq.size(0)
    #     seq = F.pad(seq, (0, pad_amt), value=self.pad_idx)
    #   input_a_padded.append(seq)
    # input_a_padded = t.stack(input_a_padded, dim=0)

    # input_b_padded = []
    # for seq in input_b:
    #   if seq.size(0) < max_len:
    #     pad_amt = max_len - seq.size(0)
    #     seq = F.pad(seq, (0, pad_amt), value=self.pad_idx)
    #   input_b_padded.append(seq)
    # input_b_padded = t.stack(input_b_padded, dim=0)

    # input_target_padded = []
    # for seq in input_target:
    #   if seq.size(0) < max_len:
    #     pad_amt = max_len - seq.size(0)
    #     seq = F.pad(seq, (0, pad_amt), value=self.pad_idx)
    #   input_target_padded.append(seq)
    # input_target_padded = t.stack(input_target_padded, dim=0)

    return {
      "input_a": input_a,
      "input_b": input_b,
      "input_target": input_target,
      # "input_a": input_a_padded,
      # "input_b": input_b_padded,
      # "input_target": input_target_padded,
    }

class TaskDataModule(LightningDataModule):
  def __init__(
    self,
    data_dir: str,
    task: Task,
    batch_size: int,
  ):
    super().__init__()
    
    self.data_dir = data_dir
    self.task = task
    self.batch_size = batch_size
    
    tokenizer_hub = get_text_tokenizer_hub()
    self.tokenizer = tokenizer_hub.load(
      name_or_card="text_sonar_basic_encoder",
    )
    self.collator = DataCollator(pad_idx=self.tokenizer.vocab_info.pad_idx)

    self.train_dataset = None
    self.val_dataset = None
    self.test_dataset = None

  def prepare_data(self):
    # download, split, etc...
    # only called on 1 GPU/TPU in distributed
    self._prepare_data_wiki_split()

  def setup(self, stage):
    self._setup_load_from_disk(
      stage=stage,
      path="google-research-datasets/wiki_split",
    )
  
  def train_dataloader(self):
    """
    Reference:
    [1] https://lightning.ai/docs/pytorch/stable/common/trainer.html#use-distributed-sampler
    """
    sampler = (
      DistributedSampler(self.train_dataset, shuffle=True)
      if dist.is_available() and dist.is_initialized()
      else None
    )
    dataloader = TaskDataLoader(
      task=self.task,
      sampler=sampler,
      shuffle=sampler is None,
      dataset=self.train_dataset,
      batch_size=self.batch_size,
      drop_last=True,
      collate_fn=self.collator,
      num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 1,
    )
    return dataloader

  def val_dataloader(self):
    sampler = (
      DistributedSampler(self.val_dataset, shuffle=False)
      if dist.is_available() and dist.is_initialized()
      else None
    )
    dataloader = TaskDataLoader(
      task=self.task,
      sampler=sampler,
      shuffle=sampler is None,
      dataset=self.val_dataset,
      batch_size=self.batch_size,
      drop_last=False,
      collate_fn=self.collator,
      num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 1,
    )
    return dataloader
  
  def test_dataloader(self):
    sampler = (
      DistributedSampler(self.test_dataset, shuffle=False)
      if dist.is_available() and dist.is_initialized()
      else None
    )
    dataloader = TaskDataLoader(
      task=self.task,
      sampler=sampler,
      shuffle=sampler is None,
      dataset=self.test_dataset,
      batch_size=self.batch_size,
      drop_last=False,
      collate_fn=self.collator,
      num_workers=os.cpu_count() // 2 if os.cpu_count() > 1 else 1,
    )
    return dataloader
  
  def _prepare_data_wiki_split(self):
    def preprocess(item):
      complex_sentence = item['complex_sentence']
      simple_sentence_1 = item['simple_sentence_1']
      simple_sentence_2 = item['simple_sentence_2']

      flip = random.random() < 0.5

      match self.task:
        case Task.ADD:
          input_a = simple_sentence_1
          input_b = simple_sentence_2
          input_target = complex_sentence
        case Task.DIFFERENCE:
          input_a = complex_sentence
          if flip:
            input_b = simple_sentence_2
            input_target = simple_sentence_1
          else:
            input_b = simple_sentence_1
            input_target = simple_sentence_2
        case _:
          raise ValueError(f"Unknown task: {task}")
      
      tokenizer_encoder = self.tokenizer.create_encoder(
        lang="eng_Latn",
      )
      input_a_tokens = tokenizer_encoder(input_a)
      input_b_tokens = tokenizer_encoder(input_b)
      input_target_tokens = tokenizer_encoder(input_target)

      return {
        # "input_a": input_a,
        # "input_b": input_b,
        # "input_target": input_target,
        "input_a": input_a_tokens,
        "input_b": input_b_tokens,
        "input_target": input_target_tokens,
      }

    try:
      load_from_disk(
        os.path.join(
          self.data_dir,
          "google-research-datasets",
          "wiki_split",
          "processed",
          f"{self.task.value}_train",
        )
      )
      load_from_disk(
        os.path.join(
          self.data_dir,
          "google-research-datasets",
          "wiki_split",
          "processed",
          f"{self.task.value}_val",
        )
      )
      load_from_disk(
        os.path.join(
          self.data_dir,
          "google-research-datasets",
          "wiki_split",
          "processed",
          f"{self.task.value}_test",
        )
      )
      return # TODO: uncomment this later
    except FileNotFoundError:
      print(f"Data for task {self.task.value} not found, downloading and processing...")
    
    dataset = load_dataset(
      path="google-research-datasets/wiki_split",
    )

    train_dataset = dataset['train'].map(
      preprocess, 
      remove_columns=dataset['train'].column_names,
      # batched=True,
      # batch_size=5,
    )

    val_dataset = dataset['validation'].map(
      preprocess, 
      remove_columns=dataset['validation'].column_names,
      # batched=True,
      # batch_size=5,
    )

    test_dataset = dataset['test'].map(
      preprocess, 
      # batched=True,
      # batch_size=5,
      remove_columns=dataset['test'].column_names,
    )

    train_dataset.save_to_disk(
      os.path.join(
        self.data_dir, 
        "google-research-datasets",
        "wiki_split",
        "processed",
        f"{self.task.value}_train",
      )
    )
    val_dataset.save_to_disk(
      os.path.join(
        self.data_dir, 
        "google-research-datasets",
        "wiki_split",
        "processed",
        f"{self.task.value}_val",
      ),
    )
    test_dataset.save_to_disk(
      os.path.join(
        self.data_dir, 
        "google-research-datasets",
        "wiki_split",
        "processed",
        f"{self.task.value}_test",
      ),
    )

  def _setup_load_from_disk(
    self, 
    stage: str, 
    path: str,
  ):
    if stage == "fit":
      self.train_dataset = load_from_disk(
        os.path.join(
          self.data_dir,
          path,
          "processed",
          f"{self.task.value}_train",
        )
      )
      self.val_dataset = load_from_disk(
        os.path.join(
          self.data_dir,
          path,
          "processed",
          f"{self.task.value}_val",
        )
      )
      rank_zero_info(f"Loaded {path} {self.task.value} train dataset with {len(self.train_dataset)} samples.")
      rank_zero_info(f"Loaded {path} {self.task.value} val dataset with {len(self.val_dataset)} samples.")
    elif stage == "test":
      self.test_dataset = load_from_disk(
        os.path.join(
          self.data_dir,
          path,
          "processed",
          f"{self.task.value}_test",
        )
      )
      rank_zero_info(f"Loaded {path} {self.task.value} test dataset with {len(self.test_dataset)} samples.")
    else:
      raise ValueError(f"Unknown stage: {stage}")
  
  def teardown(self, stage):
    if stage == "fit":
      self.train_dataset = None
      self.val_dataset = None
      rank_zero_info("Teardown: train and val datasets set to None.")
    elif stage == "test":
      self.test_dataset = None
      rank_zero_info("Teardown: test dataset set to None.")
    else:
      raise ValueError(f"Unknown stage: {stage}")

class TaskDataLoader(DataLoader):
  def __init__(self, task: Task, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.task = task

  def __iter__(self):
    for batch in super().__iter__():
      batch["task"] = self.task
      yield batch

class MultiTaskDataModule(LightningDataModule):
  def __init__(
    self, 
    data_dir: str,
    tasks: list[Task],
    batch_size: int,
  ):
    super().__init__()
    self.datasets: dict[Task, TaskDataModule] = {}
    for task in tasks:
      self.datasets[task] = TaskDataModule(
        data_dir=data_dir,
        task=task,
        batch_size=batch_size,
      )

  def prepare_data(self):
    for dataset in self.datasets.values():
      dataset.prepare_data()
  
  def setup(self, stage):
    for dataset in self.datasets.values():
      dataset.setup(stage=stage)
  
  def teardown(self, stage):
    for dataset in self.datasets.values():
      dataset.teardown(stage=stage)

  def train_dataloader(self):
    dataset = MultiTaskIterableDataset(
      dataloader_dict={
        task: dataset.train_dataloader() 
        for task, dataset 
        in self.datasets.items()
      }
    )
    dataloader = DataLoader(
      dataset=dataset,
      collate_fn=lambda x: x[0], # Prevent extra tuple wrapping
    )
    return dataloader
  
  def val_dataloader(self):
    dataset = MultiTaskIterableDataset(
      dataloader_dict={
        task: dataset.val_dataloader() 
        for task, dataset 
        in self.datasets.items()
      }
    )
    dataloader = DataLoader(
      dataset=dataset,
      collate_fn=lambda x: x[0], # Prevent extra tuple wrapping
    )
    return dataloader
  
  def test_dataloader(self):
    dataset = MultiTaskIterableDataset(
      dataloader_dict={
        task: dataset.test_dataloader() 
        for task, dataset 
        in self.datasets.items()
      }
    )
    dataloader = DataLoader(
      dataset=dataset,
      collate_fn=lambda x: x[0], # Prevent extra tuple wrapping
    )
    return dataloader

class MultiTaskIterableDataset(IterableDataset):
  def __init__(
    self, 
    dataloader_dict: dict[str, DataLoader],
  ):
    super().__init__()
    self.dataloader_dict = dataloader_dict
    self.task_list = list(dataloader_dict.keys())

    # self.dataloader_dict = dataloader_dict
    # self.num_batches_dict = {
    #   task: len(dataloader)
    #   for task, dataloader in dataloader_dict.items()
    # }
    # self.task_list = list(dataloader_dict.keys())
    # self.world_size = 1
    # self.rank = 0
    # if dist.is_initialized():
    #   self.world_size = dist.get_world_size()
    #   self.rank = dist.get_rank()

  # def __len__(self):
  #   total_batches = sum(self.num_batches_dict.values())
  #   return total_batches

  def __len__(self):
    total_batches = sum(len(dl) for dl in self.dataloader_dict.values())
    return total_batches

  def __iter__(self):
    dataloader_dicts = {
      task: iter(dataloader) 
      for task, dataloader 
      in self.dataloader_dict.items()
    }
    while True:
      for task in self.task_list:
        try:
          yield next(dataloader_dicts[task])
        except StopIteration:
          self.task_list.remove(task)
          if not self.task_list:
            return

    # all_batches = []
    # for dataloader in self.dataloader_dict.values():
    #   all_batches.extend(iter(dataloader))

    # random.shuffle(all_batches)

    # for i in range(0, len(all_batches)):
    #   yield all_batches[i]

class Add(nn.Module):
  def __init__(self, n_embd: int):
    super().__init__()
    self.linear = nn.Linear(2*n_embd, n_embd)

  def forward(
    self, 
    x: Float[Tensor, "batch_size 2*n_embd"],
  ):
    x = self.linear(x)
    return x

class Difference(nn.Module):
  def __init__(self, n_embd: int):
    super().__init__()
    self.linear = nn.Linear(2*n_embd, n_embd)

  def forward(
    self, 
    x: Float[Tensor, "batch_size 2*n_embd"],
  ):
    x = self.linear(x)
    return x

@dataclass
class EncoderModelOutput(ModelOutput):
  embeddings_sim: Float[Tensor, "batch_size n_embd"]
  embeddings_pos: Float[Tensor, "batch_size n_embd"] | None = None
  # target_seqs: list[str] | None = None

class EncoderModel(nn.Module):
  def __init__(
    self,
  ):
    super().__init__()
    self.encoder = TextToEmbeddingModelPipeline(
      tokenizer="text_sonar_basic_encoder",
      encoder="text_sonar_basic_encoder",
    )
    
    self.n_embd = 1024

    # self.norm1 = nn.RMSNorm(self.n_embd)
    self.add = Add(n_embd=self.n_embd)
    self.difference = Difference(n_embd=self.n_embd)
    # self.norm2 = nn.RMSNorm(self.n_embd)

    for param in self.encoder.model.parameters():
      param.requires_grad = False

  def forward(
    self, 
    task: Task,
    input_a: list[Float[Tensor, "seq_len"]],
    input_b: list[Float[Tensor, "seq_len"]], 
    input_target: list[Float[Tensor, "seq_len"]] | None = None,
  ):
    # Version 1: input_a, input_b, input_target are list[str]
    # input = input_a + input_b
    # if input_target is not None:
    #   input += input_target

    # device = next(self.encoder.model.parameters()).device
    # tokenizer_encoder = self.encoder.tokenizer.create_encoder(
    #   lang="eng_Latn",
    #   device=device,
    # )

    # seqs = [
    #   tokenizer_encoder(seq)
    #   for seq in input
    # ]
    # collater = Collater(
    #   pad_value=self.encoder.tokenizer.vocab_info.pad_idx,
    # )
    # batch = collater(seqs)
    # tokens, padding_mask = get_seqs_and_padding_mask(
    #   data=batch, 
    #   device=device,
    # )

    # Version 2: input_a, input_b, input_target are list[Float[Tensor, "seq_len"]]
    seqs = input_a + input_b
    if input_target is not None:
      seqs += input_target
    
    collater = Collater(
      pad_value=self.encoder.tokenizer.vocab_info.pad_idx,
    )
    batch = collater(seqs)
    device = next(self.encoder.model.parameters()).device
    tokens, padding_mask = get_seqs_and_padding_mask(
      data=batch, 
      device=device,
    )

    sequence_batch = SequenceBatch(tokens, padding_mask)
    embeddings = self.encoder.model(sequence_batch).sentence_embeddings

    # embeddings = self.norm1(embeddings)

    batch_size = len(input_a)
    embeddings_a = embeddings[0:batch_size]
    embeddings_b = embeddings[batch_size:2*batch_size]
    embeddings_target = embeddings[2*batch_size:] if input_target is not None else None

    embeddings_a_b = t.cat([embeddings_a, embeddings_b], dim=1)

    match task:
      case Task.ADD:
        embeddings_sim = self.add(embeddings_a_b)
      case Task.DIFFERENCE:
        embeddings_sim = self.difference(embeddings_a_b)
      case _:
        raise ValueError(f"Unknown task: {task}")

    # embeddings_sim = self.norm2(embeddings_sim)
    embeddings_pos = embeddings_target

    return EncoderModelOutput(
      embeddings_sim=embeddings_sim,
      embeddings_pos=embeddings_pos,
      # target_seqs=seqs[2*batch_size:] if input_target is not None else None,
    )

@dataclass
class EncoderDecoderModelOutput(ModelOutput):
  embeddings: Float[Tensor, "batch_size n_embd"]
  contrastive_loss: Float[Tensor, ""] | None = None
  reconstruction_loss: Float[Tensor, ""] | None = None
  loss: Float[Tensor, ""] | None = None
  # decoded_outputs: list[str] | None = None
  # decoded_targets: list[str] | None = None

class EncoderDecoderModel(nn.Module):
  def __init__(
    self,
  ):
    super().__init__()
    self.encoder = EncoderModel()
    self.decoder = EmbeddingToTextModelPipeline(
      tokenizer="text_sonar_basic_decoder",
      decoder="text_sonar_basic_decoder",
    )

    for param in self.decoder.model.parameters():
      param.requires_grad = False

  def forward(
    self, 
    task: Task,
    input_a: list[Float[Tensor, "seq_len"]], 
    input_b: list[Float[Tensor, "seq_len"]], 
    input_target: list[Float[Tensor, "seq_len"]] | None = None,
    # return_decoded: bool = False,
  ):
    encoder_output = self.encoder(
      task=task,
      input_a=input_a,
      input_b=input_b,
      input_target=input_target,
    )
    
    embeddings_sim = encoder_output.embeddings_sim # shape: (batch_size, 2, n_embd)
    embeddings_pos = encoder_output.embeddings_pos
    # target_seqs = encoder_output.target_seqs

    output_kwargs = {
      "embeddings": embeddings_sim,
    }

    # if return_decoded:
    #   batch_size = embeddings_sim.shape[0]
    #   inputs = (
    #     t.cat(
    #       [embeddings_sim, embeddings_pos], 
    #       dim=0,
    #     )
    #     if embeddings_pos is not None 
    #     else embeddings_sim
    #   )
    #   decoded = self.decoder.predict(
    #     inputs=inputs,
    #     target_lang="eng_Latn",
    #     max_seq_len=512,
    #   )
    #   decoded_outputs = decoded[:batch_size]
    #   decoded_targets = (
    #     decoded[batch_size:] 
    #     if embeddings_pos is not None 
    #     else None
    #   )
    #   output_kwargs.update({
    #     "decoded_outputs": decoded_outputs,
    #     "decoded_targets": decoded_targets,
    #   })

    if input_target is not None:
      cl_loss = self.compute_contrastive_loss(
        embeddings_sim=embeddings_sim, 
        embeddings_pos=embeddings_pos,
      )
      gen_loss = self.compute_reconstruction_loss(
        input_target=input_target,
        embeddings_sim=embeddings_sim,
      )
      gen_loss_weight = 0.01
      loss = cl_loss + gen_loss * gen_loss_weight
      output_kwargs.update({
        "contrastive_loss": cl_loss,
        "reconstruction_loss": gen_loss,
        "loss": loss,
      })

    return EncoderDecoderModelOutput(**output_kwargs)
  
  def compute_contrastive_loss(
    self, 
    embeddings_sim: Float[Tensor, "batch_size n_embd"], 
    embeddings_pos: Float[Tensor, "batch_size n_embd"]
  ):
    # Contrastive learning loss
    sim_fct = nn.CosineSimilarity(dim=-1)
    temperature = 0.05
    cos_sim = sim_fct(
      embeddings_sim.unsqueeze(dim=1), # shape: (batch_size, 1, n_embd)
      embeddings_pos.unsqueeze(dim=0), # shape: (1, batch_size, n_embd)
    ) / temperature
    batch_size = embeddings_sim.shape[0]
    cl_labels = t.arange(batch_size, device=cos_sim.device)
    loss_fct = nn.CrossEntropyLoss()
    cl_loss = loss_fct(cos_sim, cl_labels)
    return cl_loss

  def compute_reconstruction_loss(
    self, 
    input_target: list[Float[Tensor, "seq_len"]],
    embeddings_sim: Float[Tensor, "batch_size n_embd"],
  ):
    # Generation loss
    loss_fct = nn.CrossEntropyLoss()

    device = next(self.decoder.model.parameters()).device
    input_to_decoder = [
      t.cat([
        t.tensor(
          [self.decoder.tokenizer.vocab_info.eos_idx],
          device=device,
        ),
        seq[1:-1],  # Remove first and last token (language token and EOS)
      ])
      for seq in input_target
    ]
    collater = Collater(
      pad_value=self.decoder.tokenizer.vocab_info.pad_idx,
    )
    input_data = collater(input_to_decoder)
    input_tensor, input_padding_mask = get_seqs_and_padding_mask(
      data=input_data, 
      device=device,
    )
    decoder_output, decoder_padding_mask = self.decoder.model.decode(
      seqs=input_tensor,
      padding_mask=input_padding_mask,
      encoder_output=embeddings_sim.unsqueeze(1),
      encoder_padding_mask=None,
    )

    model_output = self.decoder.model.project(
      decoder_output, 
      decoder_padding_mask,
    )

    logits = model_output.logits

    labels = [
      seq[1:]
      for seq in input_target
    ]

    max_len_logits = logits.size(1)
    max_len_labels = max(l.size(0) for l in labels)
    max_len = max(max_len_logits, max_len_labels)

    logits_padded = logits
    if logits.size(1) < max_len:
      pad_amt = max_len - logits.size(1)
      logits_padded = F.pad(
        logits, (0, 0, 0, pad_amt), 
        value=self.decoder.tokenizer.vocab_info.pad_idx,
      )
    
    labels_padded = []
    for label in labels:
      if label.size(0) < max_len:
        pad_amt = max_len - label.size(0)
        label = F.pad(
          label, (0, pad_amt), 
          value=self.decoder.tokenizer.vocab_info.pad_idx,
        )
      labels_padded.append(label)
    labels_padded = t.stack(labels_padded, dim=0)

    gen_loss = loss_fct(
      logits_padded.reshape(-1, logits_padded.size(-1)), # (batch_size * seq_len, vocab_size)
      labels_padded.reshape(-1)                          # (batch_size * seq_len)
    )
    return gen_loss

class LitModel(LightningModule):
  def __init__(
    self,
    lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
  ):
    super().__init__()
    self.save_hyperparameters()

    self.lr = lr
    self.num_warmup_steps = num_warmup_steps
    self.num_training_steps = num_training_steps
    self.model = EncoderDecoderModel()
    # self.dummy_parameter = nn.Parameter(t.tensor(0.0, requires_grad=True))

  def on_fit_start(self):
    device = next(self.model.decoder.model.parameters()).device
    self.model.decoder.device = device

  def get_trainable_state_dict(self):
    state = {}
    for name, param in self.named_parameters():
      if param.requires_grad:
        state[name] = param.data.cpu()
    for name, buffer in self.named_buffers():
      state[name] = buffer.data.cpu()
    return state

  def on_save_checkpoint(self, checkpoint):
    checkpoint['state_dict'] = self.get_trainable_state_dict()
    # def print_keys(d, indent=0):
    #   if isinstance(d, dict):
    #     for key, value in d.items():
    #       rank_zero_info(f"{' ' * indent}Key: {key}")
    #       print_keys(value, indent=indent+1)
    # print_keys(checkpoint)

  def training_step(self, batch):
    outputs = self.model(
      task=batch['task'],
      input_a=batch['input_a'],
      input_b=batch['input_b'],
      input_target=batch['input_target'],
    )
    
    cl_loss = outputs.contrastive_loss
    gen_loss = outputs.reconstruction_loss
    loss = outputs.loss

    batch_size = len(batch['input_a'])
    lr = self.trainer.optimizers[0].param_groups[0]['lr']
    self.log(
      "lr", 
      lr, 
      prog_bar=True, 
      logger=True,
      on_step=True,
      on_epoch=False,
    )
    self.log(
      "train_cl_loss", 
      cl_loss, 
      prog_bar=True, 
      logger=True,
      on_step=True,
      on_epoch=False,
      sync_dist=True,
      batch_size=batch_size,
    )
    self.log(
      "train_gen_loss", 
      gen_loss, 
      prog_bar=True, 
      logger=True,
      on_step=True,
      on_epoch=False,
      sync_dist=True,
      batch_size=batch_size,
    )
    self.log(
      "train_loss", 
      loss, 
      prog_bar=True, 
      logger=True,
      on_step=True,
      on_epoch=False,
      sync_dist=True,
      batch_size=batch_size,
    )
    return loss
    
    # print(
    #   f"[GPU {self.global_rank}] "
    #   f"batch_idx: {batch_idx}, "
    #   f"task: {batch['task']}, "
    #   f"input_a: {batch['input_a']}, "
    #   f"input_a: {batch['input_a'][0][:30]}, "
    #   f"input_b: {batch['input_b'][0][:30]}, "
    #   f"input_target: {batch['input_target'][0][:30]}, "
    # )
    # print(
    #   f"[GPU {self.global_rank}] "
    #   f"batch_idx: {batch_idx}, "
    #   f"task: {batch['task']}, "
    #   f"input_a: {batch['input_a'][1][:30]}, "
    #   f"input_b: {batch['input_b'][1][:30]}, "
    #   f"input_target: {batch['input_target'][1][:30]}, "
    # )
    # sleep(0.1)
    # return t.tensor(0.0, requires_grad=True)
  
  def validation_step(self, batch):
    # self.validation_step_outputs = {
    #   "contrastive_loss": [],
    #   "reconstruction_loss": [],
    #   "loss": [],
    #   # "decoded_targets": [],
    #   # "decoded_outputs": [],
    # }

    outputs = self.model(
      task=batch['task'],
      input_a=batch['input_a'],
      input_b=batch['input_b'],
      input_target=batch['input_target'],
      # return_decoded=True,
    )
    cl_loss = outputs.contrastive_loss
    gen_loss = outputs.reconstruction_loss
    loss = outputs.loss
    # decoded_outputs = outputs.decoded_outputs
    # decoded_targets = outputs.decoded_targets

    # self.validation_step_outputs["contrastive_loss"].append(cl_loss)
    # self.validation_step_outputs["reconstruction_loss"].append(gen_loss)
    # self.validation_step_outputs["loss"].append(loss)
    # self.validation_step_outputs["decoded_outputs"].extend(decoded_outputs)
    # self.validation_step_outputs["decoded_targets"].extend(decoded_targets)

    batch_size = len(batch['input_a'])
    self.log(
      "val_cl_loss_epoch", 
      cl_loss,
      logger=True,
      on_step=False,
      on_epoch=True,
      sync_dist=True,
      batch_size=batch_size,
    )
    self.log(
      "val_gen_loss_epoch", 
      gen_loss,
      logger=True,
      on_step=False,
      on_epoch=True,
      sync_dist=True,
      batch_size=batch_size,
    )
    self.log(
      "val_loss_epoch", 
      loss,
      logger=True,
      on_step=False,
      on_epoch=True,
      sync_dist=True,
      batch_size=batch_size,
    )

    return loss

  # def on_validation_epoch_end(self):
  #   rank_zero_info("Validation epoch ended, calculating metrics...")

  #   outputs = self.validation_step_outputs
  #   contrastive_loss = t.stack(outputs["contrastive_loss"]).mean()
  #   reconstruction_loss = t.stack(outputs["reconstruction_loss"]).mean()
  #   loss = t.stack(outputs["loss"]).mean()
  #   # decoded_outputs = outputs["decoded_outputs"]
  #   # decoded_targets = outputs["decoded_targets"]

  #   # rouge = ROUGEScore()
  #   # rouge_score = rouge(
  #   #   preds=decoded_outputs, 
  #   #   target=decoded_targets,
  #   # )
  #   # rouge1_fmeasure = rouge_score["rouge1_fmeasure"].to(device=self.device)
  #   # rouge2_fmeasure = rouge_score["rouge2_fmeasure"].to(device=self.device)
  #   # rougeL_fmeasure = rouge_score["rougeL_fmeasure"].to(device=self.device)

  #   self.log(
  #     "val_cl_loss_epoch", 
  #     contrastive_loss,
  #     logger=True,
  #     on_epoch=True,
  #     sync_dist=True,
  #   )
  #   self.log(
  #     "val_gen_loss_epoch", 
  #     reconstruction_loss,
  #     logger=True,
  #     on_epoch=True,
  #     sync_dist=True,
  #   )
  #   self.log(
  #     "val_loss_epoch", 
  #     loss,
  #     logger=True,
  #     on_epoch=True,
  #     sync_dist=True,
  #   )

  #   # self.log(
  #   #   "val_rouge1", 
  #   #   rouge1_fmeasure,
  #   #   logger=True,
  #   #   sync_dist=True,
  #   # )
  #   # self.log(
  #   #   "val_rouge2", 
  #   #   rouge2_fmeasure,
  #   #   logger=True,
  #   #   sync_dist=True,
  #   # )
  #   # self.log(
  #   #   "val_rougeL", 
  #   #   rougeL_fmeasure,
  #   #   logger=True,
  #   #   sync_dist=True,
  #   # )

  #   self.validation_step_outputs.clear()

  def test_step(self, batch):
    # self.test_step_outputs = {
    #   "contrastive_loss": [],
    #   "reconstruction_loss": [],
    #   "loss": [],
    #   # "decoded_targets": [],
    #   # "decoded_outputs": [],
    # }

    outputs = self.model(
      task=batch['task'],
      input_a=batch['input_a'],
      input_b=batch['input_b'],
      input_target=batch['input_target'],
      # return_decoded=True,
    )
    cl_loss = outputs.contrastive_loss
    gen_loss = outputs.reconstruction_loss
    loss = outputs.loss
    # decoded_outputs = outputs.decoded_outputs
    # decoded_targets = outputs.decoded_targets

    # self.test_step_outputs["contrastive_loss"].append(cl_loss)
    # self.test_step_outputs["reconstruction_loss"].append(gen_loss)
    # self.test_step_outputs["loss"].append(loss)
    # self.test_step_outputs["decoded_outputs"].extend(decoded_outputs)
    # self.test_step_outputs["decoded_targets"].extend(decoded_targets)

    batch_size = len(batch['input_a'])
    self.log(
      "test_cl_loss_epoch", 
      cl_loss,
      logger=True,
      on_step=False,
      on_epoch=True,
      sync_dist=True,
      batch_size=batch_size,
    )
    self.log(
      "test_gen_loss_epoch", 
      gen_loss,
      logger=True,
      on_step=False,
      on_epoch=True,
      sync_dist=True,
      batch_size=batch_size,
    )
    self.log(
      "test_loss_epoch", 
      loss,
      logger=True,
      on_step=False,
      on_epoch=True,
      sync_dist=True,
      batch_size=batch_size,
    )

  # def on_test_epoch_end(self):
  #   rank_zero_info("Test epoch ended, calculating metrics...")

  #   outputs = self.test_step_outputs
  #   contrastive_loss = t.stack(outputs["contrastive_loss"]).mean()
  #   reconstruction_loss = t.stack(outputs["reconstruction_loss"]).mean()
  #   loss = t.stack(outputs["loss"]).mean()
  #   # decoded_outputs = outputs["decoded_outputs"]
  #   # decoded_targets = outputs["decoded_targets"]

  #   # rouge = ROUGEScore()
  #   # rouge_score = rouge(
  #   #   preds=decoded_outputs, 
  #   #   target=decoded_targets,
  #   # )
  #   # rouge1_fmeasure = rouge_score["rouge1_fmeasure"].to(device=self.device)
  #   # rouge2_fmeasure = rouge_score["rouge2_fmeasure"].to(device=self.device)
  #   # rougeL_fmeasure = rouge_score["rougeL_fmeasure"].to(device=self.device)

  #   self.log(
  #     "test_cl_loss_epoch", 
  #     contrastive_loss,
  #     logger=True,
  #     sync_dist=True,
  #   )
  #   self.log(
  #     "test_gen_loss_epoch", 
  #     reconstruction_loss,
  #     logger=True,
  #     sync_dist=True,
  #   )
  #   self.log(
  #     "test_loss_epoch", 
  #     loss,
  #     logger=True,
  #     sync_dist=True,
  #   )

  #   # self.log(
  #   #   "test_rouge1", 
  #   #   rouge1_fmeasure,
  #   #   logger=True,
  #   #   sync_dist=True,
  #   # )
  #   # self.log(
  #   #   "test_rouge2", 
  #   #   rouge2_fmeasure,
  #   #   logger=True,
  #   #   sync_dist=True,
  #   # )
  #   # self.log(
  #   #   "test_rougeL",
  #   #   rougeL_fmeasure,
  #   #   logger=True,
  #   #   sync_dist=True,
  #   # )

  #   self.test_step_outputs.clear()  
  
  def configure_optimizers(self):
    param_dict = {
      pn: p 
      for pn, p 
      in self.named_parameters() 
      if p.requires_grad
    }
    decay_params = [
      p for _, p in param_dict.items()
      if p.dim() >= 2
    ]
    nodecay_params = [
      p for _, p in param_dict.items()
      if p.dim() < 2
    ]
    params = [
      {'params': decay_params, 'weight_decay': 0.01},
      {'params': nodecay_params, 'weight_decay': 0.0},
    ]

    optimizer = t.optim.AdamW(
      params=params, 
      lr=self.lr,
    )
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.num_warmup_steps,
      num_training_steps=self.num_training_steps,
    )
    return {
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler": scheduler,
        "interval": "step",
        "frequency": 1,
      }
    }

# datamodule = TaskDataModule(
#   data_dir=args['data_dir'],
#   task=Task.ADD,
#   batch_size=args['batch_size'],
# )

datamodule = MultiTaskDataModule(
  data_dir=args['data_dir'],
  tasks=[Task.ADD, Task.DIFFERENCE],
  batch_size=args['batch_size'],
)

if args['checkpoint_path']:
  model = LitModel.load_from_checkpoint(
    checkpoint_path=args['checkpoint_path'],
    strict=False,
  )
  print(f"Loaded model from checkpoint: {args['checkpoint_path']}")
else:
  model = LitModel(
    lr=args['lr'],
    num_warmup_steps=args['num_warmup_steps'],
    num_training_steps=args['max_steps'],
  )
  print("Initialized new model.")

checkpoint_callback = ModelCheckpoint(
  # dirpath=f"{args['default_root_dir']}/checkpoints",
  filename="{epoch}-{step}-{val_loss_epoch:.4f}",
  monitor="val_loss_epoch",
  mode="min",
  save_top_k=1,
)

trainer = Trainer(
  accelerator=args['accelerator'],
  strategy=(
    DDPStrategy(find_unused_parameters=True)
    if args["strategy"].startswith("ddp")
    else "auto"
  ),
  devices=args['devices'],
  precision="bf16-mixed",
  callbacks=[checkpoint_callback],
  max_steps=args['max_steps'],
  # val_check_interval=args['max_steps'],
  val_check_interval=250,
  enable_checkpointing=True,
  # profiler="simple",
  default_root_dir=args['default_root_dir'],
)
trainer.fit(
  model=model, 
  datamodule=datamodule,
)
trainer.test(
  model=model, 
  datamodule=datamodule,
)

# %%
