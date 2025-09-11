# %% [markdown]
"""
Goals:
- How to compute contrastive learning loss for text embeddings
- How to compute generation loss for text generation models

Notes:
- The contrastive learning loss is penalizing the model for generating embeddings
  that is not similar to the target embeddings and similar to other embeddings.

  However, I am not sure whether this is the right way to penalize the model.
  I am not sure whether the model should be penalized for generating embeddings
  that are similar to other embeddings

  Also, a cosine similarity range values are between -1 and 1, so the loss
  will never be 0, even if the positive embeddings are identical and the
  negative embeddings are very different. The solution is to scale the
  cosine similarity by a temperature value, to exaggerate the differences
  between the correct and incorrect embeddings.

  random guess loss = -ln(1/batch_size)

- The generation loss might be miscalculated.

  The way `fairseq2.generation.BeamSearchSeq2SeqGenerator` works is not straightforward.
  It computes the logits per chunk of 1, and then compute the scores per chunk.
  Then, do beam search, select the token, and repeats until `max_seq_len` is reached.
  Lastly, it sort the hypotheses by their scores. In other words, I am not sure
  whether we can decode it in one forward pass and compute the loss in one forward pass.

  So far, reconstructing "hello world" will result in loss of 4.53, even though
  the model predicts the correct token. The solution is to scale the loss
  by 0.01, so that it does not dominate the contrastive learning loss.

  random guess loss = -ln(1/vocab_size)
"""

# %%

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.data import Collater
from fairseq2.models.sequence import SequenceBatch
from fairseq2.generation import BeamSearchSeq2SeqGenerator

import torch as t
from torch import nn
import torch.nn.functional as F

from torchmetrics.text import ROUGEScore

# %% TODO: remove this later

text2vec_model = TextToEmbeddingModelPipeline(
    tokenizer="text_sonar_basic_encoder",
    encoder="text_sonar_basic_encoder",
)

vec2text_model = EmbeddingToTextModelPipeline(
    tokenizer="text_sonar_basic_decoder",
    decoder="text_sonar_basic_decoder",
)

# %%

print("Reproducing text2vec_model.predict() behavior...")
max_seq_len = 512
tokenizer_encoder = text2vec_model.tokenizer.create_encoder(lang="eng_Latn")
sentences = [
    "hello world",
    "hello world",
    "hello world",
    # "hello",
    # "hello world. my name is jeff",
]
seqs = [tokenizer_encoder(sentence) for sentence in sentences]
seqs = [seq[:max_seq_len] if max_seq_len is not None else seq for seq in seqs]
collater = Collater(pad_value=text2vec_model.tokenizer.vocab_info.pad_idx)
batch = collater(seqs)
tokens, padding_mask = get_seqs_and_padding_mask(data=batch)
sequence_batch = SequenceBatch(tokens, padding_mask)
embeddings = text2vec_model.model(sequence_batch).sentence_embeddings

print(f"embeddings:\n{embeddings}")

# %%

print("Reproducing vec2text_model.predict() behavior...")

target_text_encoder = vec2text_model.tokenizer.create_encoder(
    task="translation",
    lang="eng_Latn",
    mode="target",
)
target_prefix_seqs = target_text_encoder.prefix_indices
text_decoder = vec2text_model.tokenizer.create_decoder()

batch_size = len(embeddings)

generator = BeamSearchSeq2SeqGenerator(
    model=vec2text_model.model,
    max_seq_len=max_seq_len,
)

generator_output = generator(
    embeddings, None, target_prefix_seqs.expand(batch_size, -1), None
)

texts: list[str] = []
for idx, hypotheses in enumerate(generator_output.hypotheses):
    texts.append(text_decoder(hypotheses[0].seq))
print("texts:", texts)

# %%

print("Get logits")

# One by one
input_to_decoder = []
logits = []
for idx, hypotheses in enumerate(generator_output.hypotheses):
    # seq = t.cat(
    #   [
    #     t.tensor([text2vec_model.tokenizer.vocab_info.eos_idx]),
    #     hypotheses[0].seq[:-1],
    #   ],
    # )
    seq = t.cat(
        [
            # t.tensor(
            #     [
            #         text2vec_model.tokenizer.vocab_info.eos_idx,
            # 		256075,
            #     ]
            # ),
            # seqs[idx][:-1],
            target_prefix_seqs,
            seqs[idx][1:-1],  # Remove first and last token (language token and EOS)
        ],
    )
    input_to_decoder.append(seq)
    decoder_output, decoder_padding_mask = vec2text_model.model.decode(
        seqs=seq.unsqueeze(0),
        padding_mask=None,
        encoder_output=embeddings[idx].unsqueeze(0).unsqueeze(0),
        encoder_padding_mask=None,
    )
    model_output = vec2text_model.model.project(decoder_output, decoder_padding_mask)
    logits.append(model_output.logits.squeeze(0))

labels = [
    seq
    # seq[1:], # Remove first token (language token)
    for seq in seqs
    # hypotheses[0].seq
    # for hypotheses in generator_output.hypotheses
]

max_len_logits = max(l.size(0) for l in logits)
max_len_labels = max(l.size(0) for l in labels)
max_len = max(max_len_logits, max_len_labels)

logits_padded = []
for logit in logits:
    if logit.size(0) < max_len:
        pad_amt = max_len - logit.size(0)
        logit = F.pad(
            logit, (0, 0, 0, pad_amt), value=vec2text_model.tokenizer.vocab_info.pad_idx
        )
    logits_padded.append(logit)
logits_padded = t.stack(logits_padded, dim=0)

labels_padded = []
for label in labels:
    if label.size(0) < max_len:
        pad_amt = max_len - label.size(0)
        label = F.pad(
            label, (0, pad_amt), value=vec2text_model.tokenizer.vocab_info.pad_idx
        )
    labels_padded.append(label)
labels_padded = t.stack(labels_padded, dim=0)

loss_fct = nn.CrossEntropyLoss(
    ignore_index=vec2text_model.tokenizer.vocab_info.pad_idx,
)
gen_loss = loss_fct(
    logits_padded.reshape(
        -1, logits_padded.size(-1)
    ),  # (batch_size * seq_len, vocab_size)
    labels_padded.reshape(-1),  # (batch_size * seq_len)
)
print("input_to_decoder:")
for seq in input_to_decoder:
    print(seq)
print(f"labels_padded:\n{labels_padded}")
print("vocab_size:", logits_padded.size(-1))
print(
    "random guess loss:",
    loss_fct(
        t.zeros_like(logits_padded).reshape(-1, logits_padded.size(-1)),
        labels_padded.reshape(-1),
    ).item(),
)
print("gen_loss:", gen_loss.item())

greedy_token = logits_padded.argmax(dim=-1)
print("greedy_token:", greedy_token)
for seq in greedy_token:
    print(text_decoder(seq))

# %%

# Batch
input_to_decoder = [
    t.cat(
        [
            # t.tensor(
            #     [
            #         text2vec_model.tokenizer.vocab_info.eos_idx,
            #     ]
            # ),
            # seq[:-1],  # Remove last token (EOS)
            target_prefix_seqs,
            seq[1:-1],  # Remove first and last token (language token and EOS)
        ]
    )
    for seq in seqs
]
# input_to_decoder = [
#     t.tensor([3, 256047, 133863, 15697]),
#     t.tensor([3, 256075, 133863, 15697]),
#     t.tensor([3, 256047, 133863, 15697]),
# ]
input_data = collater(input_to_decoder)
input_tensor, input_padding_mask = get_seqs_and_padding_mask(data=input_data)

decoder_output, decoder_padding_mask = vec2text_model.model.decode(
    seqs=input_tensor,
    padding_mask=input_padding_mask,
    encoder_output=embeddings.unsqueeze(1),
    encoder_padding_mask=None,
)

model_output = vec2text_model.model.project(
    decoder_output=decoder_output,
    decoder_padding_mask=decoder_padding_mask,
)

logits = model_output.logits

labels = seqs
# labels = [
#   # seq[1:]  # Remove first token (language token)
#   seq
#   for seq in seqs
#   # hypotheses[0].seq
#   # for hypotheses in generator_output.hypotheses
# ]

max_len_logits = logits.size(1)
max_len_labels = max(l.size(0) for l in labels)
max_len = max(max_len_logits, max_len_labels)

logits_padded = logits
if logits.size(1) < max_len:
    pad_amt = max_len - logits.size(1)
    logits_padded = F.pad(
        logits, (0, 0, 0, pad_amt), value=vec2text_model.tokenizer.vocab_info.pad_idx
    )

labels_padded = []
for label in labels:
    if label.size(0) < max_len:
        pad_amt = max_len - label.size(0)
        label = F.pad(
            label, (0, pad_amt), value=vec2text_model.tokenizer.vocab_info.pad_idx
        )
    labels_padded.append(label)
labels_padded = t.stack(labels_padded, dim=0)

loss_fct = nn.CrossEntropyLoss(
    ignore_index=vec2text_model.tokenizer.vocab_info.pad_idx,
)
gen_loss = loss_fct(
    logits_padded.reshape(
        -1, logits_padded.size(-1)
    ),  # (batch_size * seq_len, vocab_size)
    labels_padded.reshape(-1),  # (batch_size * seq_len)
)
print("input_to_decoder:")
for seq in input_to_decoder:
    print(seq)
print(f"labels_padded:\n{labels_padded}")
print("vocab_size:", logits_padded.size(-1))
print(
    "random guess loss:",
    loss_fct(
        t.zeros_like(logits_padded).reshape(-1, logits_padded.size(-1)),
        labels_padded.reshape(-1),
    ).item(),
)
print("gen_loss:", gen_loss.item())

greedy_token = logits_padded.argmax(dim=-1)
print("greedy_token:", greedy_token)
for seq in greedy_token:
    print(text_decoder(seq))

# %%

print("Decoding with greedy search...")
batch_size = embeddings.shape[0]
input_seqs = t.tensor(
    [
        [
            text2vec_model.tokenizer.vocab_info.eos_idx,
            256047,  # language token
        ]
        for _ in range(batch_size)
    ],
)

with t.no_grad():
    for i in range(max_seq_len):
        decoder_output, decoder_padding_mask = vec2text_model.model.decode(
            seqs=input_seqs,
            padding_mask=None,
            encoder_output=embeddings.unsqueeze(1),
            encoder_padding_mask=None,
        )
        greedy_token = (
            vec2text_model.model.project(
                decoder_output=decoder_output,
                decoder_padding_mask=decoder_padding_mask,
            )
            .logits[:, -1, :]
            .argmax(dim=-1, keepdim=True)
        )
        input_seqs = t.cat([input_seqs, greedy_token], dim=1)
        if greedy_token[-1] == text2vec_model.tokenizer.vocab_info.eos_idx:
            break

print(f"seqs:\n{input_seqs}")

for seq in input_seqs:
    print(text_decoder(seq))

# %%

input_a = [
    "hello world",
    "lorem ipsum",
]
input_b = [
    "world",
    "ipsum",
]
input_target = [
    "hello",
    "ipsum",
]
# input_target = None

batch_size = len(input_a)

input = input_a + input_b + input_target

embeddings = text2vec_model.predict(
    input=input,
    source_lang="eng_Latn",
)
print("embeddings shape:", embeddings.shape)

embeddings = embeddings.clone()
# norm1 = nn.RMSNorm(1024)
# embeddings = norm1(embeddings)

embeddings_a = embeddings[0:batch_size]
embeddings_b = embeddings[batch_size : 2 * batch_size]
embeddings_target = embeddings[2 * batch_size :]

print("embeddings_a shape:", embeddings_a.shape)
print("embeddings_b shape:", embeddings_b.shape)

embeddings_a_b = t.cat([embeddings_a, embeddings_b], dim=1)
print("embeddings_a_b shape:", embeddings_a_b.shape)

linear = nn.Linear(2 * 1024, 1024)
embeddings_sim = linear(embeddings_a_b)  # shape: (batch_size, 1024)
# norm2 = nn.RMSNorm(1024)
# embeddings_sim = norm2(embeddings_sim)
print("embeddings_sim shape:", embeddings_sim.shape)

embeddings_pos = embeddings_target

texts = vec2text_model.predict(
    inputs=embeddings_sim,
    target_lang="eng_Latn",
    max_seq_len=10,
)
print("texts:", texts)

if input_target is not None:
    sim_fct = nn.CosineSimilarity(dim=-1)
    temperature = 0.05
    cos_sim = (
        sim_fct(
            embeddings_sim.unsqueeze(dim=1),  # shape: (batch_size, 1, n_embd)
            embeddings_pos.unsqueeze(dim=0),  # shape: (1, batch_size, n_embd)
        )
        / temperature
    )
    # cos_sim = t.tensor([
    #   [1.0, -1.0],
    #   [-1.0, 1.0],
    # ])
    print(f"cos_sim:\n{cos_sim}")
    cl_labels = t.arange(batch_size, device=embeddings_sim.device)
    loss_fct = nn.CrossEntropyLoss()
    cl_loss = loss_fct(cos_sim, cl_labels)
    print("cl_loss:", cl_loss.item())

# %%

input = [
    "hello world",
    "world is a beautiful place",
    "my name is jeff",
    "name is a way to identify a person",
    "is that true",
    "jeff is the CEO of the company",
    "and he is a good person",
    "you can trust him",
]

embeddings = text2vec_model.predict(
    input=input,
    source_lang="eng_Latn",
)

sim_fct = nn.CosineSimilarity(dim=-1)
temperature = 0.05
cos_sim = (
    sim_fct(
        embeddings.unsqueeze(dim=1),  # shape: (batch_size, 1, n_embd)
        embeddings.unsqueeze(dim=0),  # shape: (1, batch_size, n_embd)
    )
    / temperature
)
print(f"cos_sim:\n{cos_sim}")
loss_fct = nn.CrossEntropyLoss()
cl_labels = t.arange(len(input), device=embeddings.device)
cl_loss = loss_fct(cos_sim, cl_labels)
print("cl_loss:", cl_loss.item())

# %%

preds = [
    "hello world",
    "hello",
]
target = [
    "hello world",
    "hello ?",
]

print(f"preds: {preds}")
print(f"target: {target}")

rouge = ROUGEScore(accumulate="avg")
rouge_score = rouge(preds, target)
rogue1_fmeasure = rouge_score["rouge1_fmeasure"]
rogue2_fmeasure = rouge_score["rouge2_fmeasure"]
rougeL_fmeasure = rouge_score["rougeL_fmeasure"]
rougeLsum_fmeasure = rouge_score["rougeLsum_fmeasure"]

print(f"rouge1_fmeasure: {rogue1_fmeasure:.4f}")
# unigrams, rogue1_fmeasure=1.0 because the target unigrams are 3,
# and it matched 3 times in preds
print(f"rouge2_fmeasure: {rogue2_fmeasure:.4f}")
# bigrams
print(f"rougeL_fmeasure: {rougeL_fmeasure:.4f}")
# longest common subsequence
print(f"rougeLsum_fmeasure: {rougeLsum_fmeasure:.4f}")
# LCS over concatenated text

# %%
