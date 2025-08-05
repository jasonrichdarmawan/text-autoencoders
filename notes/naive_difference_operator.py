# %% [markdown]
"""
1. `'Capital of' Vector = Embedding Relation - Embedding Control` does not work
1. `Inversion Vector = Embedding Passive - Embedding Active` does not work
2. `Relational Vector = Embedding Sentence - Embedding Cause - Embedding Effect` does not work
3. `Politeness Vector = Embedding Polite - Embedding Impolite` works for seen sentences, but not for unseen sentences
"""

# %%

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
  print(f"Adding project root to sys.path: {project_root}")  
  sys.path.append(project_root)

# %%

from argparse import ArgumentParser
from typing import TypedDict

import torch as t
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

from sklearn.decomposition import PCA
import plotly.graph_objects as go

# %%

class Args(TypedDict):
  workspace_path: str
  device: str

def parse_args():
  parser = ArgumentParser(description="Run interpretable relational experiments.")

  parser.add_argument(
    "--workspace_path",
    type=str,
    help="Path to the workspace directory where the experiments will be run.",
  )

  parser.add_argument(
    "--device",
    type=str,
    help="Device to run the model on (e.g., 'cuda:0' for GPU, 'cpu' for CPU).",
  )

  ns = parser.parse_args()
  return Args(**vars(ns))

if False:
  WORKSPACE_PATH = "/workspace/ALGOVERSE/UJR/jason"
  sys.argv = [
    'main.py',
    '--workspace_path', WORKSPACE_PATH,
    '--device', 'cuda:1',
  ]

args = parse_args()

print("Parsed arguments:")
for key, value in args.items():
  print(f"{key}: {value}")

# %%

print("Loading text-to-embedding and embedding-to-text models...")
text2vec_model = TextToEmbeddingModelPipeline(
  encoder="text_sonar_basic_encoder",
  tokenizer="text_sonar_basic_encoder",
  device=t.device(args["device"]),
)

vec2text_model = EmbeddingToTextModelPipeline(
  decoder="text_sonar_basic_decoder",
  tokenizer="text_sonar_basic_decoder",
  device=t.device(args["device"]),
)

# %% [markdown]
""" 
$$\vec{v}_{\text{capital\_of}} = S_{\text{relation}} - S_{\text{control}}$$

Conclusion: It doesn't work
"""

sentences = [
  ("Paris is the capital of France.", # S_relation
   "Paris and France are both in Europe.", # S_control
   "Berlin is a large city in Germany."), # S_target
  ("Berlin is the capital of Germany.",
   "Berlin and Germany are both in Europe.",
   "Paris is a large city in France."),
  ("Madrid is the capital of Spain.",
   "Madrid and Spain are both in Europe.",
   "Lisbon is a large city in Portugal."),
  ("Rome is the capital of Italy.",
   "Rome and Italy are both in Europe.",
   "Athens is a large city in Greece."),
  ("London is the capital of the United Kingdom.",
   "London and the United Kingdom are both in Europe.",
   "Dublin is a large city in Ireland."),
]

embeddings = text2vec_model.predict(
  [sentence for tup in sentences for sentence in tup],
  source_lang="eng_Latn",
)

v_capital_of = t.stack([
  embeddings[i * 3 + 0] - embeddings[i * 3 + 1]
  for i in range(len(sentences))
]).mean(dim=0)

intervened_embeddings = t.stack([
  embeddings[i * 3 + 2] + v_capital_of
  for i in range(len(sentences))
])

decoded_v_capital_of = vec2text_model.predict(
  [v_capital_of],
  target_lang="eng_Latn",
)

decoded = vec2text_model.predict(
  intervened_embeddings,
  target_lang="eng_Latn",
)

print(f"S_decoded_v_capital_of: {decoded_v_capital_of[0]}")

for i, tup in enumerate(sentences):
  print(f"S_relation_{i}: {tup[0]}")
  print(f"S_control_{i}: {tup[1]}")
  print(f"S_target_{i}: {tup[2]}")
  print(f"S_decoded_{i}: {decoded[i]}")
  print()

# %% [markdown]
"""
$$\vec{v}_\text{inversion} = S_{\text{passive}} - S_{\text{active}}$$

Conclusion: It does not work
"""

sentences = [
  ("The heavy rain caused flooding", 
   "The flooding was caused by the heavy rain"),
  ("The dog chased the cat.", 
   "The cat was chased by the dog."),
  ("The teacher praised the student.", 
   "The student was praised by the teacher."),
  ("The chef cooked a delicious meal.", 
   "A delicious meal was cooked by the chef."),
  ("The artist painted a beautiful picture.", 
   "A beautiful picture was painted by the artist."),
  ("The scientist discovered a new species.", 
   "A new species was discovered by the scientist."),
]

embeddings = text2vec_model.predict(
  [sentence for tup in sentences for sentence in tup], 
  source_lang="eng_Latn",
)

v_inversion = t.stack([
  embeddings[i * 2 + 1] - embeddings[i * 2 + 0]
  for i in range(len(sentences))
]).mean(dim=0)

intervened_embeddings = t.stack([
  embeddings[i * 2 + 0] + v_inversion
  for i in range(len(sentences))
])

decoded_v_inversion = vec2text_model.predict(
  [v_inversion],
  target_lang="eng_Latn",
)

decoded = vec2text_model.predict(
  intervened_embeddings,
  target_lang="eng_Latn",
)

print(f"S_decoded_v_inversion: {decoded_v_inversion[0]}")
for i, tup in enumerate(sentences):
  print(f"S_active_{i}: {tup[0]}")
  print(f"S_passive_{i}: {tup[1]}")
  print(f"S_decoded_{i}: {decoded[i]}")
  print()

# %% [markdown]
"""
$$\vec{v}_{\text{relational}_i} = z_{\text{sentence}_i} - z_{\text{cause}_i} - z_{\text{effect}_i}$$

$$\text{alignment\_score} = \text{cosine\_similarity}(\vec{v}_\text{claimed\_relation},\vec{v}_\text{relational})$$

Conclusion: It does not work
"""

sentences = [
  ("Heavy rain caused flooding", # S_sentence
    "Heavy rain _ _", # S_cause
    "_ _ _ flooding"), # S_effect
  ("Heavy rain resulted in flooding",
   "Heavy rain _ _ _",
   "_ _ _ flooding"),
  ("Heavy rain led to flooding",
   "Heavy rain _ _ _",
   "_ _ _ _ flooding"),
  ("Smoking increases the risk of cancer",
    "Smoking _ _ _ _ _",
    "_ the risk of cancer"),
  ("Smoking is a major cause of cancer",
   "Smoking _ _ _ _ _ _",
   "_ _ _ _ _ _ cancer"),
  ("Smoking contributes to the development of cancer",
   "Smoking _ _ _ _ _ _",
   "_ _ _ the development of cancer"),
]

# sentences = [
#   tuple(s.replace("_", "").strip() for s in tup)
#   for tup in sentences
# ]

claimed_sentences = [
  ("Vaccines caused a significant reduction in disease incidence", # S_sentence
   "Vaccines _ _ _ _ _ _ _", # S_cause
   "_ _ a significant reduction in disease incidence", # S_effect
   "OK"),
  ("Vaccines cause autism",
   "Vaccines _ _",
   "_ _ autism",
   "Misleading claim"),
  ("Vaccines lead to autism",
   "Vaccines _ _ _",
   "_ _ _ autism",
   "Misleading claim"),
  ("Vaccines save lives",
   "Vaccines _ _",
   "_ _ lives",
   "OK"),
  ("Vaccines prevent disease",
   "Vaccines _ _",
   "_ _ disease",
   "OK"),
  ("Vaccines protect against infections",
   "Vaccines _ _ _",
   "_ _ against infections",
   "OK"),
  ("Vacciness cause autism. This is a fact",
   "Vaccines _ _. _ _ _ _",
   "_ _ autism. _ _ _ _ _",
   "Misleading claim"),
  ("Vaccines are safe and effective",
   "Vaccines _ _ _ _",
   "_ _ _ _ effective",
   "OK"),
  ("Global warming is caused by solar activity",
   "Global warming _ _ _ _ _",
   "_ _ _ _ _ solar activity",
   "Misleading claim"),
]

# claimed_sentences = [
#   tuple(s.replace("_", "").strip() for s in tup)
#   for tup in claimed_sentences
# ]

embeddings = text2vec_model.predict(
  [sentence for tup in sentences for sentence in tup],
  source_lang="eng_Latn",
)

claimed_embeddings = text2vec_model.predict(
  [sentence for tup in claimed_sentences for sentence in tup],
  source_lang="eng_Latn",
)

v_relational = t.stack([
  embeddings[i * 3 + 0] - embeddings[i * 3 + 1] - embeddings[i * 3 + 2]
  for i in range(len(sentences))
])

v_claimed_relational = t.stack([
  claimed_embeddings[i * 3 + 0] - claimed_embeddings[i * 3 + 1] - claimed_embeddings[i * 3 + 2]
  for i in range(len(claimed_sentences))
])

alignment_score = t.cosine_similarity(
  v_relational.mean(dim=0),
  v_claimed_relational,
)

for i, tup in enumerate(claimed_sentences):
  print(f"S_claimed_sentence_{i}: {tup[0]}")
  print(f"alignment_score_{i}: {alignment_score[i].item()}")
  print(f"norm_{i}: {v_claimed_relational[i].norm().item():.4f}")

pca = PCA(n_components=2)
v_relational_2d = pca.fit_transform(
  v_relational.cpu().numpy(),
)

print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

v_claimed_relational_2d = pca.transform(
  v_claimed_relational.cpu().numpy(),
)

fig = go.Figure()
fig.add_trace(go.Scatter(
  x=v_relational_2d[:, 0], 
  y=v_relational_2d[:, 1], 
  mode='markers', 
  name="Relational",
  marker=dict(color="blue", opacity=0.5),
  text=[sentences[i][0] for i in range(len(v_relational_2d))],
))
claimed_labels = [tup[3] for tup in claimed_sentences]
ok_mask = [label == "OK" for label in claimed_labels]
misleading_mask = [label == "Misleading claim" for label in claimed_labels]
fig.add_trace(go.Scatter(
  x=v_claimed_relational_2d[ok_mask, 0], 
  y=v_claimed_relational_2d[ok_mask, 1], 
  mode='markers',
  name='Claimed OK',
  marker=dict(color='green', opacity=0.7),
  text=[claimed_sentences[i][0] for i, ok in enumerate(ok_mask) if ok]
))
fig.add_trace(go.Scatter(
  x=v_claimed_relational_2d[misleading_mask, 0], 
  y=v_claimed_relational_2d[misleading_mask, 1], 
  mode='markers',
  name='Claimed Misleading',
  marker=dict(color='red', opacity=0.7),
  text=[claimed_sentences[i][0] for i, ms in enumerate(misleading_mask) if ms]
))

fig.update_layout(
  title="PCA of Relational and Claimed Relational Vectors",
  xaxis_title="PCA Component 1",
  yaxis_title="PCA Component 2",
  legend=dict(title="Legend"),
)

# %% [markdown]
"""
$$\vec{v}_\text{politeness} = S_{\text{polite}} - S_{\text{impolite}}$$

Conclusion: It works for seen sentences, but not for unseen sentences
"""

sentences = [
  ("Could you help me?", "Help me!"),
  ("Can you assist me?", "Assist me!"),
  ("Would you be able to help me?", "Help me!"),
  ("I need your assistance.", "Need your assistance!"),
  ("Please help me with this task.", "Help me with this task!"),
]

unseen_sentences = [
  ("Look at this!", "Impolite"),
  ("Pay attention to this!", "Impolite"),
  ("Just do it!", "Impolite"),
  ("I need you to do this!", "Impolite"),
  ("Can you do this for me?", "Polite"),
  ("I would appreciate your help with this.", "Polite"),
]

embeddings = text2vec_model.predict(
  [sentence for tup in sentences for sentence in tup],
  source_lang="eng_Latn",
)

unseen_embeddings = text2vec_model.predict(
  [sentence for tup in unseen_sentences for sentence in tup],
  source_lang="eng_Latn",
)

v_politeness = t.stack([
  embeddings[i * 2 + 0] - embeddings[i * 2 + 1]
  for i in range(len(sentences))
]).mean(dim=0)

intervened_embeddings = t.stack([
  embeddings[i * 2 + 1] + v_politeness
  for i in range(len(sentences))
])

intervened_unseen_embeddings = t.stack([
  unseen_embeddings[i * 2 + 1] + v_politeness
  for i in range(len(unseen_sentences))
])

decoded_v_politeness = vec2text_model.predict(
  [v_politeness],
  target_lang="eng_Latn",
)
decoded = vec2text_model.predict(
  intervened_embeddings,
  target_lang="eng_Latn",
)
decoded_unseen = vec2text_model.predict(
  intervened_unseen_embeddings,
  target_lang="eng_Latn",
)

print(f"S_decoded_v_politeness: {decoded_v_politeness[0]}")
for i, tup in enumerate(sentences):
  print(f"S_polite_{i}: {tup[0]}")
  print(f"S_impolite_{i}: {tup[1]}")
  print(f"S_decoded_{i}: {decoded[i]}")
  print()

for i, tup in enumerate(unseen_sentences):
  print(f"S_unseen_{i}: {tup[0]}")
  print(f"S_unseen_label_{i}: {tup[1]}")
  print(f"S_decoded_unseen_{i}: {decoded_unseen[i]}")
  print()

# %%
