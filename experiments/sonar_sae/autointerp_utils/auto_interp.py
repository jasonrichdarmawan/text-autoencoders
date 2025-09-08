from sae_utils import (
    DataModule,
    LitModel,
)

from autointerp_utils import (
    AutoInterpConfig,
    Example,
)

from typing import (
    Any,
    Literal,
    TypeAlias,
)

from jaxtyping import Float

from tqdm.auto import tqdm

import torch as t
from torch import Tensor

import random

from openai import AsyncOpenAI

from tabulate import tabulate

import asyncio

Messages: TypeAlias = list[dict[Literal["role", "content"], str]]


class AutoInterp:
    """
    This is astart-to-end class for generating explanations
    and optionally scores. It's easiest to implement it as a single
    class for the time being because there's data we'll need
    to fetch that'll be used in both the generation and
    scoring phases.

    Partial code from https://arena-chapter1-transformer-interp.streamlit.app/[1.3.2]_Interpretability_with_SAEs#run
    """

    def __init__(
        self,
        cfg: AutoInterpConfig,
        data_module: DataModule,
        model: LitModel,
        api_key: str,
    ):
        self.cfg = cfg
        self.data_module = data_module
        self.model = model
        self.api_key = api_key

    async def run(
        self, debug: bool = False, max_concurrent: int = 10
    ) -> dict[int, dict[str, Any]]:
        """
        Runs both generation & scoring phases,
        and returns the results in a dictionary
        """
        generation_examples, scoring_examples = self.gather_data()
        results = {}

        semaphore = asyncio.Semaphore(value=max_concurrent)
        # Rate limit to ~10 requests at a time

        async def process_latent(latent: int):
            async with semaphore:
                gen_prompts = self.get_generation_prompts(
                    generation_examples=generation_examples[latent]
                )
                explanation_raw = (
                    await self.get_response(
                        messages=gen_prompts,
                        max_tokens=self.cfg.max_tokens,
                        debug=debug and (latent == self.cfg.latents[0]),
                    )
                )[0]
                explanation = self.parse_explanation(explanation=explanation_raw)
                result = {
                    "explanation": explanation,
                    "top activating examples": [
                        ex.to_str() for ex in generation_examples[latent]
                    ],
                }

                if self.cfg.scoring:
                    scoring_prompts = self.get_scoring_prompts(
                        explanation=explanation,
                        scoring_examples=scoring_examples[latent],
                    )
                    predictions = (
                        await self.get_response(
                            messages=scoring_prompts,
                            max_tokens=self.cfg.max_tokens,
                            debug=debug and (latent == self.cfg.latents[0]),
                        )
                    )[0]
                    predictions_parsed = self.parse_predictions(predictions=predictions)
                    accuracy = self.score_accuracy(
                        predictions=predictions_parsed,
                        scoring_examples=scoring_examples[latent],
                    )
                    f1 = self.score_f1(
                        predictions=predictions_parsed,
                        scoring_examples=scoring_examples[latent],
                    )
                    result |= {
                        "scoring prompts": scoring_prompts[1]["content"],
                        "predictions": predictions_parsed,
                        "correct seqs": [
                            i
                            for i, ex in enumerate(scoring_examples[latent], start=1)
                            if ex.is_active
                        ],
                        "f1": f1["f1"],
                        "accuracy": accuracy,
                        "precision": f1["precision"],
                        "recall": f1["recall"],
                    }
                return latent, result

        tasks = [process_latent(latent) for latent in self.cfg.latents]
        with tqdm(total=len(tasks), desc="Querying OpenAI api") as pbar:
            for f in asyncio.as_completed(tasks):
                latent, result = await f
                results[latent] = result
                pbar.update(1)

        return results

    def gather_data(self) -> tuple[dict[int, list[Example]], dict[int, list[Example]]]:
        """
        Stores top acts / random seqs data,
        which is used for generation & scoring
        respectively
        """
        batch_size = self.data_module.batch_size
        total_batches = self.cfg.total_tokens // batch_size

        # Get indices we'll take our random examples
        # from, over all batches (and over all latents)
        all_rand_indices_shape = (self.cfg.n_random_ex_for_scoring, self.cfg.n_latents)
        all_rand_indices = t.stack(
            [
                t.randint(
                    low=0, high=total_batches, size=all_rand_indices_shape
                ),  # which batch
                t.randint(
                    low=0, high=batch_size, size=all_rand_indices_shape
                ),  # which sequence in the batch
            ],
            dim=-1,
        )  # shape [n_random_ex_for_scoring, n_latents, 2]

        # Dictionary to store data for each latent
        latent_data = {
            latent: {
                "rand_texts": [],
                "rand_acts": t.empty(0, dtype=t.float32, device=self.model.sae.device),
                "top_texts": [],
                "top_values": t.empty(0, dtype=t.float32, device=self.model.sae.device),
            }
            for latent in self.cfg.latents
        }

        iterable = iter(self.data_module.train_dataloader())
        for batch in tqdm(
            iterable=range(total_batches), desc="Collecting activations data"
        ):
            data = next(iterable)[0]
            text = (
                data["nllb_200_6m_sample_embedding"]["text1"]
                + data["nllb_primary_datasets_embedding"]["text1"]
            )
            embedding = t.concat(
                (
                    data["nllb_200_6m_sample_embedding"]["embedding1"].to(
                        self.model.sae.device
                    ),
                    data["nllb_primary_datasets_embedding"]["embedding1"].to(
                        self.model.sae.device
                    ),
                ),
                dim=0,
            )
            with t.no_grad():
                with t.autocast(device_type="cuda", dtype=t.bfloat16):
                    acts = self.model.sae.encode(x=embedding)[:, self.cfg.latents]

                for i, latent in enumerate(self.cfg.latents):
                    # Get top activations from this batch,
                    # and filter down to the data we'll actually
                    # include
                    top_indices = get_k_largest_indices(
                        texts=text,
                        acts=acts[:, i],
                        k=self.cfg.n_top_ex_for_generation,
                        no_overlap=self.cfg.no_overlap,
                    )
                    top_texts = [text[idx] for idx in top_indices]
                    top_values = acts[top_indices, i]
                    latent_data[latent]["top_texts"] += top_texts
                    latent_data[latent]["top_values"] = t.cat(
                        (latent_data[latent]["top_values"], top_values), dim=0
                    )

                    # Get random examples (our `all_rand_indices` tensor
                    # tells us which random batch to take)
                    rand_indices = all_rand_indices[
                        all_rand_indices[:, i, 0] == batch, i, 1
                    ]
                    latent_data[latent]["rand_texts"] += [
                        text[idx] for idx in rand_indices.tolist()
                    ]
                    latent_data[latent]["rand_acts"] = t.cat(
                        (latent_data[latent]["rand_acts"], acts[rand_indices, i]), dim=0
                    )

        # Dicts to store all generation
        # & scoring examples for each latent
        generation_examples = {}
        scoring_examples = {}

        for i, latent in enumerate(self.cfg.latents):
            top_texts = latent_data[latent]["top_texts"]
            top_values = latent_data[latent]["top_values"]
            act_threshold = self.cfg.act_threshold_frac * top_values.max().item()
            # From our tensor of `n_top_examples * n_batches`
            # top examples, get only the top
            # `n_top_examples` of them
            topk = get_k_largest_indices(
                texts=top_texts,
                acts=top_values,
                k=self.cfg.n_top_ex,
                no_overlap=self.cfg.no_overlap,
                act_threshold=act_threshold,
            )
            rand_split_indices = t.randperm(self.cfg.n_top_ex)

            # generation_examples[latent] = random sample
            # of some of the top activating sequences
            generation_examples[latent] = [
                Example(
                    text=top_texts[topk[j]],
                    act=top_values[topk[j]].item(),
                    act_threshold=act_threshold,
                )
                for j in sorted(rand_split_indices[: self.cfg.n_top_ex_for_generation])
            ]

            # scoring_examples[latent] = random mix of
            # the sampled top activating texts & random
            # examples (with the top activating texts chosen
            # have zero overlap with those used in generation
            # examples)
            scoring_examples[latent] = random.sample(
                [
                    Example(
                        text=top_texts[topk[j]],
                        act=top_values[topk[j]].item(),
                        act_threshold=act_threshold,
                    )
                    for j in rand_split_indices[self.cfg.n_top_ex_for_generation :]
                ]
                + [
                    Example(
                        text=random_text,
                        act=random_act.item(),
                        act_threshold=act_threshold,
                    )
                    for random_text, random_act in zip(
                        latent_data[latent]["rand_texts"],
                        latent_data[latent]["rand_acts"],
                    )
                ],
                k=self.cfg.n_ex_for_scoring,
            )

        return generation_examples, scoring_examples

    def get_generation_prompts(self, generation_examples: list[Example]) -> Messages:
        assert len(generation_examples) > 0, "No generation exmaples found"

        examples_as_str = "\n".join(
            [f"{i + 1}. {ex.to_str()}" for i, ex in enumerate(generation_examples)]
        )

        SYSTEM_PROMPT = """You are an expert in linguistics and semantic analysis. Your task is to interpret features from a neural network that encodes entire sentences into a single representation. Each feature activates on sentences that share a common abstract property.

I will provide you with a list of sentences that strongly activate a specific feature. Your goal is to find the underlying commonality and describe it in a single, concise sentence.

Do NOT focus on simple keyword matching. Instead, consider properties like:
- **Semantic Role**: Are these all sentences where someone is giving a command? Expressing doubt? Asking for information?
- **Abstract Concepts**: Do these sentences relate to concepts like 'future plans', 'past regrets', 'expressions of gratitude', or 'scientific principles'?
- **Syntactic Structure**: Are they all questions? Are they complex sentences with multiple clauses?
- **Tone and Sentiment**: Do they share a sarcastic tone? Are they all optimistic?

Summarize what this feature represents. Your explanation should be short, under 20 words, and start with "This feature activates on sentences that...". Omit final punctuation.
"""
        USER_PROMPT = f"""Here are the sentences that activate this feature:\n\n{examples_as_str}"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

    def get_scoring_prompts(
        self, explanation: str, scoring_examples: list[Example]
    ) -> Messages:
        assert len(scoring_examples) > 0, "No scoring examples found"

        examples_as_str = "\n".join(
            [f"{i + 1}. {ex.to_str()}" for i, ex in enumerate(scoring_examples)]
        )

        SYSTEM_PROMPT = f"""You are an expert AI evaluating a hypothesis about a neural network feature. You will be given a description of what a feature supposedly represents at the sentence level. You will then see {len(scoring_examples)} example sentences.

Your task is to identify which of these sentences fit the description.

Return a comma-separated list of the numbers corresponding to the sentences that match the description. If no sentences match, respond with the word "None".

Your response must ONLY contain the comma-separated numbers or the word "None". Do not add any other words, explanations, or punctuation. This is critical.
"""

        # We create a fake user/assistant interaction to show the model exactly what to do.
        ONE_SHOT_USER = """The description is: this feature activates on sentences that are questions about the weather.

Here are the sentences to evaluate:
1. What is the forecast for tomorrow?
2. I think it might rain.
3. Is it sunny outside?
4. Let's go to the beach.
"""
        ONE_SHOT_ASSISTANT = "1, 3"

        USER_PROMPT = f"The description is: {explanation}.\n\nHere are the sentences to evaluate:\n\n{examples_as_str}"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ONE_SHOT_USER},
            {"role": "assistant", "content": ONE_SHOT_ASSISTANT},
            {"role": "user", "content": USER_PROMPT},
        ]

    def score_accuracy(
        self, predictions: list[int], scoring_examples: list[Example]
    ) -> float:
        classifications = [
            i in predictions for i in range(1, len(scoring_examples) + 1)
        ]
        correct_classifications = [ex.is_active for ex in scoring_examples]
        return sum(
            [c == cc for c, cc in zip(classifications, correct_classifications)]
        ) / len(classifications)

    def score_f1(self, predictions: list[int], scoring_examples: list[Example]):
        classifications = [
            i in predictions for i in range(1, len(scoring_examples) + 1)
        ]
        correct_classifications = [ex.is_active for ex in scoring_examples]

        # Edge case: no positives in ground truth and predictions
        if sum(correct_classifications) == 0 and sum(classifications) == 0:
            return {"f1": 1.0, "precision": 1.0, "recall": 1.0}

        tp = sum([c and cc for c, cc in zip(classifications, correct_classifications)])
        fp = sum(
            [c and not cc for c, cc in zip(classifications, correct_classifications)]
        )
        fn = sum(
            [not c and cc for c, cc in zip(classifications, correct_classifications)]
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    async def get_response(
        self,
        messages: list[dict],
        max_tokens: int,
        n_completions: int = 1,
        debug: bool = False,
    ) -> list[str]:
        """
        Generic API usage function for OpenAI
        """
        for message in messages:
            assert message.keys() == {"content", "role"}
            assert message["role"] in ["system", "user", "assistant"]

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        result = await client.chat.completions.create(
            # model="gpt-4o-mini",
            # model="google/gemini-2.5-pro",
            # model="openai/gpt-oss-120b",
            # model="deepseek/deepseek-r1-0528",
            model="deepseek/deepseek-chat-v3.1",
            messages=messages,
            n=n_completions,
            max_tokens=max_tokens,
            stream=False,
        )
        if debug:
            display_messages(
                messages
                + [{"role": "assistant", "content": result.choices[0].message.content}]
            )

        return [choice.message.content.strip() for choice in result.choices]

    def parse_explanation(self, explanation: str) -> str:
        return explanation.split("activates on")[-1].rstrip(".").strip()

    def parse_predictions(self, predictions: str) -> list[int]:
        predictions_split = (
            predictions.strip().rstrip(".").replace("and", ",").split(",")
        )
        predictions_list = [i.strip() for i in predictions_split if i.strip() != ""]
        if predictions_list == ["None"]:
            return []
        assert all(
            pred.strip().isdigit() for pred in predictions_list
        ), f"Prediction parsing error: predictions should be comma-separated numbers, found {predictions!r}"
        predictions = [int(pred.strip()) for pred in predictions_list]
        return predictions


def get_k_largest_indices(
    texts: list[str],
    acts: Float[Tensor, "n_texts"],
    k: int,
    no_overlap: bool = True,
    act_threshold: float | None = None,
):
    """
    Returns the indices of the k largest activations,
    optionally ensuring that the texts at those indices
    are unique

    Partial code from https://arena-chapter1-transformer-interp.streamlit.app/[1.3.2]_Interpretability_with_SAEs#exercise-implement-autointerp-scoring
    """
    assert len(texts) == acts.shape[0]
    assert k <= len(texts)

    if not no_overlap:
        return acts.topk(k=k).indices.tolist()

    # If we want no overlap, we have to do a bit more work
    # to ensure that the texts at the returned indices
    # are unique
    seen = set()
    unique_top_indices = []
    for idx in acts.topk(k=len(texts)).indices.tolist():
        text = texts[idx]
        if text not in seen and (
            acts[idx].item() > act_threshold if act_threshold is not None else True
        ):
            unique_top_indices.append(idx)
            seen.add(text)
        if len(unique_top_indices) == k:
            break
    assert (
        len(unique_top_indices) == k
    ), f"Could only find {len(unique_top_indices)} unique texts"
    return unique_top_indices


def display_messages(messages: Messages):
    print(
        tabulate(
            [m.values() for m in messages],
            tablefmt="simple_grid",
            maxcolwidths=[None, 120],
        )
    )
