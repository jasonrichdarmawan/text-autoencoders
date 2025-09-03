from dataclasses import dataclass


@dataclass
class AutoInterpConfig:
    """
    Controls all parameters for how autointerp will work.

    Arguments:
        latents:                    The latent indices we'll be
                                    studying
        no_overlap:                 Whether to allow overlapping
                                    top activating sequences
                                    for generation and scoring
        act_threshold_frac          The fraction of the maximum act
                                    to use as the act threshold
        total_tokens:               The total number of tokens
                                    we'll gather data for
        scoring:                    Whether to perform the scoring phase,
                                    or just return explanation
        max_tokens:                 The maximum number of tokens
                                    to generate
        n_top_ex_for_generation:    The number of top activating
                                    sequences to use for generation
        n_top_ex_for_scoring:       The number of top sequences
                                    to use for scoring
        n_random_ex_for_scoring:    The number of random sequences
                                    to use for scoring

    Partial code from https://arena-chapter1-transformer-interp.streamlit.app/[1.3.2]_Interpretability_with_SAEs#exercise-implement-autointerp-scoring
    """

    latents: list[int]
    # buffer: int = 10
    no_overlap: bool = True
    act_threshold_frac: float = 0.1
    total_tokens: int = 500_000
    scoring: bool = True
    max_tokens: int = 65536
    use_examples_in_explanation_prompt: bool = True
    n_top_ex_for_generation: int = 10
    n_top_ex_for_scoring: int = 4
    n_random_ex_for_scoring: int = 8

    @property
    def n_top_ex(self):
        """
        When fetching data, we get the top examples
        for generation & scoring simultaneously
        """
        return self.n_top_ex_for_generation + self.n_top_ex_for_scoring

    @property
    def n_ex_for_scoring(self) -> int:
        """
        For scoring phase, we use a randomly shuffled mix of top-k
        activations and random seqs
        """
        return self.n_top_ex_for_scoring + self.n_random_ex_for_scoring

    @property
    def n_latents(self) -> int:
        return len(self.latents)
