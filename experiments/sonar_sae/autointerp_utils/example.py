class Example:
    """
    Data for a single example sequence.

    Partial code from https://arena-chapter1-transformer-interp.streamlit.app/[1.3.2]_Interpretability_with_SAEs#exercise-implement-autointerp-scoring
    """

    def __init__(
        self,
        text: str,
        act: float,
        act_threshold: float,
    ):
        self.text = text
        self.act = act
        self.is_active = act >= act_threshold

    def to_str(self) -> str:
        return self.text.replace(">><<", "").replace("�", "").replace("\n", "↵")
