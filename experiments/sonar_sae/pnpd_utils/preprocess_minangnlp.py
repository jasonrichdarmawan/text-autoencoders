import os

LANG_MAP = {
    "min_Latn": "min_Latn",
    "ind": "ind_Latn",
}


def preprocess_minangnlp(directory: str):
    """
    |- min_Latn-ind
       |- minangnlp.ind
       |- minangnlp.min_Latn
    """
    data = []
    for langpair in os.listdir(directory):
        lang_1, lang_2 = langpair.split("-")
        with open(
            f"{directory}/{langpair}/minangnlp.{lang_1}", "r", encoding="utf-8"
        ) as f_lang1, open(
            f"{directory}/{langpair}/minangnlp.{lang_2}", "r", encoding="utf-8"
        ) as f_lang2:
            for line_lang1, line_lang2 in zip(f_lang1, f_lang2):
                line_lang1 = line_lang1.strip()
                line_lang2 = line_lang2.strip()
                if len(line_lang1) == 0 or len(line_lang2) == 0:
                    continue
                data.append(
                    {
                        "text_1": line_lang1,
                        "lang_1": LANG_MAP[lang_1],
                        "text_2": line_lang2,
                        "lang_2": LANG_MAP[lang_2],
                    }
                )
    return data
