import os


def preprocess_nllb_seed(directory: str):
    """
    |- ace_Arab-eng_Latn
       |- ace_Arab
       |- eng_Latn
    ...
    """
    data = []

    for langpair in os.listdir(directory):
        lang1, lang2 = langpair.split("-")
        with open(
            f"{directory}/{langpair}/{lang1}", "r", encoding="utf-8"
        ) as f_lang1, open(
            f"{directory}/{langpair}/{lang2}", "r", encoding="utf-8"
        ) as f_lang2:
            for line_lang1, line_lang2 in zip(f_lang1, f_lang2):
                line_lang1 = line_lang1.strip()
                line_lang2 = line_lang2.strip()
                if len(line_lang1) == 0 or len(line_lang2) == 0:
                    continue
                data.append(
                    {
                        "text_1": line_lang1,
                        "lang_1": lang1,
                        "text_2": line_lang2,
                        "lang_2": lang2,
                    }
                )
    return data
