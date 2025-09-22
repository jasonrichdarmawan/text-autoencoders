import os

LANG_MAP = {
    "en": "eng_Latn",
    "bn": "ben_Beng",
    "hi": "hin_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "or": "ori_Orya",
    "pa": "pan_Guru",
    "ta": "tam_Taml",
    "te": "tel_Telu",
}

def preprocess_indic_nlp(directory: str, split: str):
    """
    |- alt
       |- en-bn
          |- train.bn
          |- train.en
       |- en-hi
          |- train.hi
          |- train.en
    |- bibleuedin
       |- en-gu
          |- train.en
          |- train.gu
       |- ...
    |- ...
    """
    data = []
    for dataset_path in os.listdir(f"{directory}/{split}"):
        langpair_path = f"{directory}/{split}/{dataset_path}"
        if os.path.isdir(langpair_path) is False:
            continue
        for langpair in os.listdir(langpair_path):
            lang_1, lang_2 = langpair.split("-")
            with open(
                f"{langpair_path}/{langpair}/{split}.{lang_1}", "r", encoding="utf-8"
            ) as f_lang1, open(
                f"{langpair_path}/{langpair}/{split}.{lang_2}", "r", encoding="utf-8"
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
