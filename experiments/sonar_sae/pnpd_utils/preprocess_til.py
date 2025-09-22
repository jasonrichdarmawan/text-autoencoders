import os

UNSUPPORTED_LANGS = ["alt", "chv", "gag", "kaa", "kjh", "krc", "kum", "sah", "tyv", "shor", "slr", "uum"]

LANG_MAP = {
    "aze": "azj_Latn",
    "bak": "bak_Cyrl",
    "crh": "crh_Latn",
    "eng": "eng_Latn",
    "kaz": "kaz_Cyrl",
    "kir": "kir_Cyrl",
    "rus": "rus_Cyrl",
    "tat": "tat_Cyrl",
    "tuk": "tuk_Latn",
    "tur": "tur_Latn",
    "uig": "uig_Arab",
    "uzb": "uzn_Latn",
    "tat": "tat_Cyrl",
    "kaz": "kaz_Cyrl",
}

def preprocess_til(directory: str):
    """
    |- alt-aze
       |- til.alt
       |- til.aze
    |- alt-bak
    |- alt-chv
    |- alt-crh
    |- ...
    """
    data = []
    for lang_pair in os.listdir(directory):
        lang1, lang2 = lang_pair.split("-")
        if lang1 in UNSUPPORTED_LANGS or lang2 in UNSUPPORTED_LANGS:
            print(f"Skipping unsupported language pair: {lang_pair}")
            continue

        with open(
            f"{directory}/{lang_pair}/til.{lang1}", "r", encoding="utf-8"
        ) as f_lang1, open(
            f"{directory}/{lang_pair}/til.{lang2}", "r", encoding="utf-8"
        ) as f_lang2:
            for line_lang1, line_lang2 in zip(f_lang1, f_lang2):
                line_lang1 = line_lang1.strip()
                line_lang2 = line_lang2.strip()
                if len(line_lang1) == 0 or len(line_lang2) == 0:
                    continue
                data.append(
                    {
                        "text_1": line_lang1,
                        "lang_1": LANG_MAP[lang1],
                        "text_2": line_lang2,
                        "lang_2": LANG_MAP[lang2],
                    }
                )
    return data