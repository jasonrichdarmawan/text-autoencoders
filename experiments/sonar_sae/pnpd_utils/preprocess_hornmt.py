import os

UNSUPPORTED_LANGS = ["aar", "orm"]

LANG_MAP = {
    "amh": "amh_Ethi",
    "eng": "eng_Latn",
    "tir": "tir_Ethi",
    "som": "som_Latn",
}

def preprocess_hornmt(directory: str):
    """
    |- aar-amh
    |- aar-eng
       |- hornmt.aar
       |- hornmt.eng
    |- aar-orm
    |- aar-som
    |- aar-tir
    |- amh-eng
    |- amh-orm
    |- amh-som
    |- amh-tir
    |- eng-orm
    |- eng-som
    |- eng-tir
    |- orm-som
    |- orm-tir
    |- som-tir
    """
    data = []
    for langpair in os.listdir(directory):
        lang_1, lang_2 = langpair.split("-")
        if lang_1 in UNSUPPORTED_LANGS or lang_2 in UNSUPPORTED_LANGS:
            continue
        with open(
            f"{directory}/{langpair}/hornmt.{lang_1}", "r", encoding="utf-8"
        ) as f_lang1, open(
            f"{directory}/{langpair}/hornmt.{lang_2}", "r", encoding="utf-8"
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