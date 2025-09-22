import os

UNSUPPORTED_LANGS = ["nde", "ven"]

LANG_MAP = {
    "afr": "afr_Latn",
    "eng": "eng_Latn",
    "sot": "sot_Latn",
    "ssw": "ssw_Latn",
    "tsn": "tsn_Latn",
    "tso": "tso_Latn",
    "xho": "xho_Latn",
    "zul": "zul_Latn",
}

def preprocess_mburisano(directory: str):
    """
    |- afr-eng
       |- mburisano.afr
       |- mburisano.eng
    |- eng-nde
    |- eng-sot
    |- eng-ssw
    |- eng-tsn
    |- eng-tso
    |- eng-ven
    |- eng-xho
    |- eng-zul
    """
    data = []
    for langpair in os.listdir(directory):
        lang_1, lang_2 = langpair.split("-")
        if lang_1 in UNSUPPORTED_LANGS or lang_2 in UNSUPPORTED_LANGS:
            continue
        with open(
            f"{directory}/{langpair}/mburisano.{lang_1}", "r", encoding="utf-8"
        ) as f_lang1, open(
            f"{directory}/{langpair}/mburisano.{lang_2}", "r", encoding="utf-8"
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
