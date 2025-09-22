import os

UNSUPPORTED_LANGS = ["orm"]

LANG_MAP = {
    "amh": "amh_Ethi",
    "ara": "arb_Arab",
    "eng": "eng_Latn",
    "fra": "fra_Latn",
    "hin": "hin_Deva",
    "ind": "ind_Latn",
    "por": "por_Latn",
    "rus": "rus_Cyrl",
    "spa": "spa_Latn",
    "zho": "zho_Hans",
    "ben": "ben_Beng",
    "ckb": "ckb_Arab",
    "kmr": "kmr_Latn",
    "din": "dik_Latn",
    "fas": "pes_Arab",
    "fuv": "fuv_Latn",
    "hau": "hau_Latn",
    "khm": "khm_Khmr",
    "kin": "kin_Latn",
    "knc": "knc_Latn",
    "lin": "lin_Latn",
    "lug": "lug_Latn",
    "mar": "mar_Deva",
    "msa": "zsm_Latn",
    "mya": "mya_Mymr",
    "npi": "npi_Deva",
    "nus": "nus_Latn",
    "prs": "prs_Arab",
    "pus": "pbt_Arab",
    "som": "som_Latn",
    "swh": "swh_Latn",
    "tam": "tam_Taml",
    "tgl": "tgl_Latn",
    "tir_ER": "tir_Ethi",
    "tir_ET": "tir_Ethi",
    "urd": "urd_Arab",
    "zul": "zul_Latn",

}

def preprocess_tico(directory: str):
    """
    |- amh-orm
       |- tico19.amh-orm.amh
       |- tico19.amh-orm.orm
    |- ara-eng
    |- ara-fra
    |- ara-hin
    |- ara-ind
    |- ara-por
    |- ara-rus
    |- ara-spa
    |- ara-zho
    |- ben-eng
    |- ben-hin
    |- ckb-eng
    |- ckb-kmr
    |- din-eng
    |- eng-fas
    |- eng-fra
    |- eng-fuv
    |- eng-hau
    |- eng-hin
    |- eng-ind
    |- eng-khm
    |- eng-kin
    |- eng-kmr
    |- eng-knc
    |- eng-lin
    |- eng-lug
    |- eng-mar
    |- eng-msa
    |- eng-mya
    |- eng-npi
    |- eng-nus
    |- eng-orm
    |- eng-por
    |- eng-prs
    |- eng-pus
    |- eng-rus
    |- eng-som
    |- eng-spa
    |- eng-swh
    |- eng-tam
    |- eng-tgl
    |- eng-tir_ER
    |- eng-tir_ET
    |- eng_urd
    |- eng-zho
    |- eng-zul
    |- fas-prs
    |- fra-fuv
    |- fra-hin
    |- fra-ind
    |- fra-kin
    |- fra-lin
    |- fra-lug
    |- fra-por
    |- fra-rus
    |- fra-spa
    |- fra-swh
    |- fra-zho
    |- fra-zul
    |- hin-ind
    |- hin-mar
    |- hin-por
    |- hin-rus
    |- hin-spa
    |- hin-urd
    |- hin-zho
    |- ind-por
    |- ind-rus
    |- ind-spa
    |- ind-zho
    |- por-rus
    |- por-spa
    |- por-zho
    |- rus-spa
    |- rus-zho
    |- spa-zho

    `download_parallel_corpora.py` in NLLB repository indicate they combine Tirginya varities, so we will combine `tir_ER` and `tir_ET` into `tir_Ethi`
    """
    data = []
    for langpair in os.listdir(directory):
        lang_1, lang_2 = langpair.split("-")
        if lang_1 in UNSUPPORTED_LANGS or lang_2 in UNSUPPORTED_LANGS:
            continue
        with open(
            f"{directory}/{langpair}/tico19.{langpair}.{lang_1}", "r", encoding="utf-8"
        ) as f_lang1, open(
            f"{directory}/{langpair}/tico19.{langpair}.{lang_2}", "r", encoding="utf-8"
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
