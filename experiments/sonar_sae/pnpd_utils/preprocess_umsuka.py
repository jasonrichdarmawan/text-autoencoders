def preprocess_umsuka(directory: str):
    """
    |- umsuka.eng
    |- umsuka.zul
    """
    data = []
    with open(f"{directory}/umsuka.zul", "r", encoding="utf-8") as f_zul, open(
        f"{directory}/umsuka.eng", "r", encoding="utf-8"
    ) as f_eng:
        for line_zul, line_eng in zip(f_zul, f_eng):
            line_zul = line_zul.strip()
            line_eng = line_eng.strip()
            if len(line_zul) == 0 or len(line_eng) == 0:
                continue
            data.append(
                {
                    "text_1": line_zul,
                    "lang_1": "zul_Latn",
                    "text_2": line_eng,
                    "lang_2": "eng_Latn",
                }
            )
