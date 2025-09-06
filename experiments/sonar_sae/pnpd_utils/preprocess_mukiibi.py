def preprocess_mukiibi(directory: str):
    """
    |- mukiibi.eng
    |- mukiibi.lug
    """
    data = []
    with open(f"{directory}/mukiibi.lug", "r", encoding="utf-8") as f_lug, open(
        f"{directory}/mukiibi.eng", "r", encoding="utf-8"
    ) as f_eng:
        for line_lug, line_eng in zip(f_lug, f_eng):
            line_lug = line_lug.strip()
            line_eng = line_eng.strip()
            if len(line_lug) == 0 or len(line_eng) == 0:
                continue
            data.append(
                {
                    "text_1": line_lug,
                    "lang_1": "lug_Latn",
                    "text_2": line_eng,
                    "lang_2": "eng_Latn",
                }
            )

    return data
