def preprocess_menyo20k(directory: str):
    """
    |- menyo20k.eng
    |- menyo20k.yor
    """
    data = []
    with open(f"{directory}/menyo20k.yor", "r", encoding="utf-8") as f_yor, open(
        f"{directory}/menyo20k.eng", "r", encoding="utf-8"
    ) as f_eng:
        for line_yor, line_eng in zip(f_yor, f_eng):
            line_yor = line_yor.strip()
            line_eng = line_eng.strip()
            if len(line_yor) == 0 or len(line_eng) == 0:
                continue
            data.append(
                {
                    "text_1": line_yor,
                    "lang_1": "yor_Latn",
                    "text_2": line_eng,
                    "lang_2": "eng_Latn",
                }
            )

    return data
