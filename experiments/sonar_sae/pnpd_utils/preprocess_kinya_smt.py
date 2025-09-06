def preprocess_kinya_smt(directory: str):
    """
    |- kinyasmt.eng
    |- kinyasmt.kin
    """
    data = []
    with open(f"{directory}/kinyasmt.kin", "r", encoding="utf-8") as f_kin, open(
        f"{directory}/kinyasmt.eng", "r", encoding="utf-8"
    ) as f_eng:
        for line_kin, line_eng in zip(f_kin, f_eng):
            line_kin = line_kin.strip()
            line_eng = line_eng.strip()
            if len(line_kin) == 0 or len(line_eng) == 0:
                continue
            data.append(
                {
                    "text_1": line_kin,
                    "lang_1": "kin_Latn",
                    "text_2": line_eng,
                    "lang_2": "eng_Latn",
                }
            )
    return data
