def preprocess_cmu_hatian(directory: str):
    """
    |- cmu.eng
    |- cmu.hat
    """
    data = []
    with open(f"{directory}/cmu.hat", "r", encoding="utf-8") as f_hat, open(
        f"{directory}/cmu.eng", "r", encoding="utf-8"
    ) as f_eng:
        for line_hat, line_eng in zip(f_hat, f_eng):
            line_hat = line_hat.strip()
            line_eng = line_eng.strip()
            if len(line_hat) == 0 or len(line_eng) == 0:
                continue
            data.append(
                {
                    "text_1": line_hat,
                    "lang_1": "hat_Latn",
                    "text_2": line_eng,
                    "lang_2": "eng_Latn",
                }
            )

    return data
