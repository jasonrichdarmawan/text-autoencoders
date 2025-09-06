def preprocess_giossa(directory: str):
    """
    |- giossa.grn
    |- giossa.spa
    """
    data = []
    with open(f"{directory}/giossa.grn", "r", encoding="utf-8") as f_grn, open(
        f"{directory}/giossa.spa", "r", encoding="utf-8"
    ) as f_spa:
        for line_grn, line_spa in zip(f_grn, f_spa):
            line_grn = line_grn.strip()
            line_spa = line_spa.strip()
            if len(line_grn) == 0 or len(line_spa) == 0:
                continue
            data.append(
                {
                    "text_1": line_grn,
                    "lang_1": "grn_Latn",
                    "text_2": line_spa,
                    "lang_2": "spa_Latn",
                }
            )

    return data
