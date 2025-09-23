def preprocess_xhosa_navy(directory: str):
    """
    |- XhosaNavy.eng
    |- XhosaNavy.xho
    """
    data = []
    with open(f"{directory}/XhosaNavy.eng", "r", encoding="utf-8") as f_eng, open(
        f"{directory}/XhosaNavy.xho", "r", encoding="utf-8"
    ) as f_xho:
        for line_eng, line_xho in zip(f_eng, f_xho):
            line_eng = line_eng.strip()
            line_xho = line_xho.strip()
            if len(line_eng) == 0 or len(line_xho) == 0:
                continue
            data.append(
                {
                    "text_1": line_xho,
                    "lang_1": "xho_Latn",
                    "text_2": line_eng,
                    "lang_2": "eng_Latn",
                }
    )
    return data