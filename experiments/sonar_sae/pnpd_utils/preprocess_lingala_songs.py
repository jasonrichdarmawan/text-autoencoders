def preprocess_lingala_songs(directory: str):
    """
    |- songs.fr-lin.fr
    |- songs.fr-lin.lin
    """
    data = []
    with open(f"{directory}/songs.fr-lin.fr", "r", encoding="utf-8") as f_fra, open(
        f"{directory}/songs.fr-lin.lin", "r", encoding="utf-8"
    ) as f_lin:
        for line_fra, line_lin in zip(f_fra, f_lin):
            line_fra = line_fra.strip()
            line_lin = line_lin.strip()
            if len(line_fra) == 0 or len(line_lin) == 0:
                continue
            data.append(
                {
                    "text_1": line_lin,
                    "lang_1": "lin_Latn",
                    "text_2": line_fra,
                    "lang_2": "fra_Latn",
                }
            )
    return data
