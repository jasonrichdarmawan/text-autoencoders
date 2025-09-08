from tabulate import tabulate


def print_autointerp_results(
    results_dict, latents: list[int] | None = None, acts: list[float] | None = None
):
    print(
        tabulate(
            [
                (
                    k,
                    v.get("explanation", ""),
                    f"{acts[int(k)]:.3f}" if acts is not None else "N/A",
                    f"{v.get('f1', 0):.3f}",
                    f"{v.get('accuracy', 0):.3f}",
                    f"{v.get('precision', 0):.3f}",
                    f"{v.get('recall', 0):.3f}",
                )
                for k, v in sorted(
                    results_dict.items(),
                    key=lambda item: (
                        (
                            -acts[int(item[0])]
                            if acts is not None
                            and latents is not None
                            and int(item[0]) in latents
                            else 0
                        ),
                        -float(item[1]["f1"]),
                    ),
                )
                if latents is None or int(k) in latents
            ],
            headers=[
                "Latent",
                "Explanation",
                "Activation",
                "F1",
                "Accuracy",
                "Precision",
                "Recall",
            ],
            tablefmt="github",
        )
    )

    # report missing latents
    if latents is not None:
        missing_latents = [l for l in latents if str(l) not in results_dict]
        if len(missing_latents) > 0:
            print(
                f"Missing latent dimension explanations:",
                " ".join(map(str, missing_latents)),
            )
