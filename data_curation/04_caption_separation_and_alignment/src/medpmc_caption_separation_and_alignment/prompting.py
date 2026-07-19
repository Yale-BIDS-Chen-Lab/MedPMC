"""Backend-neutral prompt construction for caption separation."""

from __future__ import annotations

PROMPT_VERSION = "medpmc_caption_separation_v1"
OUTPUT_DELIMITER = "||"
PROMPT_PREFIX = (
    "Your task is to separate the given caption into subcaptions. "
    "You are provided with a compound figure, {num_subfigures} subfigures, and a main caption. "
    "For each subfigure, extract the corresponding subcaption from the main caption and separate them using \"||\". "
    "Make sure the number and order of subcaptions match the given subfigures."
)


def build_prompt(
    main_caption: str,
    num_subfigures: int,
    image_token: str = "<IMAGE_TOKEN>",
) -> str:
    """Build the released MedPMC explicit-image-token prompt."""
    if num_subfigures < 1:
        raise ValueError("num_subfigures must be at least 1")
    subfigure_blocks = "\n".join(
        f"# Subfigure\n{image_token}" for _ in range(num_subfigures)
    )
    return (
        PROMPT_PREFIX.format(num_subfigures=num_subfigures)
        + "\n\n"
        + f"# Compound Figure\n{image_token}\n"
        + f"{subfigure_blocks}\n"
        + "# Main Caption\n"
        + str(main_caption or "")
    )


def build_ordered_images_prompt(main_caption: str, num_subfigures: int) -> str:
    """Build a generic prompt for VLMs that manage image tokens internally."""
    if num_subfigures < 1:
        raise ValueError("num_subfigures must be at least 1")
    return (
        PROMPT_PREFIX.format(num_subfigures=num_subfigures)
        + "\n\n"
        + "The images are provided in this exact order: the compound figure first, "
        + f"followed by subfigures 0 through {num_subfigures - 1}.\n"
        + "# Main Caption\n"
        + str(main_caption or "")
    )


def clean_prediction(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2:
            value = "\n".join(lines[1:-1]).strip()
    return value


def parse_subcaptions(text: str) -> list[str]:
    value = clean_prediction(text)
    if not value:
        return []
    return [part.strip() for part in value.split(OUTPUT_DELIMITER)]
