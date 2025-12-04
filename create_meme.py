import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def create_statistics_meme(
    original_img: Any,
    stipple_img: Any,
    block_letter_img: Any,
    masked_stipple_img: Any,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white",
) -> None:
    """
    Create the 4-panel statistics meme and save it as a PNG.

    Panels:
      1. Reality (original image)
      2. Your Model (stipple)
      3. Selection Bias (block letter S)
      4. Estimate (masked stipple)
    """
    imgs = [
        np.asarray(original_img, dtype=np.float32),
        np.asarray(stipple_img, dtype=np.float32),
        np.asarray(block_letter_img, dtype=np.float32),
        np.asarray(masked_stipple_img, dtype=np.float32),
    ]

    titles = ["Reality", "Your Model", "Selection Bias", "Estimate"]

    if not all(img.shape == imgs[0].shape for img in imgs):
        shapes = [img.shape for img in imgs]
        raise ValueError(f"All images must have same shape. Got: {shapes}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=dpi)
    fig.set_facecolor(background_color)

    for ax, arr, title in zip(axes, imgs, titles):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=4)

    plt.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=background_color,
    )
    plt.close(fig)
