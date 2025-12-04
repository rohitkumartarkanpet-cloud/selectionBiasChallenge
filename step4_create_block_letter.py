import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Try a few bold fonts. If none are found, fall back to default.
    """
    candidates = [
        "arialbd.ttf",
        "DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue

    # Worst case: default font (not perfect, but works)
    return ImageFont.load_default()


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9,
) -> np.ndarray:
    """
    Create a block letter image (default 'S') as a 2D array in [0, 1].

    Parameters
    ----------
    height, width : int
        Desired output image size (same as your grayscale image).
    letter : str, optional
        Letter to draw. Default is 'S'.
    font_size_ratio : float, optional
        Fraction of min(height, width) to use as font size.

    Returns
    -------
    np.ndarray
        2D array (height, width) with 0.0 = black letter, 1.0 = white background.
    """
    if height <= 0 or width <= 0:
        raise ValueError("Height and width must be positive integers.")

    # Create white background (255 = white in 'L' mode)
    img = Image.new("L", (width, height), color=255)

    # Choose font size and load a bold font
    font_size = max(10, int(font_size_ratio * min(height, width)))
    font = _load_font(font_size)

    draw = ImageDraw.Draw(img)

    # Measure text size and center it
    text_bbox = draw.textbbox((0, 0), letter, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    x = (width - text_w) // 2
    y = (height - text_h) // 2

    # Draw black letter (0 = black)
    draw.text((x, y), letter, font=font, fill=0)

    # Convert to [0, 1] float array
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr
