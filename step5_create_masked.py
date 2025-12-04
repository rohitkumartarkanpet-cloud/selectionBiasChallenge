import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply a mask to a stippled image to simulate selection bias.

    Where the mask is dark (mask < threshold), stipple points are removed
    by setting them to white (1.0). Else, the original stipple value is kept.

    Parameters
    ----------
    stipple_img : np.ndarray
        2D array with stipple values (0.0 = dot, 1.0 = background).
    mask_img : np.ndarray
        2D array in [0, 1] with same shape as stipple_img.
    threshold : float, optional
        Pixels in mask < threshold are considered inside the "S" and removed.

    Returns
    -------
    np.ndarray
        2D array same shape as inputs with biased stipple image.
    """
    stipple = np.asarray(stipple_img, dtype=np.float32)
    mask = np.asarray(mask_img, dtype=np.float32)

    if stipple.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: stipple {stipple.shape} vs mask {mask.shape}"
        )

    masked = stipple.copy()
    # Inside the S (dark) → remove dots (set to white)
    masked[mask < threshold] = 1.0

    return masked
