"""Visualization tools for Lattice Elasto-Plastic Models."""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


def plot_lattice_fields(
    stress: Union[np.ndarray, jax.Array],
    thresholds: Union[np.ndarray, jax.Array],
    title: str = "Lattice EPM State",
    figsize: Tuple[int, int] = (12, 5),
    cmap_stress: str = "coolwarm",
    cmap_thresh: str = "viridis",
) -> plt.Figure:
    """Plot the current stress and yield threshold fields side-by-side.

    Args:
        stress: 2D array of local stress values (L, L).
        thresholds: 2D array of local yield thresholds (L, L).
        title: Overall figure title.
        figsize: Figure size (width, height).
        cmap_stress: Colormap for stress field (diverging).
        cmap_thresh: Colormap for threshold field (sequential).

    Returns:
        Matplotlib Figure object.
    """
    stress = np.array(stress)
    thresholds = np.array(thresholds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)

    # Stress Plot
    # Use symmetric clim centered on 0 for stress if possible, or auto
    max_stress = np.max(np.abs(stress))
    im1 = ax1.imshow(stress, cmap=cmap_stress, vmin=-max_stress, vmax=max_stress, origin="lower")
    ax1.set_title(r"Stress Field $\sigma_{ij}$")
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Threshold Plot
    im2 = ax2.imshow(thresholds, cmap=cmap_thresh, origin="lower")
    ax2.set_title(r"Yield Thresholds $\sigma_c$")
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    plt.tight_layout()
    return fig


def animate_stress_evolution(
    stress_history: Union[np.ndarray, jax.Array],
    interval: int = 50,
    cmap: str = "coolwarm",
    save_path: Optional[str] = None,
) -> animation.FuncAnimation:
    """Create an animation of the stress field evolution.

    Args:
        stress_history: 3D array of stress history (Time, L, L).
        interval: Delay between frames in milliseconds.
        cmap: Colormap for stress.
        save_path: If provided, save the animation to this path (e.g. 'movie.mp4').

    Returns:
        Matplotlib FuncAnimation object.
    """
    history = np.array(stress_history)
    n_frames, L, _ = history.shape

    fig, ax = plt.subplots(figsize=(6, 5))

    # Determine global limits for stable coloring
    max_val = np.max(np.abs(history))

    im = ax.imshow(
        history[0],
        cmap=cmap,
        vmin=-max_val,
        vmax=max_val,
        origin="lower",
        animated=True
    )
    ax.set_title(f"Time Step: 0")
    plt.colorbar(im, ax=ax, label=r"Stress $\sigma$")

    def update(frame):
        im.set_array(history[frame])
        ax.set_title(f"Time Step: {frame}")
        return (im,)

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=True
    )

    if save_path:
        # Requires ffmpeg or imagemagick installed
        anim.save(save_path)

    return anim
