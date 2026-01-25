"""Visualization tools for Lattice Elasto-Plastic Models."""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


def _plot_scalar_lattice(
    stress: np.ndarray,
    thresholds: np.ndarray,
    title: str = "Lattice EPM State (Scalar)",
    figsize: tuple[int, int] = (12, 5),
    cmap_stress: str = "coolwarm",
    cmap_thresh: str = "viridis",
) -> plt.Figure:
    """Plot scalar stress and threshold fields side-by-side.

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)

    # Stress Plot
    max_stress = np.max(np.abs(stress))
    im1 = ax1.imshow(
        stress, cmap=cmap_stress, vmin=-max_stress, vmax=max_stress, origin="lower"
    )
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


def _plot_tensorial_lattice(
    stress: np.ndarray,
    thresholds: np.ndarray,
    title: str = "Lattice EPM State (Tensorial)",
    figsize: tuple[int, int] = (16, 4),
    cmap_stress: str = "coolwarm",
    cmap_thresh: str = "viridis",
) -> plt.Figure:
    """Plot tensorial stress components and threshold fields.

    Args:
        stress: 3D array of stress tensor (3, L, L) with [σ_xx, σ_yy, σ_xy].
        thresholds: 2D array of local yield thresholds (L, L).
        title: Overall figure title.
        figsize: Figure size (width, height).
        cmap_stress: Colormap for stress fields (diverging).
        cmap_thresh: Colormap for threshold field (sequential).

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(title)

    # Find global stress scale for consistent colormaps
    max_stress = np.max(np.abs(stress))

    # Component labels
    labels = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$"]

    # Plot each stress component
    for i in range(3):
        im = axes[i].imshow(
            stress[i],
            cmap=cmap_stress,
            vmin=-max_stress,
            vmax=max_stress,
            origin="lower",
        )
        axes[i].set_title(labels[i])
        plt.colorbar(im, ax=axes[i])
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")

    # Threshold plot
    im_thresh = axes[3].imshow(thresholds, cmap=cmap_thresh, origin="lower")
    axes[3].set_title(r"Yield Thresholds $\sigma_c$")
    plt.colorbar(im_thresh, ax=axes[3])
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")

    plt.tight_layout()
    return fig


def plot_lattice_fields(
    stress: np.ndarray | jax.Array,
    thresholds: np.ndarray | jax.Array,
    title: str | None = None,
    figsize: tuple[int, int] | None = None,
    cmap_stress: str = "coolwarm",
    cmap_thresh: str = "viridis",
) -> plt.Figure:
    """Plot EPM lattice fields with auto-detection of scalar vs tensorial stress.

    Automatically detects whether stress is scalar (L, L) or tensorial (3, L, L)
    and dispatches to the appropriate plotting function.

    Args:
        stress: Either (L, L) scalar or (3, L, L) tensorial stress field.
        thresholds: 2D array of local yield thresholds (L, L).
        title: Overall figure title (auto-generated if None).
        figsize: Figure size (width, height) (auto-selected if None).
        cmap_stress: Colormap for stress field (diverging).
        cmap_thresh: Colormap for threshold field (sequential).

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If stress shape is invalid.
    """
    stress = np.array(stress)
    thresholds = np.array(thresholds)

    if stress.ndim == 2:
        # Scalar stress field
        default_title = "Lattice EPM State (Scalar)"
        default_figsize = (12, 5)
        return _plot_scalar_lattice(
            stress,
            thresholds,
            title=title or default_title,
            figsize=figsize or default_figsize,
            cmap_stress=cmap_stress,
            cmap_thresh=cmap_thresh,
        )
    elif stress.ndim == 3 and stress.shape[0] == 3:
        # Tensorial stress field
        default_title = "Lattice EPM State (Tensorial)"
        default_figsize = (16, 4)
        return _plot_tensorial_lattice(
            stress,
            thresholds,
            title=title or default_title,
            figsize=figsize or default_figsize,
            cmap_stress=cmap_stress,
            cmap_thresh=cmap_thresh,
        )
    else:
        raise ValueError(
            f"Invalid stress shape: {stress.shape}. "
            "Expected (L, L) for scalar or (3, L, L) for tensorial."
        )


def animate_stress_evolution(
    stress_history: np.ndarray | jax.Array,
    interval: int = 50,
    cmap: str = "coolwarm",
    save_path: str | None = None,
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
        animated=True,
    )
    ax.set_title("Time Step: 0")
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


def plot_tensorial_fields(
    stress: np.ndarray | jax.Array,
    figsize: tuple[int, int] = (15, 4),
    cmap: str = "coolwarm",
    ax: plt.Axes | list[plt.Axes] | None = None,
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot all three stress tensor components in a 3-panel layout.

    Args:
        stress: Stress tensor of shape (3, L, L) with [σ_xx, σ_yy, σ_xy].
        figsize: Figure size (width, height).
        cmap: Colormap for stress fields (diverging, centered at 0).
        ax: Optional pre-existing axes (3 axes required).
        **kwargs: Additional arguments passed to imshow.

    Returns:
        Tuple of (Figure, list of 3 Axes).
    """
    stress = np.array(stress)

    if stress.shape[0] != 3:
        raise ValueError(f"Expected stress shape (3, L, L), got {stress.shape}")

    # Create figure if axes not provided
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        if not isinstance(ax, (list, np.ndarray)) or len(ax) != 3:
            raise ValueError("If ax provided, must be list/array of 3 axes")
        axes = ax
        fig = axes[0].get_figure()

    # Find global stress scale for consistent colormaps
    max_stress = np.max(np.abs(stress))

    # Component labels with LaTeX
    labels = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$"]

    # Plot each component
    for i in range(3):
        im = axes[i].imshow(
            stress[i],
            cmap=cmap,
            vmin=-max_stress,
            vmax=max_stress,
            origin="lower",
            **kwargs,
        )
        axes[i].set_title(labels[i])
        plt.colorbar(im, ax=axes[i])
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")

    plt.tight_layout()
    return fig, list(axes)


def plot_normal_stress_field(
    stress: np.ndarray | jax.Array,
    nu: float = 0.5,
    figsize: tuple[int, int] = (6, 5),
    cmap: str = "coolwarm",
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot first normal stress difference field N₁ = σ_xx - σ_yy.

    Args:
        stress: Stress tensor of shape (3, L, L) with [σ_xx, σ_yy, σ_xy].
        nu: Poisson's ratio (not used for N₁, but kept for consistency).
        figsize: Figure size (width, height).
        cmap: Colormap (diverging, centered at 0).
        ax: Optional pre-existing axis.
        **kwargs: Additional arguments passed to imshow.

    Returns:
        Tuple of (Figure, Axes).
    """
    stress = np.array(stress)

    if stress.shape[0] != 3:
        raise ValueError(f"Expected stress shape (3, L, L), got {stress.shape}")

    # Compute N₁
    N1 = stress[0] - stress[1]

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot with symmetric colormap centered at 0
    max_N1 = np.max(np.abs(N1))
    im = ax.imshow(N1, cmap=cmap, vmin=-max_N1, vmax=max_N1, origin="lower", **kwargs)
    ax.set_title(r"$N_1 = \sigma_{xx} - \sigma_{yy}$")
    plt.colorbar(im, ax=ax, label=r"$N_1$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    return fig, ax


def plot_von_mises_field(
    stress: np.ndarray | jax.Array,
    thresholds: np.ndarray | jax.Array,
    nu: float = 0.5,
    figsize: tuple[int, int] = (12, 5),
    ax: plt.Axes | list[plt.Axes] | None = None,
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot von Mises effective stress and normalized yield map.

    Creates a 2-panel figure:

    - Left: σ_eff with viridis (sequential)
    - Right: σ_eff/σ_c with RdYlGn_r centered at 1
      (Green <1: elastic, Yellow ≈1: near yield, Red >1: plastic)

    Args:
        stress: Stress tensor of shape (3, L, L) with [σ_xx, σ_yy, σ_xy].
        thresholds: Yield thresholds of shape (L, L).
        nu: Poisson's ratio for plane strain constraint.
        figsize: Figure size (width, height).
        ax: Optional pre-existing axes (2 axes required).
        **kwargs: Additional arguments passed to imshow.

    Returns:
        Tuple of (Figure, list of 2 Axes).
    """
    stress = np.array(stress)
    thresholds = np.array(thresholds)

    if stress.shape[0] != 3:
        raise ValueError(f"Expected stress shape (3, L, L), got {stress.shape}")

    # Import von Mises function
    from rheojax.utils.epm_kernels_tensorial import compute_von_mises_stress

    # Reshape stress for von Mises computation: (3, L, L) -> (L, L, 3)
    stress_reshaped = np.moveaxis(stress, 0, -1)

    # Convert to JAX for computation
    stress_jax = jnp.array(stress_reshaped)
    sigma_eff = compute_von_mises_stress(stress_jax, nu)
    sigma_eff = np.array(sigma_eff)

    # Compute normalized stress
    sigma_normalized = sigma_eff / (thresholds + 1e-12)

    # Create figure if not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        if not isinstance(ax, (list, np.ndarray)) or len(ax) != 2:
            raise ValueError("If ax provided, must be list/array of 2 axes")
        axes = ax
        fig = axes[0].get_figure()

    # Left panel: σ_eff with viridis (sequential)
    im1 = axes[0].imshow(sigma_eff, cmap="viridis", origin="lower", **kwargs)
    axes[0].set_title(r"von Mises $\sigma_{\mathrm{eff}}$")
    plt.colorbar(im1, ax=axes[0], label=r"$\sigma_{\mathrm{eff}}$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Right panel: σ_eff/σ_c with RdYlGn_r centered at 1
    # Clip to reasonable range for visualization
    vmin = 0.0
    vmax = 2.0
    im2 = axes[1].imshow(
        sigma_normalized,
        cmap="RdYlGn_r",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        **kwargs,
    )
    axes[1].set_title(r"Normalized Stress $\sigma_{\mathrm{eff}} / \sigma_c$")
    plt.colorbar(im2, ax=axes[1], label=r"$\sigma_{\mathrm{eff}} / \sigma_c$")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    plt.tight_layout()
    return fig, list(axes)


def plot_normal_stress_ratio(
    shear_rates: np.ndarray | jax.Array,
    N1: np.ndarray | jax.Array,
    sigma_xy: np.ndarray | jax.Array,
    figsize: tuple[int, int] = (8, 6),
    ax: plt.Axes | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot log-log of N₁/σ_xy vs shear rate.

    Args:
        shear_rates: Array of shear rates.
        N1: First normal stress difference values.
        sigma_xy: Shear stress values.
        figsize: Figure size (width, height).
        ax: Optional pre-existing axis.
        **kwargs: Additional arguments passed to plot.

    Returns:
        Tuple of (Figure, Axes).
    """
    shear_rates = np.array(shear_rates)
    N1 = np.array(N1)
    sigma_xy = np.array(sigma_xy)

    # Compute ratio (avoid division by zero)
    ratio = N1 / (np.abs(sigma_xy) + 1e-12)

    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Log-log plot
    ax.loglog(shear_rates, ratio, marker="o", **kwargs)
    ax.set_xlabel(r"Shear Rate $\dot{\gamma}$ (1/s)")
    ax.set_ylabel(r"$N_1 / \sigma_{xy}$")
    ax.set_title("Normal Stress Ratio")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    return fig, ax


def animate_tensorial_evolution(
    history: dict[str, np.ndarray | jax.Array],
    component: str = "all",
    interval: int = 50,
    save_path: str | None = None,
    **kwargs,
) -> animation.FuncAnimation:
    """Create animation of tensorial stress field evolution.

    Args:
        history: Dictionary with keys:
            - 'stress': Stress history of shape (T, 3, L, L)
            - 'time': Time array of shape (T,)
        component: Component to animate:
            - 'all': All 3 components (3-panel animation)
            - 'xx', 'yy', 'xy': Individual components
            - 'N1': First normal stress difference
            - 'vm': von Mises effective stress
        interval: Delay between frames in milliseconds.
        save_path: If provided, save animation to this path.
        **kwargs: Additional arguments (e.g., nu for von Mises).

    Returns:
        Matplotlib FuncAnimation object.
    """
    stress_history = np.array(history["stress"])
    time = np.array(history["time"])

    T, n_comp, L, _ = stress_history.shape

    if n_comp != 3:
        raise ValueError(
            f"Expected stress shape (T, 3, L, L), got {stress_history.shape}"
        )

    nu = kwargs.get("nu", 0.5)

    # Determine component mapping
    component_map = {
        "xx": 0,
        "yy": 1,
        "xy": 2,
    }

    if component == "all":
        # 3-panel animation
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Find global limits
        max_stress = np.max(np.abs(stress_history))

        labels = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$"]
        images = []

        for i in range(3):
            im = axes[i].imshow(
                stress_history[0, i],
                cmap="coolwarm",
                vmin=-max_stress,
                vmax=max_stress,
                origin="lower",
                animated=True,
            )
            axes[i].set_title(f"{labels[i]} - t={time[0]:.3f}")
            plt.colorbar(im, ax=axes[i])
            images.append(im)

        def update(frame):
            for i in range(3):
                images[i].set_array(stress_history[frame, i])
                axes[i].set_title(f"{labels[i]} - t={time[frame]:.3f}")
            return images

        anim = animation.FuncAnimation(
            fig, update, frames=T, interval=interval, blit=True
        )

    elif component in component_map:
        # Single component animation
        idx = component_map[component]
        fig, ax = plt.subplots(figsize=(6, 5))

        max_val = np.max(np.abs(stress_history[:, idx]))
        im = ax.imshow(
            stress_history[0, idx],
            cmap="coolwarm",
            vmin=-max_val,
            vmax=max_val,
            origin="lower",
            animated=True,
        )
        ax.set_title(f"$\\sigma_{{{component}}}$ - t={time[0]:.3f}")
        plt.colorbar(im, ax=ax)

        def update(frame):
            im.set_array(stress_history[frame, idx])
            ax.set_title(f"$\\sigma_{{{component}}}$ - t={time[frame]:.3f}")
            return (im,)

        anim = animation.FuncAnimation(
            fig, update, frames=T, interval=interval, blit=True
        )

    elif component == "N1":
        # Normal stress difference animation
        fig, ax = plt.subplots(figsize=(6, 5))

        # Compute N₁ for all frames
        N1_history = stress_history[:, 0] - stress_history[:, 1]
        max_N1 = np.max(np.abs(N1_history))

        im = ax.imshow(
            N1_history[0],
            cmap="coolwarm",
            vmin=-max_N1,
            vmax=max_N1,
            origin="lower",
            animated=True,
        )
        ax.set_title(f"$N_1$ - t={time[0]:.3f}")
        plt.colorbar(im, ax=ax, label=r"$N_1$")

        def update(frame):
            im.set_array(N1_history[frame])
            ax.set_title(f"$N_1$ - t={time[frame]:.3f}")
            return (im,)

        anim = animation.FuncAnimation(
            fig, update, frames=T, interval=interval, blit=True
        )

    elif component == "vm":
        # von Mises animation
        fig, ax = plt.subplots(figsize=(6, 5))

        # Import von Mises function
        from rheojax.utils.epm_kernels_tensorial import compute_von_mises_stress

        # Compute von Mises for all frames
        vm_history = []
        for t_idx in range(T):
            stress_reshaped = np.moveaxis(stress_history[t_idx], 0, -1)
            stress_jax = jnp.array(stress_reshaped)
            sigma_eff = compute_von_mises_stress(stress_jax, nu)
            vm_history.append(np.array(sigma_eff))

        vm_history = np.array(vm_history)
        max_vm = np.max(vm_history)

        im = ax.imshow(
            vm_history[0],
            cmap="viridis",
            vmin=0,
            vmax=max_vm,
            origin="lower",
            animated=True,
        )
        ax.set_title(f"$\\sigma_{{\\mathrm{{eff}}}}$ - t={time[0]:.3f}")
        plt.colorbar(im, ax=ax, label=r"$\sigma_{\mathrm{eff}}$")

        def update(frame):
            im.set_array(vm_history[frame])
            ax.set_title(f"$\\sigma_{{\\mathrm{{eff}}}}$ - t={time[frame]:.3f}")
            return (im,)

        anim = animation.FuncAnimation(
            fig, update, frames=T, interval=interval, blit=True
        )

    else:
        raise ValueError(
            f"Unknown component: {component}. "
            "Expected 'all', 'xx', 'yy', 'xy', 'N1', or 'vm'."
        )

    if save_path:
        anim.save(save_path)

    return anim
