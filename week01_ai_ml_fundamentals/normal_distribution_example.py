# python
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Optional: if SciPy is available, we'll overlay exact PDFs and 2D contours
try:
    from scipy.stats import norm, multivariate_normal
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def generate_points(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # 2D standard normal: mean 0, std 1 per coordinate
    X = rng.normal(size=(n_samples, 2))
    return X


def plot_samples_with_distributions(X: np.ndarray):
    x1, x2 = X[:, 0], X[:, 1]

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1], wspace=0.3, hspace=0.3)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_hist_x1 = fig.add_subplot(gs[0, 1])
    ax_hist_x2 = fig.add_subplot(gs[1, 0])

    # 2D scatter
    ax_scatter.scatter(x1, x2, s=20, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax_scatter.set_title("2D scatter of samples (each dim ~ N(0,1))")
    ax_scatter.set_xlabel("x1")
    ax_scatter.set_ylabel("x2")
    ax_scatter.axhline(0, color="gray", lw=1, alpha=0.5)
    ax_scatter.axvline(0, color="gray", lw=1, alpha=0.5)

    # 2D Normal contours (N([0,0], I)) if SciPy is available
    if SCIPY_AVAILABLE:
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        x_min, x_max = x1.min() - 3, x1.max() + 3
        y_min, y_max = x2.min() - 3, x2.max() + 3
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 150),
            np.linspace(y_min, y_max, 150),
        )
        pos = np.dstack((xx, yy))
        rv = multivariate_normal(mean=mean, cov=cov)
        zz = rv.pdf(pos)
        ax_scatter.contour(xx, yy, zz, levels=6, cmap="viridis", alpha=0.8)
    else:
        # Fallback: draw reference circles (rough guide to radial density)
        for r, c in zip([1, 2, 3], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            theta = np.linspace(0, 2 * np.pi, 200)
            ax_scatter.plot(r * np.cos(theta), r * np.sin(theta), color=c, alpha=0.7)

    # Histogram for x1 with standard normal PDF
    ax_hist_x1.hist(x1, bins=20, density=True, color="#aec7e8", edgecolor="k", alpha=0.8)
    ax_hist_x1.set_title("x1 ~ N(0,1)")
    ax_hist_x1.set_xlabel("x1")
    ax_hist_x1.set_ylabel("density")
    xs = np.linspace(x1.min() - 3, x1.max() + 3, 300)
    if SCIPY_AVAILABLE:
        ax_hist_x1.plot(xs, norm.pdf(xs, loc=0, scale=1), "r-", lw=2, label="N(0,1) PDF")
        ax_hist_x1.legend()

    # Histogram for x2 with standard normal PDF
    ax_hist_x2.hist(x2, bins=20, density=True, color="#ffbb78", edgecolor="k", alpha=0.8)
    ax_hist_x2.set_title("x2 ~ N(0,1)")
    ax_hist_x2.set_xlabel("x2")
    ax_hist_x2.set_ylabel("density")
    xs2 = np.linspace(x2.min() - 3, x2.max() + 3, 300)
    if SCIPY_AVAILABLE:
        ax_hist_x2.plot(xs2, norm.pdf(xs2, loc=0, scale=1), "r-", lw=2, label="N(0,1) PDF")
        ax_hist_x2.legend()

    out = "normal_distribution_example.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {out}")

    # plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot 2D standard normal samples and distributions")
    parser.add_argument("--n", type=int, default=200, help="Number of samples")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    X = generate_points(n_samples=args.n, seed=args.seed)
    print(f"Generated X with shape: {X.shape}")
    plot_samples_with_distributions(X)


if __name__ == "__main__":
    main()