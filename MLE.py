"""
Maximum Likelihood Estimation (MLE) for Simple Linear Regression
Synthetic, reproducible demonstration for an Applied Machine Learning assignment.

Run from the project root:
    python MLE.py

Static figures are saved in figures/; the animated GIF is saved in the project root.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Fixed random seed makes the example reproducible.
rng = np.random.default_rng(21)
x = np.arange(1, 11, dtype=float)
y = 2.0 + 1.5 * x + rng.normal(0.0, 0.9, size=x.size)
n = len(x)

# Design matrix includes the intercept column.
X = np.column_stack((np.ones_like(x), x))
beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
y_hat = X @ beta_hat
residuals = y - y_hat
rss = float(np.sum(residuals ** 2))

# Under Gaussian errors, the MLE variance uses RSS / n (not RSS / (n - 2)).
sigma2_hat = rss / n
sigma_hat = float(np.sqrt(sigma2_hat))

print("Gaussian linear-regression MLE demonstration")
print(f"Number of observations: {n}")
print(f"Intercept estimate: {beta_hat[0]:.4f}")
print(f"Slope estimate: {beta_hat[1]:.4f}")
print(f"Residual sum of squares: {rss:.4f}")
print(f"MLE variance: {sigma2_hat:.4f}")
print(f"MLE standard deviation: {sigma_hat:.4f}")

# Figure 1: observed data and fitted line.
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.scatter(x, y, label="Observed synthetic data")
ax.plot(x, y_hat, label="MLE fitted regression", linewidth=2)
ax.set_xlabel("Predictor x")
ax.set_ylabel("Response y")
ax.set_title("Gaussian MLE: observed data and fitted line")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "fit.png", dpi=180)
plt.close(fig)

# Profile negative log likelihood for the slope. For each candidate slope,
# the intercept is re-estimated and the Gaussian variance is re-estimated.
slopes = np.linspace(beta_hat[1] - 0.9, beta_hat[1] + 0.9, 300)
profile_nll = []
for slope in slopes:
    intercept = float(np.mean(y - slope * x))
    residual_sum = float(np.sum((y - intercept - slope * x) ** 2))
    variance = max(residual_sum / n, 1e-12)
    nll = n / 2.0 * (np.log(2.0 * np.pi * variance) + 1.0)
    profile_nll.append(nll)

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(slopes, profile_nll, label="Profile negative log likelihood")
ax.axvline(beta_hat[1], linestyle="--", label=f"MLE slope = {beta_hat[1]:.3f}")
ax.set_xlabel("Candidate slope")
ax.set_ylabel("Profile negative log likelihood (smaller is better)")
ax.set_title("Likelihood optimization over the slope")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "likelihood.png", dpi=180)
plt.close(fig)

# Four key frames for the software-demonstration section.
frame_slopes = [beta_hat[1] - 0.65, beta_hat[1] - 0.25, beta_hat[1] + 0.20, beta_hat[1]]
for i, slope in enumerate(frame_slopes, start=1):
    intercept = float(np.mean(y - slope * x))
    line = intercept + slope * x
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.scatter(x, y, label="Observed data")
    ax.plot(x, line, label=f"Candidate line (slope={slope:.2f})", linewidth=2)
    if i == 4:
        ax.plot(x, y_hat, linestyle="--", label="MLE fitted line")
    ax.set_xlabel("Predictor x")
    ax.set_ylabel("Response y")
    ax.set_title(f"Demonstration frame {i}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"frame{i}.png", dpi=160)
    plt.close(fig)

# Animated demonstration: candidate regression line moves toward the MLE fit.
frame_slopes_gif = np.linspace(beta_hat[1] - 0.8, beta_hat[1] + 0.8, 45)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.scatter(x, y, label="Observed synthetic data", zorder=3)
line, = ax.plot([], [], linewidth=2, label="Candidate regression line")
ax.plot(x, y_hat, "--", linewidth=1.5, label="MLE fitted line")
ax.set_xlim(float(x.min()) - 0.5, float(x.max()) + 0.5)
pad = max(float(np.ptp(y)) * 0.18, 1.0)
ax.set_ylim(float(y.min()) - pad, float(y.max()) + pad)
ax.set_xlabel("Predictor x")
ax.set_ylabel("Response y")
ax.set_title("Gaussian MLE: candidate line approaching the fitted line")
ax.grid(alpha=0.25)
ax.legend()
def update(frame):
    slope = float(frame_slopes_gif[frame])
    intercept = float(np.mean(y - slope * x))
    line.set_data(x, intercept + slope * x)
    ax.set_title(f"Candidate slope = {slope:.3f} | MLE slope = {beta_hat[1]:.3f}")
    return (line,)
ani = FuncAnimation(fig, update, frames=len(frame_slopes_gif), interval=100, blit=False)
ani.save(ROOT / "MLE_animation.gif", writer=PillowWriter(fps=10))
plt.close(fig)

results = (
    "Gaussian linear-regression MLE demonstration\n"
    f"Number of observations: {n}\n"
    f"Intercept estimate: {beta_hat[0]:.6f}\n"
    f"Slope estimate: {beta_hat[1]:.6f}\n"
    f"Residual sum of squares: {rss:.6f}\n"
    f"MLE variance: {sigma2_hat:.6f}\n"
    f"MLE standard deviation: {sigma_hat:.6f}\n"
)
(ROOT / "results.txt").write_text(results, encoding="utf-8")
print(f"Figures saved to: {FIGURES}")
