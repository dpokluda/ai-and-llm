<<<<<<< HEAD

=======
>>>>>>> ec0369992458dc015ccf9e13e85baf93a35c9ba7
import numpy as np
import argparse
from typing import Tuple

def add_bias_column(X: np.ndarray) -> np.ndarray:
<<<<<<< HEAD
=======
    """
    Add a column of ones to X so the first weight acts as bias.
    """
>>>>>>> ec0369992458dc015ccf9e13e85baf93a35c9ba7
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([ones, X])

def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
<<<<<<< HEAD
    return X @ w

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
=======
    """
    Predict y given augmented X and weights w.
    """
    return X @ w

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error loss.
    """
>>>>>>> ec0369992458dc015ccf9e13e85baf93a35c9ba7
    diff = y_true - y_pred
    return float(np.mean(diff * diff))

def compute_gradients(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
<<<<<<< HEAD
=======
    """
    Compute gradient of MSE with respect to weights w.
    """
>>>>>>> ec0369992458dc015ccf9e13e85baf93a35c9ba7
    n = X.shape[0]
    residual = (X @ w) - y
    grad = (2.0 / n) * (X.T @ residual)
    return grad

def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 2000,
    tol: float = 1e-8,
    verbose: bool = False,
):
<<<<<<< HEAD
=======
    """
    Gradient descent optimizer for linear regression.
    """
>>>>>>> ec0369992458dc015ccf9e13e85baf93a35c9ba7
    w = np.zeros(X.shape[1], dtype=X.dtype)
    losses = []
    prev_loss = np.inf

    for t in range(epochs):
        y_pred = predict(X, w)
        loss = mse_loss(y, y_pred)
        losses.append(loss)

        if abs(prev_loss - loss) < tol:
            if verbose:
                print(f"Early stopping at epoch {t}, loss={loss:.6f}")
            break
        prev_loss = loss

        grad = compute_gradients(X, y, w)
        w = w - lr * grad

        if verbose and (t % max(1, epochs // 10) == 0):
            print(f"Epoch {t:5d} | loss={loss:.6f}")

    return w, losses

def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
<<<<<<< HEAD
    return np.linalg.pinv(X) @ y

=======
    """
    Closed-form solution using pseudoinverse.
    """
    return np.linalg.pinv(X) @ y

# def make_synthetic_linear_data(
#     n_samples: int = 200,
#     true_bias: float = 5.0,
#     true_weights: np.ndarray = None,
#     noise_std: float = 2.0,
#     random_seed: int = 0,
# ):
#     """
#     Generate y = b + Xw + noise with synthetic data.
#     """
#     rng = np.random.default_rng(random_seed)
#     if true_weights is None:
#         true_weights = np.array([3.0, -2.0], dtype=float)
#
#     X = rng.normal(size=(n_samples, true_weights.shape[0]))
#     noise = rng.normal(scale=noise_std, size=n_samples)
#     y = true_bias + X @ true_weights + noise
#     w_true_full = np.concatenate(([true_bias], true_weights))
#     return X, y, w_true_full

>>>>>>> ec0369992458dc015ccf9e13e85baf93a35c9ba7
def make_synthetic_linear_data(
    n_samples: int = 200,
    true_bias: float = 5.0,
    true_weights: np.ndarray = None,
    noise_std: float = 2.0,
    random_seed: int = 0,
<<<<<<< HEAD
):
    rng = np.random.default_rng(random_seed)
    if true_weights is None:
        true_weights = np.array([3.0, -2.0], dtype=float)

    X = rng.normal(size=(n_samples, true_weights.shape[0]))
    noise = rng.normal(scale=noise_std, size=n_samples)
    y = true_bias + X @ true_weights + noise
    w_true_full = np.concatenate(([true_bias], true_weights))
    return X, y, w_true_full

def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42):
=======
    debug: bool = True,
):
    """
    Generate y = b + Xw + noise with synthetic data.
    """
    if debug:
        print(f"n_samples = {n_samples}")
        print(f"true_bias = {true_bias}")
        print(f"true_weights (input) = {true_weights}")
        print(f"noise_std = {noise_std}")
        print(f"random_seed = {random_seed}")

    rng = np.random.default_rng(random_seed)
    if debug:
        print(f"rng = np.random.default_rng({random_seed}) -> {rng}")

    if true_weights is None:
        true_weights = np.array([3.0, -2.0], dtype=float)
        if debug:
            print("true_weights was None; using default np.array([3.0, -2.0])")

    if debug:
        print(f"true_weights (used) = {true_weights}")
        print(f"n_features = {true_weights.shape[0]}")

    X = rng.normal(size=(n_samples, true_weights.shape[0]))
    if debug:
        if X.size <= 50:
            print(f"X =\n{X}")
        else:
            print(f"X.shape = {X.shape}")
            print(f"X (first 5 rows) =\n{X[:5]}")

    noise = rng.normal(scale=noise_std, size=n_samples)
    if debug:
        if noise.size <= 50:
            print(f"noise = {noise}")
        else:
            print(f"noise.shape = {noise.shape}")
            print(f"noise (first 10) = {noise[:10]}")

    y = true_bias + X @ true_weights + noise
    if debug:
        if y.size <= 50:
            print(f"y = {y}")
        else:
            print(f"y.shape = {y.shape}")
            print(f"y (first 10) = {y[:10]}")

    w_true_full = np.concatenate(([true_bias], true_weights))
    if debug:
        print(f"w_true_full = {w_true_full}")

    return X, y, w_true_full


def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42):
    """
    Split data into train and test sets.
    """
>>>>>>> ec0369992458dc015ccf9e13e85baf93a35c9ba7
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_ratio))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def main():
    parser = argparse.ArgumentParser(description="Linear Regression from scratch (NumPy)")
    parser.add_argument("--n", type=int, default=200, help="Number of samples for synthetic data")
    parser.add_argument("--noise", type=float, default=2.0, help="Noise std for synthetic data")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2000, help="Max epochs for GD")
    parser.add_argument("--tol", type=float, default=1e-8, help="Early stopping tolerance on loss change")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

<<<<<<< HEAD
    X, y, w_true = make_synthetic_linear_data(n_samples=args.n, noise_std=args.noise, random_seed=0)
    X_aug = add_bias_column(X)

    w_gd, losses = gradient_descent(X_aug, y, lr=args.lr, epochs=args.epochs, tol=args.tol, verbose=args.verbose)
    w_ne = normal_equation(X_aug, y)

=======
    # 1) Generate synthetic data
    X, y, w_true = make_synthetic_linear_data(n_samples=args.n, noise_std=args.noise, random_seed=0)
    X_aug = add_bias_column(X)

    # 2) Fit with Gradient Descent
    w_gd, losses = gradient_descent(X_aug, y, lr=args.lr, epochs=args.epochs, tol=args.tol, verbose=args.verbose)

    # 3) Fit with Normal Equation
    w_ne = normal_equation(X_aug, y)

    # 4) Train/test split
>>>>>>> ec0369992458dc015ccf9e13e85baf93a35c9ba7
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_ratio=0.2, seed=42)
    X_tr_aug, X_te_aug = add_bias_column(X_tr), add_bias_column(X_te)

    w_gd_train, _ = gradient_descent(X_tr_aug, y_tr, lr=args.lr, epochs=args.epochs, tol=args.tol, verbose=False)
    w_ne_train = normal_equation(X_tr_aug, y_tr)

    def mse(Xa, ya, w):
        pred = Xa @ w
        return float(np.mean((ya - pred) ** 2))

    train_mse_gd = mse(X_tr_aug, y_tr, w_gd_train)
    test_mse_gd  = mse(X_te_aug, y_te, w_gd_train)
    train_mse_ne = mse(X_tr_aug, y_tr, w_ne_train)
    test_mse_ne  = mse(X_te_aug, y_te, w_ne_train)

<<<<<<< HEAD
=======
    # Results
>>>>>>> ec0369992458dc015ccf9e13e85baf93a35c9ba7
    print("\n=== True parameters (bias first) ===")
    print(w_true)

    print("\n=== Gradient Descent (fit on all data) ===")
    print("w_gd =", w_gd)

    print("\n=== Normal Equation (fit on all data) ===")
    print("w_ne =", w_ne)

    print("\n=== Train/Test evaluation (fit on train only) ===")
    print(f"GD   -> train MSE: {train_mse_gd:.4f} | test MSE: {test_mse_gd:.4f}")
    print(f"NE   -> train MSE: {train_mse_ne:.4f} | test MSE: {test_mse_ne:.4f}")

<<<<<<< HEAD
=======
    # Example prediction
>>>>>>> ec0369992458dc015ccf9e13e85baf93a35c9ba7
    x_new = np.array([[0.5, -1.2]], dtype=float)
    x_new_aug = add_bias_column(x_new)
    y_hat = x_new_aug @ w_gd
    print("\nExample prediction for x_new [[0.5, -1.2]] using GD:", float(y_hat))

if __name__ == "__main__":
    main()
