"""Thin PyTorch training utilities.

Provides a fit/predict cycle for feedforward networks with early stopping.
Intentionally minimal — no training framework, no config objects.
The model architecture is the caller's responsibility.

Device handling: all functions accept a `device` parameter (str or
torch.device). The caller's model must already be on the target device
before calling fit_nn(). Data tensors are moved automatically.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a device specification to a torch.device.

    Args:
        device: 'cpu', 'mps', 'cuda', a torch.device, or None.
            None auto-detects: CUDA > MPS > CPU.

    Returns:
        torch.device
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_nn(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    patience: int = 10,
    device: str | torch.device | None = None,
) -> dict:
    """Train a PyTorch model with optional early stopping.

    Modifies `model` in-place. If validation data is provided, restores
    the best checkpoint (lowest validation loss) after training.

    The model is moved to `device` at the start. All data tensors are
    created on the same device. After training, the model stays on
    `device` — call model.cpu() yourself if you need it back.

    Args:
        model: nn.Module — caller defines architecture.
        x_train, y_train: training data as numpy arrays.
        epochs: maximum epochs.
        lr: Adam learning rate.
        batch_size: mini-batch size.
        weight_decay: L2 regularization strength.
        x_val, y_val: optional validation data for early stopping.
        patience: epochs without val improvement before stopping.
        device: 'cpu', 'mps', 'cuda', or None (auto-detect).

    Returns:
        dict with keys:
            final_epoch: int — last completed epoch (1-indexed).
            train_loss: float — mean training loss in the final epoch.
            val_loss: float — validation loss in the final epoch
                (present only when validation data is provided).
            device: str — device the model trained on (e.g. "cpu", "cuda").
    """
    dev = _resolve_device(device)
    model = model.to(dev)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()

    xt = torch.as_tensor(
        np.ascontiguousarray(x_train), dtype=torch.float32, device=dev
    )
    yt = torch.as_tensor(
        np.ascontiguousarray(y_train), dtype=torch.float32, device=dev
    )

    use_val = x_val is not None and y_val is not None
    if use_val:
        xv = torch.as_tensor(
            np.ascontiguousarray(x_val), dtype=torch.float32, device=dev
        )
        yv = torch.as_tensor(
            np.ascontiguousarray(y_val), dtype=torch.float32, device=dev
        )

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    info: dict = {"device": str(dev)}

    for epoch in range(epochs):
        # ---- train epoch ----
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(xt), batch_size):
            idx = perm[start : start + batch_size]
            loss = loss_fn(model(xt[idx]), yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        info["train_loss"] = epoch_loss / max(n_batches, 1)
        info["final_epoch"] = epoch + 1

        # ---- early stopping ----
        if use_val:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(xv), yv).item()
            info["val_loss"] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    return info


class SequenceDataset(torch.utils.data.Dataset):
    """Sliding-window dataset for time-series DL (LSTM, TCN, Transformer).

    Converts a (T, F) feature matrix into (T - seq_len, seq_len, F) windows,
    each paired with the target value at the window's end.

    Data is stored on CPU regardless of training device. Use a DataLoader
    and move batches to device in the training loop — this avoids
    exhausting GPU memory for large sequence datasets.

    Args:
        features: (T, F) numpy array of features.
        targets: (T,) numpy array of targets.
        seq_len: number of time steps per window.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 seq_len: int = 20):
        self.features = torch.as_tensor(
            np.ascontiguousarray(features), dtype=torch.float32
        )
        self.targets = torch.as_tensor(
            np.ascontiguousarray(targets), dtype=torch.float32
        )
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, y


def predict_nn(
    model: nn.Module,
    x: np.ndarray,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Get predictions from a trained model.

    Args:
        model: trained nn.Module (output shape: (batch,) or (batch, 1)).
        x: feature array.
        device: device to run inference on. None uses the device the
            model's parameters are already on.

    Returns:
        1-D numpy array of predictions (on CPU).
    """
    if device is None:
        dev = next(model.parameters()).device
    else:
        dev = _resolve_device(device)
        model = model.to(dev)

    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(
            np.ascontiguousarray(x), dtype=torch.float32, device=dev
        )
        out = model(xt)
        if out.ndim > 1:
            out = out.squeeze(-1)
        return out.cpu().numpy()


class SklearnNNRegressor:
    """Scikit-learn-compatible wrapper for PyTorch feedforward networks.

    Exported from Week 04 (ML Alpha).

    Wraps the fit_nn/predict_nn utilities into a class with standard
    fit(X, y) / predict(X) / get_params() / set_params() interface,
    making it usable inside sklearn pipelines, GridSearchCV, and any
    code that expects an sklearn-style estimator.

    NaN handling: training and prediction inputs are imputed with
    per-column median (computed on training data).  This is necessary
    because PyTorch cannot handle NaN natively.

    The model architecture is a two-layer feedforward ReLU network with
    dropout. Override `_build_model()` to customize the architecture.

    Args:
        n_features: number of input features (required).
        hidden: hidden layer size (first layer; second layer = hidden // 2).
        dropout: dropout probability between layers.
        lr: Adam learning rate.
        epochs: maximum training epochs.
        batch_size: mini-batch size.
        patience: early stopping patience (epochs without improvement).
        weight_decay: L2 regularization strength.
        val_frac: fraction of training data held out for early stopping.
        device: 'cpu', 'mps', 'cuda', or None (auto-detect).
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 256,
        patience: int = 10,
        weight_decay: float = 1e-4,
        val_frac: float = 0.2,
        device: str | None = None,
    ):
        self.n_features = n_features
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.weight_decay = weight_decay
        self.val_frac = val_frac
        self.device = device
        self.model_: nn.Module | None = None
        self._train_medians: np.ndarray | None = None

    def _build_model(self) -> nn.Module:
        """Construct the feedforward network. Override for custom architectures."""
        return nn.Sequential(
            nn.Linear(self.n_features, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, max(1, self.hidden // 2)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(max(1, self.hidden // 2), 1),
        )

    def _impute(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Impute NaN with per-column median."""
        X = X.copy()
        if fit:
            self._train_medians = np.nanmedian(X, axis=0)
        medians = self._train_medians
        if medians is None:
            medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                med = medians[j] if np.isfinite(medians[j]) else 0.0
                X[mask, j] = med
        return X

    def fit(self, X, y, **kwargs):
        """Train the network with early stopping.

        Args:
            X: feature array (n_samples, n_features). NaN values are
                imputed with per-column median.
            y: target array (n_samples,).

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        X = self._impute(X, fit=True)

        # Wrap raw Sequential in a module that squeezes output
        class _Wrapper(nn.Module):
            def __init__(self_inner, net):
                super().__init__()
                self_inner.net = net

            def forward(self_inner, x):
                return self_inner.net(x).squeeze(-1)

        self.model_ = _Wrapper(self._build_model())

        n_val = max(1, int(len(X) * self.val_frac))
        X_tr, y_tr = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

        fit_nn(
            self.model_, X_tr, y_tr,
            x_val=X_val, y_val=y_val,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            patience=self.patience,
            weight_decay=self.weight_decay,
            device=self.device,
        )
        return self

    def predict(self, X):
        """Generate predictions.

        Args:
            X: feature array. NaN values are imputed using medians
                learned during fit().

        Returns:
            1-D numpy array of predictions.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float32)
        X = self._impute(X, fit=False)
        return predict_nn(self.model_, X, device=self.device)

    def get_params(self, deep=True):
        """Return estimator parameters (sklearn protocol)."""
        return dict(
            n_features=self.n_features, hidden=self.hidden,
            dropout=self.dropout, lr=self.lr, epochs=self.epochs,
            batch_size=self.batch_size, patience=self.patience,
            weight_decay=self.weight_decay, val_frac=self.val_frac,
            device=self.device,
        )

    def set_params(self, **params):
        """Set estimator parameters (sklearn protocol)."""
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __sklearn_clone__(self):
        """Return a fresh unfitted copy (sklearn protocol)."""
        return SklearnNNRegressor(**self.get_params())
