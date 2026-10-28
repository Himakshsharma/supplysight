"""Autoencoder anomaly scorer: reconstruction error on scaled PO features.

Uses PyTorch when installed (``SUPPLYSIGHT_NN_BACKEND=torch|auto``) and falls back to a
scikit-learn MLP with a bottleneck (same architecture idea, no extra dependency).
"""

from __future__ import annotations

import numpy as np

from supplysight.utils import optional_import


class Autoencoder:
    """Dense autoencoder: in -> 16 -> 6 -> 16 -> in."""

    def __init__(self, backend: str = "auto", seed: int = 42, epochs: int = 200) -> None:
        torch = optional_import("torch") if backend in ("auto", "torch") else None
        self.backend = "torch" if torch is not None else "sklearn"
        self.seed, self.epochs = seed, epochs
        self.model = None

    # ------------------------------------------------------------------ fit
    def fit(self, x: np.ndarray) -> Autoencoder:
        """Train on (mostly normal) rows of x."""
        if self.backend == "torch":  # pragma: no cover - torch not installed in CI image
            self._fit_torch(x)
        else:
            from sklearn.neural_network import MLPRegressor

            self.model = MLPRegressor(
                hidden_layer_sizes=(16, 6, 16),
                activation="tanh",
                alpha=1e-3,
                learning_rate_init=3e-3,
                max_iter=self.epochs,
                early_stopping=True,
                random_state=self.seed,
                n_iter_no_change=15,
            )
            self.model.fit(x, x)
        return self

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Return reconstructed rows."""
        if self.backend == "torch":  # pragma: no cover
            import torch

            with torch.no_grad():
                return self.model(torch.tensor(x, dtype=torch.float32)).numpy()
        return self.model.predict(x)

    def feature_errors(self, x: np.ndarray) -> np.ndarray:
        """Squared reconstruction error per feature (used for explanations)."""
        return (x - self.reconstruct(x)) ** 2

    def score(self, x: np.ndarray) -> np.ndarray:
        """Mean squared reconstruction error per row (higher = more anomalous)."""
        return self.feature_errors(x).mean(axis=1)

    # ------------------------------------------------------------------ torch
    def _fit_torch(self, x: np.ndarray) -> None:  # pragma: no cover
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        d = x.shape[1]
        net = nn.Sequential(
            nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, 6), nn.Tanh(), nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, d)
        )
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        xt = torch.tensor(x, dtype=torch.float32)
        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(xt), generator=gen)
            for i in range(0, len(xt), 256):
                b = xt[perm[i : i + 256]]
                opt.zero_grad()
                loss = ((net(b) - b) ** 2).mean()
                loss.backward()
                opt.step()
        net.eval()
        self.model = net

    def export_onnx(self, path: str, n_features: int) -> bool:  # pragma: no cover
        """Export the torch model to ONNX (no-op for the sklearn backend)."""
        if self.backend != "torch":
            return False
        import torch

        torch.onnx.export(
            self.model,
            torch.zeros(1, n_features),
            path,
            input_names=["x"],
            output_names=["recon"],
            dynamic_axes={"x": {0: "n"}, "recon": {0: "n"}},
        )
        return True
