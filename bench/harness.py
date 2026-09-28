"""Timing harness.

For every model: `warmup` discarded runs, then `repeats` measured runs. Each run records
  transfer_ms  host<->GPU copies (CuPy only; 0 otherwise)
  fit_ms       model.fit
  predict_ms   model.predict on the test split
plus accuracy and macro F1 on the test split, so a faster-but-worse result is visible.
GPU work is synchronised before each timer stops, otherwise asynchronous kernels would
be billed to whichever step happens to block next.
"""
import time

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from bench import models


class _Gpu:
    """Thin wrapper so the CuPy import only happens for the CuPy implementation."""

    def __init__(self):
        import cupy
        import cupyx.scipy.sparse
        self.cp = cupy
        self.cpsparse = cupyx.scipy.sparse
        # Create the CUDA context now so its one-off cost is not charged to the first run.
        self.cp.zeros(1)
        self.sync()

    def sync(self):
        self.cp.cuda.Device().synchronize()

    def to_device(self, a):
        if sp.issparse(a):
            return self.cpsparse.csr_matrix(a)
        return self.cp.asarray(a)

    def to_host(self, a):
        if self.cpsparse.issparse(a):
            return a.get()
        if isinstance(a, self.cp.ndarray):
            return self.cp.asnumpy(a)
        return a


def _timed(fn, sync=None):
    start = time.perf_counter()
    out = fn()
    if sync:
        sync()
    return out, (time.perf_counter() - start) * 1000


def _run_once(name, implementation, X_train, X_test, y_train, y_test, gpu):
    model = models.build(name, implementation)
    transfer_ms = 0.0

    if gpu is not None:
        (Xtr, Xte, ytr), ms = _timed(
            lambda: (gpu.to_device(X_train), gpu.to_device(X_test), gpu.to_device(y_train)), gpu.sync)
        transfer_ms += ms
        # XGBoost has no CuPy-sparse input path, so sparse data stays on the host for it
        # (XGBoost then does the copy itself, inside fit).
        native = name in models.GPU_NATIVE and not sp.issparse(X_train)
        if not native:
            (Xtr, Xte, ytr), ms = _timed(lambda: (gpu.to_host(Xtr), gpu.to_host(Xte), gpu.to_host(ytr)))
            transfer_ms += ms
        sync = gpu.sync
    else:
        Xtr, Xte, ytr, sync = X_train, X_test, y_train, None

    _, fit_ms = _timed(lambda: model.fit(Xtr, ytr), sync)
    y_pred, predict_ms = _timed(lambda: model.predict(Xte), sync)

    if gpu is not None:
        y_pred, ms = _timed(lambda: gpu.to_host(y_pred))
        transfer_ms += ms

    y_pred = np.asarray(y_pred).astype(y_test.dtype)
    return model, {
        "transfer_ms": transfer_ms,
        "fit_ms": fit_ms,
        "predict_ms": predict_ms,
        "total_ms": transfer_ms + fit_ms + predict_ms,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }


def run(dataset, implementation, split, model_names, repeats, warmup):
    """Yield one result row per measured run."""
    X_train, X_test, y_train, y_test = split
    gpu = _Gpu() if implementation == "CuPY" else None

    for name in model_names:
        Xtr, Xte = X_train, X_test
        if name in models.NEEDS_SCALING:
            # with_mean=False keeps sparse (TF-IDF) input sparse.
            scaler = StandardScaler(with_mean=not sp.issparse(X_train))
            Xtr = scaler.fit_transform(X_train)
            Xte = scaler.transform(X_test)

        for i in range(warmup + repeats):
            model, row = _run_once(name, implementation, Xtr, Xte, y_train, y_test, gpu)
            status = "warm-up" if i < warmup else f"run {i - warmup + 1}/{repeats}"
            print(f"  {name:<34} {status:<9} fit {row['fit_ms']:>11.1f} ms  "
                  f"predict {row['predict_ms']:>9.1f} ms  transfer {row['transfer_ms']:>8.1f} ms  "
                  f"acc {row['accuracy']:.4f}  F1 {row['f1_macro']:.4f}", flush=True)
            if i < warmup:
                continue
            yield {
                "Dataset": dataset,
                "Algorithm": name,
                "Implementation": implementation,
                # Records which library actually ran, e.g. "sklearnex.linear_model" when the
                # Intelex patch applied, "sklearn.ensemble._weight_boosting" when it did not.
                "Backend": type(model).__module__,
                "Repeat": i - warmup + 1,
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
                "n_features": X_train.shape[1],
                **row,
            }
