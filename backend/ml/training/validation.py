"""Dependency-light validation for multi-task training arrays."""
import numpy as np


def _validate_forecast_targets(name, y_fc, require_both_classes=False):
    """Allow NaN for unknown labels, but reject infinities and non-binary known labels."""
    if np.isinf(y_fc).any():
        raise ValueError(f"{name.capitalize()} forecast targets cannot contain infinite values.")
    known = np.isfinite(y_fc)
    if not np.isin(y_fc[known], (0, 1)).all():
        raise ValueError(f"{name.capitalize()} known forecast targets must be binary.")
    if require_both_classes:
        for horizon in range(y_fc.shape[1]):
            values = y_fc[known[:, horizon], horizon]
            if np.unique(values).size < 2:
                raise ValueError(
                    f"Every training forecast horizon must contain both positive and negative examples among known labels (horizon {horizon})."
                )


def validate_training_data(X_train, y_now_train, y_fc_train, X_val, y_now_val, y_fc_val):
    arrays = [np.asarray(value) for value in
              (X_train, y_now_train, y_fc_train, X_val, y_now_val, y_fc_val)]
    X_train, y_now_train, y_fc_train, X_val, y_now_val, y_fc_val = arrays
    if X_train.ndim != 3 or X_val.ndim != 3 or X_train.shape[1:] != X_val.shape[1:]:
        raise ValueError("Training and validation inputs must be compatible 3D sequence arrays.")
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training and validation splits must both contain samples.")
    if not all(np.isfinite(x).all() for x in (X_train, X_val)):
        raise ValueError("Training and validation inputs must contain only finite values.")
    if y_now_train.ndim != 1 or y_now_val.ndim != 1:
        raise ValueError("Nowcast targets must be one-dimensional class-index arrays.")
    if y_fc_train.ndim != 2 or y_fc_val.ndim != 2 or y_fc_train.shape[1] == 0:
        raise ValueError("Forecast targets must be two-dimensional horizon arrays.")
    if (len(y_now_train), len(y_fc_train)) != (len(X_train), len(X_train)) or \
            (len(y_now_val), len(y_fc_val)) != (len(X_val), len(X_val)):
        raise ValueError("Input and target sample counts must match within each split.")
    if y_fc_train.shape[1] != y_fc_val.shape[1]:
        raise ValueError("Training and validation forecast horizons must match.")
    for name, y_now, y_fc in (("training", y_now_train, y_fc_train), ("validation", y_now_val, y_fc_val)):
        if not np.isfinite(y_now).all() or not np.equal(y_now, np.floor(y_now)).all() or \
                ((y_now < 0) | (y_now > 3)).any():
            raise ValueError(f"{name.capitalize()} nowcast targets must be finite class indices from 0 to 3.")
        _validate_forecast_targets(name, y_fc, require_both_classes=(name == "training"))
    if np.unique(y_now_train).size < 2:
        raise ValueError("Training data must contain at least two nowcast classes.")


def validate_classical_training_data(X_train, y_now_train, y_fc_train,
                                     X_val, y_now_val, y_fc_val):
    X_train, y_now_train, y_fc_train, X_val, y_now_val, y_fc_val = [
        np.asarray(value) for value in
        (X_train, y_now_train, y_fc_train, X_val, y_now_val, y_fc_val)
    ]
    if X_train.ndim != 2 or X_val.ndim != 2 or X_train.shape[1] == 0 or \
            X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Classical training and validation features must be compatible non-empty matrices.")
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Classical training and validation splits must both contain samples.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise ValueError("Classical model features must contain only finite values.")
    for name, y_now, y_fc, n_rows in (
            ("training", y_now_train, y_fc_train, len(X_train)),
            ("validation", y_now_val, y_fc_val, len(X_val))):
        if y_now.ndim != 1 or y_fc.ndim != 2 or len(y_now) != n_rows or len(y_fc) != n_rows:
            raise ValueError(f"Classical {name} features and targets have incompatible shapes.")
        if not np.isfinite(y_now).all() or not np.equal(y_now, np.floor(y_now)).all() or \
                ((y_now < 0) | (y_now > 3)).any():
            raise ValueError(f"Classical {name} nowcast targets must be finite class indices from 0 to 3.")
        _validate_forecast_targets(name, y_fc, require_both_classes=(name == "training"))
    if np.unique(y_now_train).size < 2:
        raise ValueError("Classical training data must contain at least two nowcast classes.")
