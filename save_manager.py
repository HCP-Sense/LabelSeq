
import numpy as np
import pandas as pd
from pathlib import Path

def save_array_csv(path, y, x=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if x is None:
        x = np.arange(len(y), dtype=float)
    df = pd.DataFrame({"x": x, "y": y})
    df.to_csv(path, index=False)
    return str(path)

def save_array_parquet(path, y, x=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if x is None:
        x = np.arange(len(y), dtype=float)
    df = pd.DataFrame({"x": x, "y": y})
    df.to_parquet(path, index=False)
    return str(path)

def save_array_npy(path, y):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(y, dtype=float))
    return str(path)
