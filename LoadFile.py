"""loadfile.py: Opens file dialog, loads CSV/Excel/JSON/Audio into usable sequence data with optional column selection."""

__author___ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Moye Nyuysoni Glein Perry"

import tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path
import pandas as pd
import numpy as np
import librosa


MAX_POINTS = 20000  # limit for fast plotting


def _downsample(data, max_points=MAX_POINTS):
    """Reduce data length for faster plotting."""
    if len(data) > max_points:
        step = len(data) // max_points
        return data[::step]
    return data


def load_file():
    # create hidden root window
    root = tk.Tk()
    root.withdraw()

    # open file dialog
    file_path = filedialog.askopenfilename(
        title="Select a data file",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Audio files", "*.wav;*.mp3;*.aiff"),
            ("Excel files", "*.xls;*.xlsx"),
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        print("No file selected.")
        return None

    ext = Path(file_path).suffix.lower()

    # Handle tabular data
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path, engine="openpyxl")
    elif ext == ".json":
        df = pd.read_json(file_path)
    elif ext in [".wav", ".mp3", ".aiff"]:
        # audio = numpy array
        data, sr = librosa.load(file_path, sr=None)
        return _downsample(data).astype(float)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # If DataFrame is loaded → ask user to select a column
    if isinstance(df, pd.DataFrame):
        columns = list(df.columns)
        if not columns:
            print("No columns found in file.")
            return None

        # ask for column choice
        col = simpledialog.askstring(
            "Select Column",
            f"Available columns:\n{', '.join(columns)}\n\nEnter column name:"
        )

        if col not in df.columns:
            print("Invalid column selected.")
            return None

        data = df[col].dropna().values
        return _downsample(data).astype(float)

    return None
