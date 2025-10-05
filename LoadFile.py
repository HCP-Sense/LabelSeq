"""loadfile.py: Opens file dialog, loads CSV/Excel/JSON/Audio into usable sequence data with chunk navigation and column dropdown."""

__author___ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Moye Nyuysoni Glein Perry"

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
import librosa

# Globals to handle chunk navigation
CURRENT_FILE = None
CURRENT_EXT = None
CURRENT_CHUNK = 0
CHUNK_SIZE = 8000
SELECTED_COLUMNS = None


def open_file_dialog():
    """Open file dialog and return path or None."""
    root = tk.Tk()
    root.withdraw()
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
    return file_path if file_path else None


def load_file(new_file=True, direction="current"):
    """
    Load data from file with chunk support.
    new_file=True -> open new file dialog.
    direction: "next", "prev", or "current".
    Returns (data, columns, current_chunk_index, total_chunks).
    """

    global CURRENT_FILE, CURRENT_EXT, CURRENT_CHUNK, SELECTED_COLUMNS

    if new_file:
        CURRENT_FILE = open_file_dialog()
        if not CURRENT_FILE:
            return None, [], 0, 0
        CURRENT_EXT = Path(CURRENT_FILE).suffix.lower()
        CURRENT_CHUNK = 0
        SELECTED_COLUMNS = None

    if not CURRENT_FILE:
        return None, [], 0, 0

    # Handle direction (chunk navigation)
    if direction == "next":
        CURRENT_CHUNK += 1
    elif direction == "prev" and CURRENT_CHUNK > 0:
        CURRENT_CHUNK -= 1

    # --- Handle different file types ---
    if CURRENT_EXT in [".csv", ".txt"]:
        total_rows = sum(1 for _ in open(CURRENT_FILE)) - 1
        total_chunks = (total_rows // CHUNK_SIZE) + 1
        skip = CURRENT_CHUNK * CHUNK_SIZE
        df = pd.read_csv(CURRENT_FILE, skiprows=range(1, skip + 1),
                         nrows=CHUNK_SIZE)
    elif CURRENT_EXT in [".xls", ".xlsx"]:
        df = pd.read_excel(CURRENT_FILE, engine="openpyxl")
        total_rows = len(df)
        total_chunks = (total_rows // CHUNK_SIZE) + 1
        start = CURRENT_CHUNK * CHUNK_SIZE
        df = df.iloc[start:start + CHUNK_SIZE]
    elif CURRENT_EXT == ".json":
        df = pd.read_json(CURRENT_FILE)
        total_rows = len(df)
        total_chunks = (total_rows // CHUNK_SIZE) + 1
        start = CURRENT_CHUNK * CHUNK_SIZE
        df = df.iloc[start:start + CHUNK_SIZE]
    elif CURRENT_EXT in [".wav", ".mp3", ".aiff"]:
        data, sr = librosa.load(CURRENT_FILE, sr=None)
        total_chunks = (len(data) // CHUNK_SIZE) + 1
        start = CURRENT_CHUNK * CHUNK_SIZE
        df = data[start:start + CHUNK_SIZE]
        return df, ["Audio"], CURRENT_CHUNK, total_chunks
    else:
        raise ValueError(f"Unsupported file type: {CURRENT_EXT}")

    # Save columns if not already selected
    columns = df.columns.tolist() if isinstance(df, pd.DataFrame) else []
    if SELECTED_COLUMNS is not None and len(columns) > 0:
        df = df[SELECTED_COLUMNS]

    return df, columns, CURRENT_CHUNK, total_chunks


def set_selected_columns(columns):
    """Set which columns to keep for subsequent loads."""
    global SELECTED_COLUMNS
    SELECTED_COLUMNS = columns
