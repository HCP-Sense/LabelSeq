"""loadfile.py: Opens file dialog, loads CSV/Excel/JSON/Audio into usable sequence data."""

__author___ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Moye Nyuysoni Glein Perry"

import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pandas as pd
import librosa


def load_file():
    # create the hidden root window first
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
    if ext in [".csv", ".txt"]:
        file_content = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        file_content = pd.read_excel(file_path, engine="openpyxl")
    elif ext == ".json":
        file_content = pd.read_json(file_path)
    elif ext in [".wav", ".mp3", ".aiff"]:
        file_content, sr = librosa.load(file_path, sr=None)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


    file_content = file_content[0:10000]
    file_content = file_content * 100
 
    return file_content



