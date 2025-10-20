__author__ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Development"

# Project Imports
from original import OriginalPanel
from annotator import AnnotatorPanel
from annotation import AnnotationPanel
from result import ResultPanel
from InteractiveSequencePanel import InteractiveSequencePanel
from LoadFile import load_file
try:
    # optional helper; not guaranteed present in your LoadFile.py
    from LoadFile import set_selected_columns as _set_selected_columns
except Exception:
    _set_selected_columns = None

from model import Model
from save_manager import save_array_csv, save_array_parquet, save_array_npy
from project_manager import save_project, load_project
from dialogs import interpret_non_numeric_series  # keep if you use it

# Third-Party Imports
import wx
import numpy as np
import pandas as pd
import typing
from pathlib import Path
from collections import Counter

# Color cycle for panels
COLOR_CYCLE = [
    wx.BLUE,
    wx.RED,
    wx.GREEN,
    wx.Colour(255, 165, 0),    # Orange
    wx.Colour(128, 0, 128),    # Purple
    wx.Colour(0, 206, 209),    # Turquoise
    wx.Colour(255, 105, 180),  # Hot pink
]

class MainFrame(wx.Frame):
    DEFAULT_PANEL_LENGTH = 100

    def __init__(self):
        super().__init__(None, title="LabelSeq — 4-panel Annotator", size=(1320, 920))

        # ----- Core State -----
        self.color_index: int = 0
        self.signals: typing.List[dict] = []
        self.current_file_content = None          # DataFrame / ndarray / array-like for current chunk
        self.available_columns: typing.List[str] = []
        self.selected_column_index: typing.Optional[int] = None
        self.current_chunk: int = 0
        self.total_chunks: int = 0
        self.current_path: typing.Optional[Path] = None

        # user-chosen visible cap for speed/clarity (set on first import)
        self.target_visible_count: typing.Optional[int] = None

        # ----- Core Array Data -----
        self.original: np.ndarray = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        self.annotator_seq: np.ndarray = self.original.copy()
        self.annotation_seq: np.ndarray = self.original - self.annotator_seq
        self.result: np.ndarray = self.original + self.annotation_seq

        # ----- Panels -----
        self.original_panel: typing.Optional[OriginalPanel] = None
        self.annotation_panel: typing.Optional[AnnotationPanel] = None
        self.annotator_panel: typing.Optional[AnnotatorPanel] = None
        self.result_panel: typing.Optional[ResultPanel] = None

        # ----- Model -----
        self.model = Model()
        self.auto_label_enabled = True

        self.model.set_original(self.original)
        self.model.set_annotator(self.annotator_seq)

        # ----- Build UI -----
        self._build_ui()
        self.update_panels()
        self.Centre()
        self.Show()

        # ----- Timer for Updating Playhead (optional) -----
        self._playhead_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_tick, self._playhead_timer)
        self._playhead_timer.Start(200)

    # =========================================================
    # UI Construction
    # =========================================================
    def _build_ui(self):
        main_panel = wx.Panel(self)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # ----------------- Legend Panel -----------------
        self.legend_panel = wx.Panel(main_panel, size=(220, -1))
        self.legend_panel.SetBackgroundColour(wx.Colour(245, 245, 245))
        self.legend_sizer = wx.BoxSizer(wx.VERTICAL)
        legend_label = wx.StaticText(self.legend_panel, label="Legend")
        font = legend_label.GetFont()
        try:
            font.MakeBold()
            legend_label.SetFont(font)
        except Exception:
            pass
        self.legend_sizer.Add(legend_label, 0, wx.ALL, 8)
        self.legend_panel.SetSizer(self.legend_sizer)

        # ----------------- Right Panel (Controls + Data Panels) -----------------
        right_panel = wx.Panel(main_panel)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        # ----- Buttons Row -----
        btn_row = wx.BoxSizer(wx.HORIZONTAL)

        self.btn_undo = wx.Button(right_panel, label="Undo")
        self.btn_redo = wx.Button(right_panel, label="Redo")
        self.btn_save_proj = wx.Button(right_panel, label="Save Project")
        self.btn_load_proj = wx.Button(right_panel, label="Load Project")
        self.btn_export_ann = wx.Button(right_panel, label="Export Annotation")
        self.btn_export_res = wx.Button(right_panel, label="Export Result")
        self.btn_import = wx.Button(right_panel, label="Import Data")
        self.btn_prev = wx.Button(right_panel, label="<< Previous Chunk")
        self.btn_next = wx.Button(right_panel, label="Next Chunk >>")
        self.btn_delete = wx.Button(right_panel, label="Delete File")
        self.btn_add_anno = wx.Button(right_panel, label="Add Annotation")

        self.btn_undo.Bind(wx.EVT_BUTTON, self.on_undo)
        self.btn_redo.Bind(wx.EVT_BUTTON, self.on_redo)
        self.btn_save_proj.Bind(wx.EVT_BUTTON, self.on_save_project)
        self.btn_load_proj.Bind(wx.EVT_BUTTON, self.on_load_project)
        self.btn_export_ann.Bind(wx.EVT_BUTTON, self.on_export_annotation)
        self.btn_export_res.Bind(wx.EVT_BUTTON, self.on_export_result)
        self.btn_import.Bind(wx.EVT_BUTTON, self.on_load_file)
        self.btn_prev.Bind(wx.EVT_BUTTON, self.on_prev_chunk)
        self.btn_next.Bind(wx.EVT_BUTTON, self.on_next_chunk)
        self.btn_delete.Bind(wx.EVT_BUTTON, self.on_delete_file)
        self.btn_add_anno.Bind(wx.EVT_BUTTON, self.on_add_annotation)

        self.btn_prev.Disable()
        self.btn_next.Disable()

        for btn in [self.btn_undo, self.btn_redo, self.btn_save_proj, self.btn_load_proj,
                    self.btn_export_ann, self.btn_export_res, self.btn_import,
                    self.btn_prev, self.btn_next, self.btn_delete, self.btn_add_anno]:
            btn_row.Add(btn, 0, wx.ALL, 4)

        right_sizer.Add(btn_row, 0, wx.ALIGN_CENTER_HORIZONTAL)

        # ----- Column Selector -----
        col_row = wx.BoxSizer(wx.HORIZONTAL)
        col_label = wx.StaticText(right_panel, label="Column:")
        self.column_choice = wx.Choice(right_panel, choices=[])
        self.column_choice.Bind(wx.EVT_CHOICE, self.on_column_choice)
        self.chunk_info_label = wx.StaticText(right_panel, label="")

        col_row.Add(col_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        col_row.Add(self.column_choice, 1, wx.ALL | wx.EXPAND, 6)
        col_row.Add(self.chunk_info_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        right_sizer.Add(col_row, 0, wx.EXPAND)

        # ----- Playhead Label -----
        self.playhead_label = wx.StaticText(right_panel, label="Playhead: -")
        right_sizer.Add(self.playhead_label, 0, wx.LEFT | wx.BOTTOM, 6)

        # ----- Scrollable Panel Area -----
        self.sig_area = wx.ScrolledWindow(right_panel, style=wx.VSCROLL)
        self.sig_area.SetScrollRate(0, 20)
        self.sig_sizer = wx.BoxSizer(wx.VERTICAL)
        self.sig_area.SetSizer(self.sig_sizer)
        right_sizer.Add(self.sig_area, 1, wx.EXPAND | wx.ALL, 6)

        right_panel.SetSizer(right_sizer)

        # ----- Combine Legend + Right -----
        top_sizer.Add(self.legend_panel, 0, wx.EXPAND | wx.ALL, 6)
        top_sizer.Add(right_panel, 1, wx.EXPAND | wx.ALL, 6)
        main_panel.SetSizer(top_sizer)

        # ----- Status Bar -----
        self.CreateStatusBar()
        self.SetStatusText("Ready")

    # =========================================================
    # Active Learning — confirmation & application
    # (Called from InteractiveSequencePanel._on_al_suggest_clicked)
    # =========================================================
    def on_active_labels(self, indices: list, labels: list, source_panel):
        if not indices:
            wx.MessageBox("No indices suggested.", "Active Learning", wx.OK | wx.ICON_INFORMATION)
            return

        counts = Counter(labels or [])
        summary_lines = [f"{lab if lab else '(unlabeled)'}: {cnt}" for lab, cnt in counts.items()]
        summary = "\n".join(summary_lines) if summary_lines else f"{len(indices)} points"

        msg = (f"Active Learning Suggestions\n\n"
               f"Total points: {len(indices)}\n"
               f"{summary}\n\n"
               f"Apply these suggestions to the Annotation layer? (value = 1.0)")
        dlg = wx.MessageDialog(self, msg, "Confirm Suggestions", wx.YES_NO | wx.ICON_QUESTION)
        res = dlg.ShowModal()
        dlg.Destroy()

        if res != wx.ID_YES:
            self.SetStatusText("Suggestions discarded.")
            return

        self.on_apply_active_learning(indices, assign_value=1.0)

        # Clear overlay after applying
        try:
            if self.result_panel:
                self.result_panel.set_active_suggestions([], [])
        except Exception:
            pass

    def on_apply_active_learning(self, indices: list, assign_value: float = 1.0):
        if not indices:
            wx.MessageBox("No indices selected from active learning.", "Info", wx.OK | wx.ICON_INFORMATION)
            return
        if not hasattr(self.model, "annotation"):
            wx.MessageBox("Model has no annotation array.", "Error", wx.OK | wx.ICON_ERROR)
            return

        try:
            for idx in indices:
                if 0 <= idx < len(self.model.annotation):
                    self.model.annotation[idx] = float(assign_value)
        except Exception as e:
            wx.MessageBox(f"Failed to apply labels: {e}", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.model.recompute()

        if self.annotation_panel:
            self.annotation_panel.seq = self.model.annotation.copy()
            self.annotation_panel.n = len(self.annotation_panel.seq)
            self.annotation_panel.Refresh()
        if self.result_panel:
            self.result_panel.seq = self.model.result.copy()
            self.result_panel.n = len(self.result_panel.seq)
            self.result_panel.Refresh()

        if hasattr(self, "_refresh_legend"):
            self._refresh_legend()

        self.SetStatusText(f"Applied annotation={assign_value} to {len(indices)} points via active learning.")

    # =========================================================
    # Button Handlers (added)
    # =========================================================
    def on_undo(self, evt):
        # delegate to annotator panel if available
        if self.annotator_panel and hasattr(self.annotator_panel, "_undo"):
            self.annotator_panel._undo()
            self.SetStatusText("Undo.")
        else:
            self.SetStatusText("Nothing to undo.")

    def on_redo(self, evt):
        if self.annotator_panel and hasattr(self.annotator_panel, "_redo"):
            self.annotator_panel._redo()
            self.SetStatusText("Redo.")
        else:
            self.SetStatusText("Nothing to redo.")

    def on_save_project(self, evt):
        with wx.FileDialog(self, "Save Project", wildcard="JSON files (*.json)|*.json",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        panels = {
            "original": self.original_panel,
            "annotation": self.annotation_panel,
            "annotator": self.annotator_panel,
            "result": self.result_panel,
        }
        try:
            save_project(path, self.model, panels)
            self.SetStatusText(f"Project saved: {path}")
        except Exception as e:
            wx.MessageBox(f"Save failed: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_load_project(self, evt):
        with wx.FileDialog(self, "Load Project", wildcard="JSON files (*.json)|*.json",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        panels = {
            "original": self.original_panel,
            "annotation": self.annotation_panel,
            "annotator": self.annotator_panel,
            "result": self.result_panel,
        }
        try:
            load_project(path, self.model, panels)
            self._refresh_legend()
            self.SetStatusText(f"Project loaded: {path}")
        except Exception as e:
            wx.MessageBox(f"Load failed: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_export_annotation(self, evt):
        with wx.FileDialog(self, "Export Annotation", wildcard="CSV (*.csv)|*.csv;|Parquet (*.parquet)|*.parquet;|NumPy (*.npy)|*.npy",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()

        try:
            ext = Path(path).suffix.lower()
            if ext == ".csv":
                save_array_csv(path, self.model.annotation)
            elif ext == ".parquet":
                save_array_parquet(path, self.model.annotation)
            elif ext == ".npy":
                save_array_npy(path, self.model.annotation)
            else:
                save_array_csv(str(Path(path).with_suffix(".csv")), self.model.annotation)
            self.SetStatusText(f"Annotation exported: {path}")
        except Exception as e:
            wx.MessageBox(f"Export failed: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_export_result(self, evt):
        with wx.FileDialog(self, "Export Result", wildcard="CSV (*.csv)|*.csv;|Parquet (*.parquet)|*.parquet;|NumPy (*.npy)|*.npy",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()

        try:
            ext = Path(path).suffix.lower()
            if ext == ".csv":
                save_array_csv(path, self.model.result)
            elif ext == ".parquet":
                save_array_parquet(path, self.model.result)
            elif ext == ".npy":
                save_array_npy(path, self.model.result)
            else:
                save_array_csv(str(Path(path).with_suffix(".csv")), self.model.result)
            self.SetStatusText(f"Result exported: {path}")
        except Exception as e:
            wx.MessageBox(f"Export failed: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_load_file(self, evt):
        # Load a new file (first chunk)
        data, columns, cur_idx, total = load_file(new_file=True, direction="current")
        if data is None:
            return

        self.current_file_content = data
        self.available_columns = list(columns) if isinstance(columns, list) else []
        self.current_chunk = int(cur_idx)
        self.total_chunks = int(total)

        # Column choice UI
        self.column_choice.Clear()
        if self.available_columns:
            self.column_choice.AppendItems(self.available_columns)
            self.column_choice.SetSelection(0)
            self.selected_column_index = 0
        else:
            # If data is 1D (audio), fake a single column
            self.available_columns = ["Audio"]
            self.column_choice.Append("Audio")
            self.column_choice.SetSelection(0)
            self.selected_column_index = 0

        self._apply_selected_column_to_original()
        self._update_chunk_label()
        self.update_panels()
        self._update_chunk_buttons()
        self.SetStatusText("File loaded.")

    def on_prev_chunk(self, evt):
        if self.current_chunk <= 0:
            return
        data, columns, cur_idx, total = load_file(new_file=False, direction="prev")
        self._after_chunk_load(data, columns, cur_idx, total)

    def on_next_chunk(self, evt):
        if self.current_chunk + 1 >= self.total_chunks:
            return
        data, columns, cur_idx, total = load_file(new_file=False, direction="next")
        self._after_chunk_load(data, columns, cur_idx, total)

    def _after_chunk_load(self, data, columns, cur_idx, total):
        if data is None:
            return
        self.current_file_content = data
        self.available_columns = list(columns) if isinstance(columns, list) else self.available_columns
        self.current_chunk = int(cur_idx)
        self.total_chunks = int(total)

        # If new columns, refresh selector
        if columns:
            self.column_choice.Clear()
            self.column_choice.AppendItems(self.available_columns)
            if self.selected_column_index is not None and 0 <= self.selected_column_index < len(self.available_columns):
                self.column_choice.SetSelection(self.selected_column_index)
            else:
                self.column_choice.SetSelection(0)
                self.selected_column_index = 0

        self._apply_selected_column_to_original()
        self._update_chunk_label()
        self.update_panels()
        self._update_chunk_buttons()
        self.SetStatusText(f"Chunk {self.current_chunk+1}/{self.total_chunks} loaded.")

    def on_delete_file(self, evt):
        self.current_file_content = None
        self.available_columns = []
        self.selected_column_index = None
        self.current_chunk = 0
        self.total_chunks = 0
        self.column_choice.Clear()
        # reset sequences
        self.original = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        self.annotator_seq = self.original.copy()
        self.annotation_seq = self.original - self.annotator_seq
        self.result = self.original + self.annotation_seq
        self.model.set_original(self.original)
        self.model.set_annotator(self.annotator_seq)
        self.update_panels()
        self._update_chunk_label()
        self._update_chunk_buttons()
        self.SetStatusText("File cleared.")

    def on_add_annotation(self, evt):
        # Insert a new empty AnnotationPanel stacked under existing annotations.
        color_ann = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1
        new_seq = np.zeros_like(self.original)

        p = AnnotationPanel(
            self.sig_area, new_seq, f"Annotation {self.color_index}",
            draggable=False,
            visible_count=(self.target_visible_count or 200),
            color=color_ann, role="annotation"
        )
        p.original_seq = self.original

        # Inherit current view (zoom/pan/y) from an existing panel (prefer result or original)
        inherit_from = self.result_panel or self.original_panel or None
        if inherit_from is not None:
            try:
                p.zoom_factor = float(getattr(inherit_from, "zoom_factor", 1.0))
                p.pan_offset = float(getattr(inherit_from, "pan_offset", 0.0))
                p.y_offset = float(getattr(inherit_from, "y_offset", 0.0))
            except Exception:
                pass

        self._add_panel_object(p, insert_before_result=True)

        # 🔧 CRITICAL: resync all panels so hover/zoom/pan are wired
        self._sync_panels()

        # 🔄 immediately reflow scroller and legend
        self.sig_area.FitInside()
        self.sig_area.Layout()
        self._refresh_legend()

        self.SetStatusText("Added annotation panel.")

    def on_column_choice(self, evt):
        sel = self.column_choice.GetSelection()
        if sel == wx.NOT_FOUND:
            return
        self.selected_column_index = int(sel)

        # optional: notify loader about selected columns (if supported)
        if _set_selected_columns is not None and self.available_columns:
            try:
                _set_selected_columns([self.available_columns[sel]])
            except Exception:
                pass

        self._apply_selected_column_to_original()
        self.update_panels()
        self.SetStatusText(f"Column selected: {self.available_columns[sel] if self.available_columns else 'Audio'}")

    # =========================================================
    # Helpers (legend, panels, chunk UI)
    # =========================================================
    def _normalize_seq(self, seq) -> np.ndarray:
        if seq is None:
            return np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        if isinstance(seq, pd.DataFrame):
            if seq.shape[1] > 0:
                ser = pd.to_numeric(seq.iloc[:, 0], errors='coerce').fillna(0.0)
                arr = ser.to_numpy(dtype=float)
            else:
                arr = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        elif isinstance(seq, pd.Series):
            arr = pd.to_numeric(seq, errors='coerce').fillna(0.0).to_numpy(dtype=float)
        else:
            try:
                arr = np.asarray(seq, dtype=float)
            except Exception:
                arr = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        arr = arr.flatten()
        if arr.size < 2:
            arr = np.resize(arr, max(2, self.DEFAULT_PANEL_LENGTH))
        return arr

    def _apply_selected_column_to_original(self):
        # Determine a 1D array from current_file_content & selection
        if self.current_file_content is None:
            self.original = self._normalize_seq(self.original)
            self.model.set_original(self.original)
            self.model.set_annotator(self.annotator_seq)
            return

        data = self.current_file_content
        if isinstance(data, pd.DataFrame):
            if self.available_columns and self.selected_column_index is not None:
                col = self.available_columns[self.selected_column_index]
                ser = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
            else:
                # first column fallback
                ser = pd.to_numeric(data.iloc[:, 0], errors="coerce").fillna(0.0)
            arr = ser.to_numpy(dtype=float)
        elif isinstance(data, (np.ndarray, list, tuple)):
            arr = np.asarray(data, dtype=float)
        else:
            # unknown type => best effort
            try:
                arr = np.asarray(data, dtype=float).flatten()
            except Exception:
                arr = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)

        self.original = arr.flatten()
        if self.original.size < 2:
            self.original = np.resize(self.original, max(2, self.DEFAULT_PANEL_LENGTH))

        # re-align dependent arrays
        self.annotator_seq = np.copy(self.original)
        self.annotation_seq = self.original - self.annotator_seq
        self.result = self.original + self.annotation_seq

        self.model.set_original(self.original)
        self.model.set_annotator(self.annotator_seq)

    def _clear_existing_panels(self):
        for entry in self.signals[:]:
            p = entry['panel']
            try:
                self.sig_sizer.Detach(p)
                p.Destroy()
            except Exception:
                pass
            try:
                self.signals.remove(entry)
            except ValueError:
                pass
        self.color_index = 0
        self.original_panel = None
        self.annotation_panel = None
        self.annotator_panel = None
        self.result_panel = None

    def _add_panel_object(self, sp: InteractiveSequencePanel, insert_before_result: bool = False):
        sp.SetMinSize((-1, 250))
        inserted = False

        if isinstance(sp, AnnotationPanel):
            # Insert directly under the last existing AnnotationPanel (stacking order)
            last_anno_idx = -1
            for i in range(self.sig_sizer.GetItemCount()):
                win = self.sig_sizer.GetItem(i).GetWindow()
                if isinstance(win, AnnotationPanel):
                    last_anno_idx = i
            if last_anno_idx >= 0:
                self.sig_sizer.Insert(last_anno_idx + 1, sp, 0, wx.EXPAND | wx.ALL, 5)
                inserted = True

        if not inserted:
            if self.result_panel and insert_before_result:
                try:
                    idx = next(i for i in range(self.sig_sizer.GetItemCount())
                               if self.sig_sizer.GetItem(i).GetWindow() is self.result_panel)
                    self.sig_sizer.Insert(idx, sp, 0, wx.EXPAND | wx.ALL, 5)
                except StopIteration:
                    self.sig_sizer.Add(sp, 0, wx.EXPAND | wx.ALL, 5)
            else:
                self.sig_sizer.Add(sp, 0, wx.EXPAND | wx.ALL, 5)

        self.signals.append({'panel': sp, 'formula': getattr(sp, 'formula', None)})

    def _sync_panels(self):
        all_panels = [entry['panel'] for entry in self.signals]
        for entry in self.signals:
            p = entry['panel']
            p.sync_panels = [x for x in all_panels if x is not p]

    def _clear_legend_entries(self):
        children = self.legend_panel.GetChildren()
        for i, child in enumerate(children):
            if i == 0:
                continue
            try:
                child.Destroy()
            except Exception:
                pass

    def _add_legend_row(self, label: str, color: wx.Colour):
        row = wx.Panel(self.legend_panel)
        s = wx.BoxSizer(wx.HORIZONTAL)
        sw = wx.Panel(row, size=(18, 18))
        sw.SetBackgroundColour(color)
        lbl = wx.StaticText(row, label=label)
        s.Add(sw, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 6)
        s.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL)
        row.SetSizer(s)
        # Simple color picker on click (optional)
        def on_click(evt):
            data = wx.ColourData()
            data.SetChooseFull(True)
            data.SetColour(sw.GetBackgroundColour())
            dlg = wx.ColourDialog(self, data)
            if dlg.ShowModal() == wx.ID_OK:
                col = dlg.GetColourData().GetColour()
                sw.SetBackgroundColour(col)
                sw.Refresh()
                if label in self.model.annotation_colors:
                    self.model.set_annotation_color(label, (col.Red(), col.Green(), col.Blue()))
                    if hasattr(self.result_panel, 'set_annotation_legend'):
                        self.result_panel.set_annotation_legend(self.model.annotation_colors)
            dlg.Destroy()
        row.Bind(wx.EVT_LEFT_DOWN, on_click)
        sw.Bind(wx.EVT_LEFT_DOWN, on_click)
        lbl.Bind(wx.EVT_LEFT_DOWN, on_click)
        self.legend_sizer.Add(row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

    def _refresh_legend(self):
        self._clear_legend_entries()
        seen = set()
        # Only non-annotation at the top
        for entry in self.signals:
            p = entry['panel']
            role = getattr(p, 'role', '')
            if role == 'annotation':
                continue
            title = getattr(p, 'title', getattr(p, 'label', ''))
            color = getattr(p, 'color', wx.BLACK)
            if title not in seen:
                self._add_legend_row(title, color)
                seen.add(title)

        # Annotation block (if Result supports legend queries)
        if self.result_panel and hasattr(self.result_panel, "get_annotation_legend"):
            ann_map = self.result_panel.get_annotation_legend()
            if ann_map:
                heading = wx.StaticText(self.legend_panel, label="Annotations")
                f = heading.GetFont()
                try:
                    f.MakeBold()
                    heading.SetFont(f)
                except Exception:
                    pass
                self.legend_sizer.Add(heading, 0, wx.LEFT | wx.TOP, 8)
                for name, col in ann_map.items():
                    c = col if isinstance(col, wx.Colour) else wx.Colour(col)
                    self._add_legend_row(name, c)

        self.legend_panel.Layout()

    def _update_chunk_label(self):
        if self.total_chunks > 0:
            self.chunk_info_label.SetLabel(f"Chunk {self.current_chunk+1}/{self.total_chunks}")
        else:
            self.chunk_info_label.SetLabel("")

    def _update_chunk_buttons(self):
        self.btn_prev.Enable(self.current_chunk > 0)
        self.btn_next.Enable(self.current_chunk + 1 < max(self.total_chunks, 1))

    # =========================================================
    # Panel updates
    # =========================================================
    def update_panels(self):
        self._clear_existing_panels()

        orig = self._normalize_seq(self.original)
        anntr = self._normalize_seq(self.annotator_seq) if self.annotator_seq.size >= 2 else orig.copy()
        annotation = orig - anntr
        result_seq = orig + annotation

        self.original = orig
        self.annotator_seq = anntr
        self.annotation_seq = annotation
        self.result = result_seq

        # --- Original panel ---
        color_orig = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1
        self.original_panel = OriginalPanel(
            self.sig_area, self.original, "Original", draggable=False,
            visible_count=(self.target_visible_count or 200),
            color=color_orig, role="original"
        )
        self.original_panel.original_seq = self.original
        self._add_panel_object(self.original_panel)

        # --- Annotation panel ---
        color_ann = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1
        self.annotation_panel = AnnotationPanel(
            self.sig_area, self.annotation_seq, "Annotation", draggable=False,
            visible_count=(self.target_visible_count or 200),
            color=color_ann, role="annotation"
        )
        self.annotation_panel.original_seq = self.original
        self._add_panel_object(self.annotation_panel)

        # Hook title edit -> rename in model (keeps legend + result synced)
        if hasattr(self.annotation_panel, "edit_title"):
            _orig_edit = self.annotation_panel.edit_title

            def _wrapped_edit(evt):
                old = getattr(self.annotation_panel, "title", "Annotation")
                _orig_edit(evt)
                new = getattr(self.annotation_panel, "title", old)
                if new and new != old:
                    self.model.rename_annotation(new)
                    if hasattr(self.result_panel, 'set_annotation_legend'):
                        self.result_panel.set_annotation_legend(self.model.annotation_colors)
                    self._refresh_legend()
            self.annotation_panel.edit_title = _wrapped_edit

        # --- Annotator panel ---
        color_annot = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1

        def annotator_updated_callback(new_annot_np: np.ndarray):
            new_annot = np.asarray(new_annot_np, dtype=float).flatten()
            if new_annot.size != self.original.size:
                tmp = np.zeros_like(self.original)
                tmp[:min(len(new_annot), len(tmp))] = new_annot[:len(tmp)]
                new_annot = tmp

            self.annotator_seq = new_annot
            self.annotation_seq = self.original - self.annotator_seq
            self.result = self.original + self.annotation_seq

            self.annotation_panel.seq = self.annotation_seq
            self.annotation_panel.n = len(self.annotation_seq)
            self.annotation_panel.Refresh()

            self.result_panel.seq = self.result
            self.result_panel.n = len(self.result)
            try:
                self.result_panel.set_annotator_ref(self.annotator_panel)
            except Exception:
                self.result_panel.annotator_ref = self.annotator_panel
            self.result_panel.Refresh()

            self._refresh_legend()

        self.annotator_panel = AnnotatorPanel(
            self.sig_area, self.annotator_seq, "Annotator",
            draggable=True,
            visible_count=(self.target_visible_count or 200),
            color=color_annot,
            role="annotator",
            on_update=annotator_updated_callback
        )
        self.annotator_panel.original_seq = self.original
        self._add_panel_object(self.annotator_panel)

        # --- Result panel ---
        color_res = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1
        self.result_panel = ResultPanel(
            self.sig_area, self.result, "Result",
            draggable=False,
            visible_count=(self.target_visible_count or 200),
            color=color_res, role="result"
        )
        self.result_panel.original_seq = self.original
        try:
            self.result_panel.set_annotator_ref(self.annotator_panel)
        except Exception:
            self.result_panel.annotator_ref = self.annotator_panel
        if hasattr(self.result_panel, 'set_annotation_legend'):
            self.result_panel.set_annotation_legend(self.model.annotation_colors)
        self._add_panel_object(self.result_panel)

        self._sync_panels()
        self._refresh_legend()
        self.sig_area.FitInside()
        self.sig_area.Layout()
        self._update_chunk_buttons()

    # =========================================================
    # Misc
    # =========================================================
    def remove_panel(self, panel):
        # Remove a non-core panel (e.g., extra annotation)
        for entry in list(self.signals):
            if entry['panel'] is panel:
                try:
                    self.sig_sizer.Detach(panel)
                    panel.Destroy()
                except Exception:
                    pass
                self.signals.remove(entry)
                break
        self.sig_area.FitInside()
        self.sig_area.Layout()
        self._refresh_legend()

    def _on_tick(self, evt):
        # place any playhead animation/sync logic here (optional)
        pass


# =========================================================
# App bootstrap
# =========================================================
def main():
    app = wx.App(False)
    MainFrame()
    app.MainLoop()

if __name__ == "__main__":
    main()
