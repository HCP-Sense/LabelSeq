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
        # derived "diff" annotation (Original - Annotator)
        self.derived_annotation: np.ndarray = self.original - self.annotator_seq
        # manual annotations stack (list of arrays)
        self.manual_annotations: typing.List[np.ndarray] = []
        # result is Original + (Derived + Sum(Manual))
        self.result: np.ndarray = self._compute_result(
            self.original,
            self.derived_annotation,
            self.manual_annotations
        )

        # ----- Panels -----
        self.original_panel: typing.Optional[OriginalPanel] = None
        self.annotation_panel: typing.Optional[AnnotationPanel] = None  # derived annotation visual
        self.annotator_panel: typing.Optional[AnnotatorPanel] = None
        self.result_panel: typing.Optional[ResultPanel] = None
        self.annotation_panels: typing.List[AnnotationPanel] = []       # manual annotations

        # ----- Model -----
        self.model = Model()
        self.auto_label_enabled = True
        self.model.set_original(self.original)
        self.model.set_annotator(self.annotator_seq)

        # ----- Modes (Toolbar) -----
        self.drag_mode = "preview"          # Preview | Realtime
        self.annotation_mode = "auto"       # Auto | Point | Region

        # ----- Build UI -----
        self._build_ui()
        self._init_toolbar()  # includes Drag Mode + Label Mode

        # ----- Build Panels -----
        self.update_panels()

        self.Centre()
        self.Show()

        # NOTE: Removed playhead timer/animation for performance

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

        for btn in [
            self.btn_undo, self.btn_redo, self.btn_save_proj, self.btn_load_proj,
            self.btn_export_ann, self.btn_export_res, self.btn_import,
            self.btn_prev, self.btn_next, self.btn_delete, self.btn_add_anno
        ]:
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

        # (Playhead label removed for performance)

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

    def _init_toolbar(self):
        toolbar = self.CreateToolBar()
        toolbar.AddSeparator()

        # Drag Mode
        drag_mode_label = wx.StaticText(toolbar, label="Drag Mode:")
        toolbar.AddControl(drag_mode_label)

        self.drag_mode_choice = wx.Choice(toolbar, choices=["preview", "realtime"])
        self.drag_mode_choice.SetStringSelection(self.drag_mode)
        self.drag_mode_choice.Bind(wx.EVT_CHOICE, self.on_drag_mode_changed)
        toolbar.AddControl(self.drag_mode_choice)

        toolbar.AddSeparator()

        # Label Mode (was "Annotation Mode")
        anno_mode_label = wx.StaticText(toolbar, label="Label Mode:")
        toolbar.AddControl(anno_mode_label)

        self.anno_mode_choice = wx.Choice(toolbar, choices=["auto", "point", "region"])
        self.anno_mode_choice.SetStringSelection(self.annotation_mode)
        self.anno_mode_choice.Bind(wx.EVT_CHOICE, self.on_annotation_mode_changed)
        toolbar.AddControl(self.anno_mode_choice)

        toolbar.Realize()

    # =========================================================
    # Helpers — Computation
    # =========================================================
    def _sum_manual_annotations(self) -> np.ndarray:
        if not self.manual_annotations:
            return np.zeros_like(self.original)
        # Align and sum
        acc = np.zeros_like(self.original, dtype=float)
        for arr in self.manual_annotations:
            if arr is None:
                continue
            a = np.asarray(arr, dtype=float).flatten()
            if a.size != acc.size:
                tmp = np.zeros_like(acc)
                m = min(tmp.size, a.size)
                tmp[:m] = a[:m]
                a = tmp
            acc += a
        return acc

    def _compute_result(
        self,
        original: np.ndarray,
        derived_ann: np.ndarray,
        manual_list: typing.List[np.ndarray]
    ) -> np.ndarray:
        manual_sum = self._sum_manual_annotations() if manual_list is not None else 0.0
        # Result = Original + (Derived + ManualSum)
        res = (
            np.asarray(original, dtype=float)
            + np.asarray(derived_ann, dtype=float)
            + np.asarray(manual_sum, dtype=float)
        )
        return res

    # =========================================================
    # Active Learning — confirmation & application
    # =========================================================
    def on_active_labels(self, indices: list, labels: list, source_panel):
        if not indices:
            wx.MessageBox(
                "No indices suggested.",
                "Active Learning",
                wx.OK | wx.ICON_INFORMATION
            )
            return

        counts = Counter(labels or [])
        summary_lines = [f"{lab if lab else '(unlabeled)'}: {cnt}" for lab, cnt in counts.items()]
        summary = "\n".join(summary_lines) if summary_lines else f"{len(indices)} points"

        msg = (
            f"Active Learning Suggestions\n\n"
            f"Total points: {len(indices)}\n"
            f"{summary}\n\n"
            f"Apply these suggestions to the TOPMOST manual Annotation layer? (value = 1.0)\n"
            f"(If there is no manual layer yet, I'll create one automatically.)"
        )
        dlg = wx.MessageDialog(self, msg, "Confirm Suggestions", wx.YES_NO | wx.ICON_QUESTION)
        res = dlg.ShowModal()
        dlg.Destroy()

        if res != wx.ID_YES:
            self.SetStatusText("Suggestions discarded.")
            return

        # pick target manual annotation panel; if none, create one
        if not self.annotation_panels:
            self._create_manual_annotation_layer()

        target_panel = self.annotation_panels[-1]  # topmost
        ann = np.asarray(target_panel.seq, dtype=float)

        try:
            for idx in indices:
                if 0 <= idx < ann.size:
                    ann[idx] = 1.0
        except Exception as e:
            wx.MessageBox(f"Failed to apply labels: {e}", "Error", wx.OK | wx.ICON_ERROR)
            return

        target_panel.seq = ann
        target_panel.Refresh()

        # recompute result
        self._update_arrays_from_panels()
        self._refresh_result_panel_fast()
        self._refresh_legend()
        self.SetStatusText(
            f"Applied {len(indices)} AL points to '{target_panel.title}' (value=1.0)."
        )

    # quick internal recompute from panels (derived + manual)
    def _update_arrays_from_panels(self):
        # annotator drives derived annotation
        if self.annotator_panel is not None:
            self.annotator_seq = np.asarray(self.annotator_panel.seq, dtype=float)
        self.derived_annotation = self.original - self.annotator_seq
        # manual list mirrors annotation_panels
        self.manual_annotations = [np.asarray(p.seq, dtype=float) for p in self.annotation_panels]
        self.result = self._compute_result(
            self.original,
            self.derived_annotation,
            self.manual_annotations
        )

    def _refresh_result_panel_fast(self):
        if self.annotation_panel:
            self.annotation_panel.seq = self.derived_annotation.copy()
            self.annotation_panel.n = len(self.annotation_panel.seq)
            self.annotation_panel.Refresh()

        if self.result_panel:
            self.result_panel.seq = self.result.copy()
            self.result_panel.n = len(self.result_panel.seq)
            try:
                self.result_panel.set_annotator_ref(self.annotator_panel)
            except Exception:
                self.result_panel.annotator_ref = self.annotator_panel
            self.result_panel.Refresh()

    # =========================================================
    # Button Handlers (undo/redo/save/load/export/import/chunks/annot add)
    # =========================================================
    def on_undo(self, evt):
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
        with wx.FileDialog(
            self, "Save Project",
            wildcard="JSON files (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dlg:
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
        with wx.FileDialog(
            self, "Load Project",
            wildcard="JSON files (*.json)|*.json",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as dlg:
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
        # export combined annotation = derived + sum(manual)
        combined = self.derived_annotation + self._sum_manual_annotations()
        with wx.FileDialog(
            self, "Export Annotation",
            wildcard="CSV (*.csv)|*.csv;|Parquet (*.parquet)|*.parquet;|NumPy (*.npy)|*.npy",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()

        try:
            ext = Path(path).suffix.lower()
            if ext == ".csv":
                save_array_csv(path, combined)
            elif ext == ".parquet":
                save_array_parquet(path, combined)
            elif ext == ".npy":
                save_array_npy(path, combined)
            else:
                save_array_csv(str(Path(path).with_suffix(".csv")), combined)
            self.SetStatusText(f"Combined Annotation exported: {path}")
        except Exception as e:
            wx.MessageBox(f"Export failed: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_export_result(self, evt):
        with wx.FileDialog(
            self, "Export Result",
            wildcard="CSV (*.csv)|*.csv;|Parquet (*.parquet)|*.parquet;|NumPy (*.npy)|*.npy",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()

        try:
            ext = Path(path).suffix.lower()
            if ext == ".csv":
                save_array_csv(path, self.result)
            elif ext == ".parquet":
                save_array_parquet(path, self.result)
            elif ext == ".npy":
                save_array_npy(path, self.result)
            else:
                save_array_csv(str(Path(path).with_suffix(".csv")), self.result)
            self.SetStatusText(f"Result exported: {path}")
        except Exception as e:
            wx.MessageBox(f"Export failed: {e}", "Error", wx.OK | wx.ICON_ERROR)

    # =========================================================
    # File loading / chunking
    # =========================================================
    def on_load_file(self, evt):
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
            # 1D (audio) — single virtual column
            self.available_columns = ["Audio"]
            self.column_choice.Append("Audio")
            self.column_choice.SetSelection(0)
            self.selected_column_index = 0

        # Detect audio-like -> default to region mode if Auto
        self._apply_selected_column_to_original()
        self._maybe_autoset_annotation_mode()
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

        if columns:
            self.column_choice.Clear()
            self.column_choice.AppendItems(self.available_columns)
            if (
                self.selected_column_index is not None
                and 0 <= self.selected_column_index < len(self.available_columns)
            ):
                self.column_choice.SetSelection(self.selected_column_index)
            else:
                self.column_choice.SetSelection(0)
                self.selected_column_index = 0

        self._apply_selected_column_to_original()
        self._maybe_autoset_annotation_mode()
        self._update_chunk_label()
        self.update_panels()
        self._update_chunk_buttons()
        self.SetStatusText(f"Chunk {self.current_chunk + 1}/{self.total_chunks} loaded.")

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
        self.derived_annotation = self.original - self.annotator_seq
        self.manual_annotations = []
        self.result = self._compute_result(
            self.original,
            self.derived_annotation,
            self.manual_annotations
        )
        self.model.set_original(self.original)
        self.model.set_annotator(self.annotator_seq)
        self.update_panels()
        self._update_chunk_label()
        self._update_chunk_buttons()
        self.SetStatusText("File cleared.")

    # ---- create a new manual annotation layer (helper) ----
    def _create_manual_annotation_layer(self):
        color_ann = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1

        seq = np.zeros_like(self.original, dtype=float)
        title = f"Annotation {len(self.annotation_panels) + 1}"

        p = AnnotationPanel(
            self.sig_area,
            seq,
            title,
            draggable=False,
            visible_count=(self.target_visible_count or 200),
            color=color_ann,
            role="annotation"
        )
        p.original_seq = self.original
        p.drag_mode = self.drag_mode

        # inherit view from result or original
        inherit = self.result_panel or self.original_panel
        if inherit is not None:
            p.zoom_factor = float(getattr(inherit, "zoom_factor", 1.0))
            p.pan_offset = float(getattr(inherit, "pan_offset", 0.0))
            p.y_offset = float(getattr(inherit, "y_offset", 0.0))

        # Wrap: title edit -> refresh legend
        if hasattr(p, "edit_title"):
            orig_edit = p.edit_title

            def _wrap(evt):
                old = getattr(p, "title", title)
                orig_edit(evt)
                new = getattr(p, "title", old)
                if new != old:
                    self._refresh_legend()

            p.edit_title = _wrap

        # Wrap: left-up to detect new spans and prompt label in REGION mode
        if hasattr(p, "on_left_up"):
            _orig_left_up = p.on_left_up

            def _wrapped_left_up(evt):
                if getattr(p, "frozen", False):
                    # If frozen, just ignore any editing
                    try:
                        evt.Skip()
                    except Exception:
                        pass
                    return
                before = len(getattr(p, "spans", []))
                _orig_left_up(evt)
                after = len(getattr(p, "spans", []))
                # If a new span was added and we're in region mode -> prompt label
                if after > before and self._effective_annotation_mode() == "region":
                    try:
                        s = p.spans[-1]
                    except Exception:
                        return
                    dlg = wx.TextEntryDialog(
                        self,
                        "Enter label for selected region:",
                        "Annotation Label",
                        s.get("label", p.title)
                    )
                    if dlg.ShowModal() == wx.ID_OK:
                        lbl = dlg.GetValue().strip() or p.title
                        s["label"] = lbl
                    else:
                        # cancel => remove the just-added span
                        try:
                            p.spans.pop(-1)
                        except Exception:
                            pass
                    dlg.Destroy()
                    p.Refresh(False)

            p.on_left_up = _wrapped_left_up

        # add to UI just above the result panel (stack below previous annotations)
        self._add_panel_object(p, insert_before_result=True)
        self.annotation_panels.append(p)
        self.manual_annotations = [pp.seq for pp in self.annotation_panels]

    def on_add_annotation(self, evt):
        self._create_manual_annotation_layer()
        # resync panels and layout
        self._sync_panels()
        self.sig_area.FitInside()
        self.sig_area.Layout()
        self._refresh_legend()
        self.SetStatusText("Added manual annotation panel.")

    def on_column_choice(self, evt):
        sel = self.column_choice.GetSelection()
        if sel == wx.NOT_FOUND:
            return
        self.selected_column_index = int(sel)

        if _set_selected_columns is not None and self.available_columns:
            try:
                _set_selected_columns([self.available_columns[sel]])
            except Exception:
                pass

        self._apply_selected_column_to_original()
        self._maybe_autoset_annotation_mode()
        self.update_panels()
        self.SetStatusText(
            f"Column selected: {self.available_columns[sel] if self.available_columns else 'Audio'}"
        )

    # =========================================================
    # Helpers (legend, panels, chunk UI)
    # =========================================================
    def _normalize_seq(self, seq) -> np.ndarray:
        if seq is None:
            return np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        if isinstance(seq, pd.DataFrame):
            if seq.shape[1] > 0:
                ser = pd.to_numeric(seq.iloc[:, 0], errors="coerce").fillna(0.0)
                arr = ser.to_numpy(dtype=float)
            else:
                arr = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        elif isinstance(seq, pd.Series):
            arr = pd.to_numeric(seq, errors="coerce").fillna(0.0).to_numpy(dtype=float)
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
                ser = pd.to_numeric(data.iloc[:, 0], errors="coerce").fillna(0.0)
            arr = ser.to_numpy(dtype=float)
        elif isinstance(data, (np.ndarray, list, tuple)):
            arr = np.asarray(data, dtype=float)
        else:
            try:
                arr = np.asarray(data, dtype=float).flatten()
            except Exception:
                arr = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)

        self.original = arr.flatten()
        if self.original.size < 2:
            self.original = np.resize(self.original, max(2, self.DEFAULT_PANEL_LENGTH))

        # re-align dependent arrays
        self.annotator_seq = np.copy(self.original)
        self.derived_annotation = self.original - self.annotator_seq
        # clear manual layers because their length changed
        self.manual_annotations = []
        self.annotation_panels = []
        self.result = self._compute_result(
            self.original,
            self.derived_annotation,
            self.manual_annotations
        )

        self.model.set_original(self.original)
        self.model.set_annotator(self.annotator_seq)

    def _clear_existing_panels(self):
        for entry in self.signals[:]:
            p = entry["panel"]
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
        self.annotation_panel = None   # derived
        self.annotator_panel = None
        self.result_panel = None
        self.annotation_panels = []

    def _add_panel_object(self, sp: InteractiveSequencePanel, insert_before_result: bool = False):
        sp.SetMinSize((-1, 250))
        inserted = False

        if isinstance(sp, AnnotationPanel) and sp is not self.annotation_panel:
            # Insert manual annotation directly under the last existing manual AnnotationPanel
            last_anno_idx = -1
            for i in range(self.sig_sizer.GetItemCount()):
                win = self.sig_sizer.GetItem(i).GetWindow()
                if isinstance(win, AnnotationPanel) and win is not self.annotation_panel:
                    last_anno_idx = i
            if last_anno_idx >= 0:
                self.sig_sizer.Insert(last_anno_idx + 1, sp, 0, wx.EXPAND | wx.ALL, 5)
                inserted = True

        if not inserted:
            if self.result_panel and insert_before_result:
                try:
                    idx = next(
                        i
                        for i in range(self.sig_sizer.GetItemCount())
                        if self.sig_sizer.GetItem(i).GetWindow() is self.result_panel
                    )
                    self.sig_sizer.Insert(idx, sp, 0, wx.EXPAND | wx.ALL, 5)
                except StopIteration:
                    self.sig_sizer.Add(sp, 0, wx.EXPAND | wx.ALL, 5)
            else:
                self.sig_sizer.Add(sp, 0, wx.EXPAND | wx.ALL, 5)

        self.signals.append({"panel": sp, "formula": getattr(sp, "formula", None)})

    def _sync_panels(self):
        all_panels = [entry["panel"] for entry in self.signals]
        for entry in self.signals:
            p = entry["panel"]
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
                # also propagate to panels
                for entry in self.signals:
                    p = entry["panel"]
                    if getattr(p, "title", "") == label:
                        p.color = col
                        p.Refresh()
                # update annotation legend for result panel
                if self.result_panel and hasattr(self.result_panel, "set_annotation_legend"):
                    cur = self.result_panel.get_annotation_legend()
                    if label in cur:
                        cur[label] = (col.Red(), col.Green(), col.Blue())
                    self.result_panel.set_annotation_legend(cur)
            dlg.Destroy()

        row.Bind(wx.EVT_LEFT_DOWN, on_click)
        sw.Bind(wx.EVT_LEFT_DOWN, on_click)
        lbl.Bind(wx.EVT_LEFT_DOWN, on_click)
        self.legend_sizer.Add(row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

    def _collect_annotation_legend(self) -> dict:
        legend = {}
        for p in self.annotation_panels:
            legend[getattr(p, "title", "Annotation")] = getattr(p, "color", wx.BLACK)
        return legend

    def _refresh_legend(self):
        self._clear_legend_entries()
        seen = set()

        # Non-annotation panels: Original, Annotator, Result, Derived Annotation
        for entry in self.signals:
            p = entry["panel"]
            role = getattr(p, "role", "")
            if role == "annotation" and p is not self.annotation_panel:
                continue
            title = getattr(p, "title", getattr(p, "label", ""))
            color = getattr(p, "color", wx.BLACK)
            if title not in seen:
                self._add_legend_row(title, color)
                seen.add(title)

        # Manual Annotations block
        ann_map = self._collect_annotation_legend()
        if ann_map:
            heading = wx.StaticText(self.legend_panel, label="Manual Annotations")
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

        # Also push annotation legend into Result (optional)
        if self.result_panel and hasattr(self.result_panel, "set_annotation_legend"):
            ann_tuple_map = {}
            for name, col in ann_map.items():
                c = col if isinstance(col, wx.Colour) else wx.Colour(col)
                ann_tuple_map[name] = (c.Red(), c.Green(), c.Blue())
            self.result_panel.set_annotation_legend(ann_tuple_map)

        self.legend_panel.Layout()

    def _update_chunk_label(self):
        if self.total_chunks > 0:
            self.chunk_info_label.SetLabel(
                f"Chunk {self.current_chunk + 1}/{self.total_chunks}"
            )
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

        # normalize and build base arrays
        orig = self._normalize_seq(self.original)
        anntr = (
            self._normalize_seq(self.annotator_seq)
            if getattr(self.annotator_seq, "size", 0) >= 2
            else orig.copy()
        )
        self.original = orig
        self.annotator_seq = anntr
        self.derived_annotation = self.original - self.annotator_seq
        self.manual_annotations = []
        self.result = self._compute_result(
            self.original,
            self.derived_annotation,
            self.manual_annotations
        )

        # --- Original panel ---
        color_orig = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1
        self.original_panel = OriginalPanel(
            self.sig_area,
            self.original,
            "Original",
            draggable=False,
            visible_count=(self.target_visible_count or 200),
            color=color_orig,
            role="original",
        )
        self.original_panel.drag_mode = self.drag_mode
        self.original_panel.original_seq = self.original
        self._add_panel_object(self.original_panel)

        # --- Derived Annotation panel (visualizes Original - Annotator) ---
        color_der = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1
        self.annotation_panel = AnnotationPanel(
            self.sig_area,
            self.derived_annotation,
            "Derived Annotation",
            draggable=False,
            visible_count=(self.target_visible_count or 200),
            color=color_der,
            role="annotation",
        )
        self.annotation_panel.drag_mode = self.drag_mode
        self.annotation_panel.original_seq = self.original
        self._add_panel_object(self.annotation_panel)

        # --- Annotator panel ---
        color_annot = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1

        def annotator_updated_callback(new_annot_np: np.ndarray):
            # realtime or preview-on-release will call this
            new_annot = np.asarray(new_annot_np, dtype=float).flatten()
            if new_annot.size != self.original.size:
                tmp = np.zeros_like(self.original)
                m = min(len(new_annot), len(tmp))
                tmp[:m] = new_annot[:m]
                new_annot = tmp

            self.annotator_seq = new_annot
            self.derived_annotation = self.original - self.annotator_seq
            self.result = self._compute_result(
                self.original,
                self.derived_annotation,
                self.manual_annotations
            )

            # update derived annotation panel
            self.annotation_panel.seq = self.derived_annotation
            self.annotation_panel.n = len(self.derived_annotation)
            self.annotation_panel.Refresh()

            # update result
            self.result_panel.seq = self.result
            self.result_panel.n = len(self.result)
            try:
                self.result_panel.set_annotator_ref(self.annotator_panel)
            except Exception:
                self.result_panel.annotator_ref = self.annotator_panel
            self.result_panel.Refresh()

            self._refresh_legend()

        self.annotator_panel = AnnotatorPanel(
            self.sig_area,
            self.annotator_seq,
            "Annotator",
            draggable=True,
            visible_count=(self.target_visible_count or 200),
            color=color_annot,
            role="annotator",
            on_update=annotator_updated_callback,
        )
        self.annotator_panel.drag_mode = self.drag_mode
        self.annotator_panel.original_seq = self.original
        self._add_panel_object(self.annotator_panel)

        # --- Result panel ---
        color_res = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1
        self.result_panel = ResultPanel(
            self.sig_area,
            self.result,
            "Result",
            draggable=False,
            visible_count=(self.target_visible_count or 200),
            color=color_res,
            role="result",
        )
        self.result_panel.original_seq = self.original
        try:
            self.result_panel.set_annotator_ref(self.annotator_panel)
        except Exception:
            self.result_panel.annotator_ref = self.annotator_panel
        if hasattr(self.result_panel, "set_annotation_legend"):
            self.result_panel.set_annotation_legend({})
        self._add_panel_object(self.result_panel)

        # --- Allow annotator region highlight -> label -> write to manual annotation layer ---
        if hasattr(self.annotator_panel, "on_left_up"):
            orig_left_up = self.annotator_panel.on_left_up

            def _annot_left_up(evt):
                # Let the panel finalize its region vars first
                prev_spans = list(getattr(self.annotator_panel, "spans", []))
                orig_left_up(evt)

                # If a new span appeared, ask for label and attach it
                new_spans = getattr(self.annotator_panel, "spans", [])
                if (
                    len(new_spans) > len(prev_spans)
                    and self._effective_annotation_mode() == "region"
                ):
                    s = new_spans[-1]
                    # Prompt for a label
                    dlg = wx.TextEntryDialog(
                        self,
                        "Enter label for highlighted region:",
                        "Region Label",
                        self.annotator_panel.title,
                    )
                    if dlg.ShowModal() == wx.ID_OK:
                        s["label"] = dlg.GetValue() or self.annotator_panel.title
                    dlg.Destroy()

                    # Also create (or reuse) a topmost manual annotation layer and set that region to 1.0
                    if not self.annotation_panels:
                        self._create_manual_annotation_layer()
                    target = self.annotation_panels[-1]
                    a = int(s.get("start", 0))
                    b = int(s.get("end", a))
                    arr = np.asarray(target.seq, dtype=float)
                    a = max(0, min(a, arr.size - 1))
                    b = max(0, min(b, arr.size - 1))
                    if a <= b:
                        arr[a : b + 1] = 1.0
                        target.seq = arr
                        target.Refresh(False)
                        self._update_arrays_from_panels()
                        self._refresh_result_panel_fast()
                        self._refresh_legend()
                        self.SetStatusText(
                            f"Labeled region {a}:{b} as '{s.get('label', '')}'."
                        )

            self.annotator_panel.on_left_up = _annot_left_up

        self._sync_panels()
        self._refresh_legend()
        self.sig_area.FitInside()
        self.sig_area.Layout()
        self._update_chunk_buttons()

    # =========================================================
    # Modes (toolbar) handlers
    # =========================================================
    def on_drag_mode_changed(self, event):
        self.drag_mode = self.drag_mode_choice.GetStringSelection()
        print(f"[DEBUG] Drag mode changed to: {self.drag_mode}")
        for panel in [
            self.original_panel,
            self.annotator_panel,
            self.annotation_panel,
            self.result_panel,
            *self.annotation_panels,
        ]:
            try:
                if panel is not None:
                    panel.drag_mode = self.drag_mode
            except Exception:
                pass

    def on_annotation_mode_changed(self, event):
        self.annotation_mode = self.anno_mode_choice.GetStringSelection()
        print(f"[DEBUG] Annotation mode changed to: {self.annotation_mode}")

        tip = {
            "auto": "Auto: If audio-like, use Region mode; else Point mode.",
            "point": "Point: Drag points in Annotator; spans need Shift on annotation panels.",
            "region": "Region: Create spans on annotator/annotation; you'll be prompted for a label.",
        }[self.annotation_mode]
        self.SetStatusText(tip)

        # propagate to all panels immediately
        effective = self._effective_annotation_mode()
        for p in [
            self.original_panel,
            self.annotator_panel,
            self.annotation_panel,
            self.result_panel,
            *self.annotation_panels,
        ]:
            if not p:
                continue
            try:
                p.annotation_mode = effective
                # sync the per-panel dropdown if present
                if hasattr(p, "mode_choice") and p.mode_choice:
                    try:
                        p.mode_choice.SetStringSelection(effective)
                    except Exception:
                        pass
                # clear any ongoing drags if we just switched away from point
                if effective == "region":
                    p.dragging = False
                    p.selected_idx = None
                p.Refresh(False)
            except Exception:
                pass

    def _effective_annotation_mode(self) -> str:
        return self.annotation_mode

    def _maybe_autoset_annotation_mode(self):
        """If in 'auto', pick region for 1D audio-like data, else point."""
        if self.annotation_mode != "auto":
            return
        # Simple heuristic: if data has no columns (Audio) or a single long 1D series -> region
        is_audio_like = False
        try:
            if self.available_columns == ["Audio"]:
                is_audio_like = True
            elif isinstance(self.current_file_content, (np.ndarray, list, tuple)):
                arr = np.asarray(self.current_file_content)
                is_audio_like = (arr.ndim == 1 and arr.size >= 1024)
            elif isinstance(self.current_file_content, pd.DataFrame):
                is_audio_like = (
                    self.current_file_content.shape[1] == 1
                    and len(self.current_file_content) >= 1024
                )
        except Exception:
            pass

        chosen = "region" if is_audio_like else "point"
        self.annotation_mode = chosen
        try:
            self.anno_mode_choice.SetStringSelection(self.annotation_mode)
        except Exception:
            pass
        self.SetStatusText(f"Auto annotation mode -> {self.annotation_mode}")


# =========================================================
# App bootstrap
# =========================================================
def main():
    app = wx.App(False)
    MainFrame()
    app.MainLoop()


if __name__ == "__main__":
    main()
