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
from LoadFile import load_file, set_selected_columns
from model import Model
from save_manager import save_array_csv, save_array_parquet, save_array_npy
from project_manager import save_project, load_project

# Third-Party Imports
import wx
import numpy as np
import pandas as pd
import typing

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
        self.current_file_content = None
        self.available_columns: typing.List[str] = []
        self.selected_column_index: typing.Optional[int] = None
        self.current_chunk: int = 0
        self.total_chunks: int = 0

        # ----- Core Array Data -----
        self.original: np.ndarray = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        self.annotator_seq: np.ndarray = self.original.copy()
        self.annotation_seq: np.ndarray = self.original - self.annotator_seq
        self.result: np.ndarray = self.original + self.annotation_seq

        # ----- Panels -----
        self.original_panel = None
        self.annotation_panel = None
        self.annotator_panel = None
        self.result_panel = None

        # ----- Model Initialized BEFORE Panels -----
        self.model = Model()
        self.model.set_original(self.original)
        self.model.set_annotator(self.annotator_seq)

        # ----- Build UI -----
        self._build_ui()
        self.update_panels()
        self.Centre()
        self.Show()

        # ----- Timer for Updating Playhead -----
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
        font.MakeBold()
        legend_label.SetFont(font)
        self.legend_sizer.Add(legend_label, 0, wx.ALL, 8)
        self.legend_panel.SetSizer(self.legend_sizer)

        # ----------------- Right Panel (Controls + Data Panels) -----------------
        right_panel = wx.Panel(main_panel)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        # ----- Buttons Row -----
        btn_row = wx.BoxSizer(wx.HORIZONTAL)

        btn_undo = wx.Button(right_panel, label="Undo")
        btn_redo = wx.Button(right_panel, label="Redo")
        btn_save_proj = wx.Button(right_panel, label="Save Project")
        btn_load_proj = wx.Button(right_panel, label="Load Project")
        btn_export_ann = wx.Button(right_panel, label="Export Annotation")
        btn_export_res = wx.Button(right_panel, label="Export Result")
        btn_import = wx.Button(right_panel, label="Import Data")
        btn_prev = wx.Button(right_panel, label="<< Previous Chunk")
        btn_next = wx.Button(right_panel, label="Next Chunk >>")
        btn_delete = wx.Button(right_panel, label="Delete File")
        btn_add_anno = wx.Button(right_panel, label="Add Annotation")

        btn_undo.Bind(wx.EVT_BUTTON, self.on_undo)
        btn_redo.Bind(wx.EVT_BUTTON, self.on_redo)
        btn_save_proj.Bind(wx.EVT_BUTTON, self.on_save_project)
        btn_load_proj.Bind(wx.EVT_BUTTON, self.on_load_project)
        btn_export_ann.Bind(wx.EVT_BUTTON, self.on_export_annotation)
        btn_export_res.Bind(wx.EVT_BUTTON, self.on_export_result)
        btn_import.Bind(wx.EVT_BUTTON, self.on_load_file)
        btn_prev.Bind(wx.EVT_BUTTON, self.on_prev_chunk)
        btn_next.Bind(wx.EVT_BUTTON, self.on_next_chunk)
        btn_delete.Bind(wx.EVT_BUTTON, self.on_delete_file)
        btn_add_anno.Bind(wx.EVT_BUTTON, self.on_add_annotation)

        btn_prev.Disable()
        btn_next.Disable()

        for btn in [btn_undo, btn_redo, btn_save_proj, btn_load_proj, btn_export_ann,
                    btn_export_res, btn_import, btn_prev, btn_next, btn_delete, btn_add_anno]:
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
    # Sequence normalization
    # =========================================================
    def _normalize_seq(self, seq) -> np.ndarray:
        if seq is None:
            return np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        if isinstance(seq, pd.DataFrame):
            if seq.shape[1] > 0:
                arr = seq.iloc[:, 0].dropna().to_numpy(dtype=float)
            else:
                arr = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        elif isinstance(seq, pd.Series):
            arr = seq.dropna().to_numpy(dtype=float)
        else:
            arr = np.asarray(seq, dtype=float)
        arr = arr.flatten()
        if arr.size < 2:
            arr = np.resize(arr, max(2, self.DEFAULT_PANEL_LENGTH))
        return arr

    # =========================================================
    # Panel helpers
    # =========================================================
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

    # =========================================================
    # Legend helpers
    # =========================================================
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

        def on_click(evt):
            data = wx.ColourData()
            data.SetChooseFull(True)
            data.SetColour(sw.GetBackgroundColour())
            dlg = wx.ColourDialog(self, data)
            if dlg.ShowModal() == wx.ID_OK:
                col = dlg.GetColourData().GetColour()
                sw.SetBackgroundColour(col)
                sw.Refresh()
                # update model color map if this is an annotation layer
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
        # Original/visible panels first
        for entry in self.signals:
            p = entry['panel']
            title = getattr(p, 'title', getattr(p, 'label', ''))
            color = getattr(p, 'color', wx.BLACK)
            if title not in seen:
                self._add_legend_row(title, color)
                seen.add(title)
        # Annotation-specific legend from model → ask ResultPanel for mapping
        if self.result_panel and hasattr(self.result_panel, "get_annotation_legend"):
            ann_map = self.result_panel.get_annotation_legend()
            if ann_map:
                heading = wx.StaticText(self.legend_panel, label="Annotations")
                f = heading.GetFont()
                f.MakeBold()
                heading.SetFont(f)
                self.legend_sizer.Add(heading, 0, wx.LEFT | wx.TOP, 8)
                for name, col in ann_map.items():
                    c = col if isinstance(col, wx.Colour) else wx.Colour(col)
                    self._add_legend_row(name, c)
        self.legend_panel.Layout()

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
            self.sig_area, self.original, "Original", draggable=False, visible_count=200,
            color=color_orig, role="original"
        )
        self.original_panel.original_seq = self.original
        self._add_panel_object(self.original_panel)

        # --- Annotation panel ---
        color_ann = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1
        self.annotation_panel = AnnotationPanel(
            self.sig_area, self.annotation_seq, "Annotation", draggable=False, visible_count=200,
            color=color_ann, role="annotation"
        )
        self.annotation_panel.original_seq = self.original
        self._add_panel_object(self.annotation_panel)

        # Hook title edit to rename annotation in model
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
            draggable=True, visible_count=200, color=color_annot,
            role="annotator", on_update=annotator_updated_callback
        )
        self.annotator_panel.original_seq = self.original
        self._add_panel_object(self.annotator_panel)

        # --- Result panel ---
        color_res = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1
        self.result_panel = ResultPanel(
            self.sig_area, self.result, "Result",
            draggable=False, visible_count=200, color=color_res, role="result"
        )
        self.result_panel.original_seq = self.original
        try:
            self.result_panel.set_annotator_ref(self.annotator_panel)
        except Exception:
            self.result_panel.annotator_ref = self.annotator_panel
        if hasattr(self.result_panel, 'set_annotation_legend'):
            self.result_panel.set_annotation_legend(self.model.annotation_colors)
        self._add_panel_object(self.result_panel)

        # Sync all, refresh legend
        self._sync_panels()
        self._refresh_legend()
        self.sig_area.FitInside()
        self.sig_area.Layout()
        self._update_chunk_buttons()

        # Ensure prev/next button handles exist (in case UI changed)
        # This safely binds them to attributes if not set earlier.
        try:
            if not hasattr(self, "btn_prev") or not hasattr(self, "btn_next"):
                # Try to find by label
                for child in self.FindWindowById(self.GetId()).GetChildren():
                    pass  # placeholder (avoid errors)
        except Exception:
            pass

    # =========================================================
    # File & chunk handling
    # =========================================================
    def on_load_file(self, event):
        content, columns, chunk_idx, total = load_file(new_file=True, direction="current")
        if content is None:
            wx.MessageBox("No file selected or failed to load.", "Info", wx.OK | wx.ICON_INFORMATION)
            return
        self.current_file_content = content
        self.available_columns = columns or []
        self.current_chunk = chunk_idx
        self.total_chunks = total
        if self.available_columns:
            self.selected_column_index = 0
            set_selected_columns([self.available_columns[0]])
        else:
            self.selected_column_index = None
            set_selected_columns([])
        self._update_column_choice()
        self._prepare_and_show_data()
        self._update_chunk_buttons()

    def on_next_chunk(self, event):
        content, columns, chunk_idx, total = load_file(new_file=False, direction="next")
        if content is None:
            wx.MessageBox("Failed to load next chunk.", "Error", wx.OK | wx.ICON_ERROR)
            return
        self.current_file_content = content
        self.available_columns = columns or self.available_columns
        self.current_chunk = chunk_idx
        self.total_chunks = total
        if self.selected_column_index is None and self.available_columns:
            self.selected_column_index = 0
            set_selected_columns([self.available_columns[0]])
        self._update_column_choice()
        self._prepare_and_show_data()
        self._update_chunk_buttons()

    def on_prev_chunk(self, event):
        content, columns, chunk_idx, total = load_file(new_file=False, direction="prev")
        if content is None:
            wx.MessageBox("Failed to load previous chunk.", "Error", wx.OK | wx.ICON_ERROR)
            return
        self.current_file_content = content
        self.available_columns = columns or self.available_columns
        self.current_chunk = chunk_idx
        self.total_chunks = total
        if self.selected_column_index is None and self.available_columns:
            self.selected_column_index = 0
            set_selected_columns([self.available_columns[0]])
        self._update_column_choice()
        self._prepare_and_show_data()
        self._update_chunk_buttons()

    def on_delete_file(self, event):
        self.current_file_content = None
        self.available_columns = []
        self.selected_column_index = None
        self.current_chunk = 0
        self.total_chunks = 0
        set_selected_columns([])
        self.column_choice.Clear()
        self.chunk_info_label.SetLabel("")
        self.SetStatusText("File removed")
        self._clear_existing_panels()
        self.original = np.zeros(self.DEFAULT_PANEL_LENGTH)
        self.annotator_seq = self.original.copy()
        # update model
        self.model.set_original(self.original)
        self.model.set_annotator(self.annotator_seq)
        self.annotation_seq = self.original - self.annotator_seq
        self.result = self.original + self.annotation_seq
        self.update_panels()

    # =========================================================
    # Column helpers
    # =========================================================
    def _update_column_choice(self):
        if not self.available_columns:
            self.column_choice.Clear()
            self.chunk_info_label.SetLabel("")
            return
        self.column_choice.Clear()
        self.column_choice.SetItems(self.available_columns)
        if self.selected_column_index is None:
            self.selected_column_index = 0
        if 0 <= self.selected_column_index < len(self.available_columns):
            self.column_choice.SetSelection(self.selected_column_index)
        if self.total_chunks and self.total_chunks > 1:
            self.chunk_info_label.SetLabel(f"Chunk {self.current_chunk+1}/{self.total_chunks}")
            self.SetStatusText(f"Viewing chunk {self.current_chunk+1} of {self.total_chunks}")
        else:
            self.chunk_info_label.SetLabel("Loaded")
            self.SetStatusText("Loaded")

    def on_column_choice(self, event):
        sel = self.column_choice.GetSelection()
        if sel == wx.NOT_FOUND:
            return
        self.selected_column_index = int(sel)
        chosen_name = self.available_columns[self.selected_column_index]
        set_selected_columns([chosen_name])
        content, columns, chunk_idx, total = load_file(new_file=False, direction="current")
        if content is None:
            wx.MessageBox("Failed to reload with selected column.", "Error", wx.OK | wx.ICON_ERROR)
            return
        self.current_file_content = content
        self.available_columns = columns or self.available_columns
        self.current_chunk = chunk_idx
        self.total_chunks = total
        self._prepare_and_show_data()
        self._update_column_choice()

    def _prepare_and_show_data(self):
        c = self.current_file_content
        if c is None:
            self.original = np.zeros(self.DEFAULT_PANEL_LENGTH)
        elif isinstance(c, (np.ndarray, pd.Series)):
            self.original = np.asarray(c, dtype=float).flatten()
        elif isinstance(c, pd.DataFrame):
            idx = self.selected_column_index or 0
            idx = min(idx, c.shape[1]-1)
            self.original = c.iloc[:, idx].dropna().to_numpy(dtype=float).flatten()
        else:
            try:
                self.original = np.asarray(c, dtype=float).flatten()
            except Exception:
                self.original = np.zeros(self.DEFAULT_PANEL_LENGTH)

        if self.original.size < 2:
            self.original = np.resize(self.original, max(2, self.DEFAULT_PANEL_LENGTH))

        self.annotator_seq = self.original.copy()
        # update model
        self.model.set_original(self.original)
        self.model.set_annotator(self.annotator_seq)
        self.annotation_seq = self.original - self.annotator_seq
        self.result = self.original + self.annotation_seq
        self.update_panels()

    def _update_chunk_buttons(self):
        # If we don't have direct refs, just skip enabling/disabling
        btn_prev = getattr(self, "btn_prev", None)
        btn_next = getattr(self, "btn_next", None)
        if not btn_prev or not btn_next:
            return
        if self.total_chunks and self.total_chunks > 1:
            btn_prev.Enable(self.current_chunk > 0)
            btn_next.Enable(self.current_chunk < (self.total_chunks - 1))
        else:
            btn_prev.Disable()
            btn_next.Disable()

    # =========================================================
    # Add Annotation Panel
    # =========================================================
    def on_add_annotation(self, event):
        dlg = wx.TextEntryDialog(self, "Label for annotation:", "Add Annotation", "")
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy()
            return
        label = dlg.GetValue()
        dlg.Destroy()

        seq = np.zeros_like(self.original)
        color = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1
        sp = AnnotationPanel(self.sig_area, seq, label, draggable=True, visible_count=200, color=color, role="annotation")
        sp.original_seq = self.original
        self._add_panel_object(sp, insert_before_result=True)

        def live_update_callback(new_seq):
            sp.seq = new_seq
            sp.Refresh()
            self._refresh_legend()

        sp.on_update = live_update_callback

        self._sync_panels()
        self._refresh_legend()
        self.sig_area.FitInside()
        self.sig_area.Layout()
        sp.Refresh()

    # =========================================================
    # Export Handlers
    # =========================================================
    def on_export_annotation(self, evt):
        with wx.FileDialog(self, "Save annotation", wildcard="CSV (*.csv)|*.csv|Parquet (*.parquet)|*.parquet|NumPy (*.npy)|*.npy",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
            if path.endswith(".csv"):
                save_array_csv(path, self.model.annotation)
            elif path.endswith(".parquet"):
                save_array_parquet(path, self.model.annotation)
            else:
                save_array_npy(path, self.model.annotation)

    def on_export_result(self, evt):
        with wx.FileDialog(self, "Save result", wildcard="CSV (*.csv)|*.csv|Parquet (*.parquet)|*.parquet|NumPy (*.npy)|*.npy",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
            self.model.recompute()
            if path.endswith(".csv"):
                save_array_csv(path, self.model.result)
            elif path.endswith(".parquet"):
                save_array_parquet(path, self.model.result)
            else:
                save_array_npy(path, self.model.result)

    # =========================================================
    # Undo/Redo + playhead
    # =========================================================
    def _focused_panel(self):
        # Prefer the panel with focus; fallback to annotator
        for entry in self.signals:
            p = entry['panel']
            try:
                if p.HasFocus():
                    return p
            except Exception:
                pass
        return self.annotator_panel or (self.signals[0]['panel'] if self.signals else None)

    def on_undo(self, evt):
        p = self._focused_panel()
        if p and hasattr(p, "_undo"):
            p._undo()

    def on_redo(self, evt):
        p = self._focused_panel()
        if p and hasattr(p, "_redo"):
            p._redo()

    def _on_tick(self, evt):
        p = self._focused_panel()
        idx = getattr(p, "play_idx", None) if p else None
        if idx is None:
            self.playhead_label.SetLabel("Playhead: -")
        else:
            self.playhead_label.SetLabel(f"Playhead: {idx}")

    # =========================================================
    # Project Save / Load
    # =========================================================
    def on_save_project(self, evt):
        with wx.FileDialog(self, "Save Project", wildcard="Project (*.json)|*.json",
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
            save_project(path, self.model, panels)
            self.SetStatusText(f"Project saved: {path}")

    def on_load_project(self, evt):
        with wx.FileDialog(self, "Load Project", wildcard="Project (*.json)|*.json",
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
            if load_project(path, self.model, panels):
                if hasattr(self.result_panel, 'set_annotation_legend'):
                    self.result_panel.set_annotation_legend(self.model.annotation_colors)
                self._refresh_legend()
                self.SetStatusText(f"Project loaded: {path}")


def main():
    app = wx.App(False)
    MainFrame()
    app.MainLoop()


if __name__ == "__main__":
    main()
