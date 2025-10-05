"""main.py: Entry point; builds main window, manages multiple panels, links legends and updates with chunk-navigation and column dropdown."""

__author___ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Moye Nyuysoni Glein Perry"

# Project Imports
from InteractiveSequencePanel import InteractiveSequencePanel
from LoadFile import load_file, set_selected_columns

# Third-Party Imports
import wx
import numpy as np
import pandas as pd
import math
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
    """
    Main application window controlling:
      - file chunk loading (next/prev)
      - column dropdown selection
      - panel creation / deletion / synchronization
      - legend display
      - status bar (chunk X of Y)
    """

    DEFAULT_PANEL_LENGTH = 100  # fallback length when no file loaded

    def __init__(self):
        super().__init__(None, title="LabelSeq (GC-only)", size=(1320, 880))

        # Application state
        self.color_index: int = 0
        self.signals: typing.List[dict] = []
        self.current_file_content = None     # raw chunk returned from load_file (DataFrame or ndarray or Series)
        self.available_columns: typing.List[str] = []
        self.selected_column_index: typing.Optional[int] = None
        self.current_chunk: int = 0
        self.total_chunks: int = 0
        self.original: np.ndarray = np.array([])  # 1D numpy array used to populate the Original panel
        self.result: np.ndarray = np.array([])    # 1D numpy array copy of original (editable)
        self.result_panel = None                  # reference to draggable result panel

        # Build UI
        self._build_ui()

        # Add default placeholder panels
        self.update_panels()  # builds Original, Signal 1, Result placeholders

        # Final frame show
        self.Centre()
        self.Show()

    # ----------------------
    # UI Construction
    # ----------------------
    def _build_ui(self):
        """Create the main layout, buttons, column dropdown, signal area and status bar."""
        main_panel = wx.Panel(self)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Left: Legend
        self.legend_panel = wx.Panel(main_panel, size=(200, -1))
        self.legend_panel.SetBackgroundColour(wx.Colour(245, 245, 245))
        self.legend_sizer = wx.BoxSizer(wx.VERTICAL)
        legend_label = wx.StaticText(self.legend_panel, label="Legend")
        font = legend_label.GetFont()
        font.MakeBold()
        legend_label.SetFont(font)
        self.legend_sizer.Add(legend_label, 0, wx.ALL, 8)
        self.legend_panel.SetSizer(self.legend_sizer)

        # Right: Controls + panels
        right_panel = wx.Panel(main_panel)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        # Top control row (buttons)
        btn_row = wx.BoxSizer(wx.HORIZONTAL)

        self.btn_import = wx.Button(right_panel, label="Import Data")
        self.btn_import.Bind(wx.EVT_BUTTON, self.on_load_file)
        btn_row.Add(self.btn_import, 0, wx.ALL, 6)

        self.btn_prev = wx.Button(right_panel, label="<< Previous Chunk")
        self.btn_prev.Bind(wx.EVT_BUTTON, self.on_prev_chunk)
        self.btn_prev.Disable()
        btn_row.Add(self.btn_prev, 0, wx.ALL, 6)

        self.btn_next = wx.Button(right_panel, label="Next Chunk >>")
        self.btn_next.Bind(wx.EVT_BUTTON, self.on_next_chunk)
        self.btn_next.Disable()
        btn_row.Add(self.btn_next, 0, wx.ALL, 6)

        self.btn_remove_file = wx.Button(right_panel, label="Remove File")
        self.btn_remove_file.Bind(wx.EVT_BUTTON, self.on_remove_file)
        btn_row.Add(self.btn_remove_file, 0, wx.ALL, 6)

        self.btn_add_component = wx.Button(right_panel, label="Add Component")
        self.btn_add_component.Bind(wx.EVT_BUTTON, self.on_add_component)
        btn_row.Add(self.btn_add_component, 0, wx.ALL, 6)

        right_sizer.Add(btn_row, 0, wx.ALIGN_CENTER_HORIZONTAL)

        # Column selection row (dropdown)
        col_row = wx.BoxSizer(wx.HORIZONTAL)
        col_label = wx.StaticText(right_panel, label="Column:")
        col_row.Add(col_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)

        self.column_choice = wx.Choice(right_panel, choices=[])
        self.column_choice.Bind(wx.EVT_CHOICE, self.on_column_choice)
        col_row.Add(self.column_choice, 1, wx.ALL | wx.EXPAND, 6)

        self.chunk_info_label = wx.StaticText(right_panel, label="")
        col_row.Add(self.chunk_info_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)

        right_sizer.Add(col_row, 0, wx.EXPAND)

        # Scrollable area for signal panels
        self.sig_area = wx.ScrolledWindow(right_panel, style=wx.VSCROLL)
        self.sig_area.SetScrollRate(0, 20)
        self.sig_sizer = wx.BoxSizer(wx.VERTICAL)
        self.sig_area.SetSizer(self.sig_sizer)
        right_sizer.Add(self.sig_area, 1, wx.EXPAND | wx.ALL, 6)

        right_panel.SetSizer(right_sizer)

        # Combine left + right
        top_sizer.Add(self.legend_panel, 0, wx.EXPAND | wx.ALL, 6)
        top_sizer.Add(right_panel, 1, wx.EXPAND | wx.ALL, 6)

        main_panel.SetSizer(top_sizer)

        # Status bar at the bottom for chunk information
        self.CreateStatusBar()
        self.SetStatusText("Ready")

    # ----------------------
    # Panel helpers
    # ----------------------
    def _normalize_seq(self, seq) -> np.ndarray:
        """
        Ensure seq is a 1D numpy array of floats with length >= 2.
        Accepts list, tuple, numpy array, pandas Series.
        """
        if seq is None:
            n = max(2, self.DEFAULT_PANEL_LENGTH)
            return np.zeros(n, dtype=float)

        # Pandas DataFrame should not reach this function (handled earlier)
        if isinstance(seq, pd.DataFrame):
            # pick first column if a dataframe is accidentally passed
            if seq.shape[1] > 0:
                series = seq.iloc[:, 0]
                arr = np.asarray(series.dropna(), dtype=float)
            else:
                arr = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        elif isinstance(seq, pd.Series):
            arr = np.asarray(seq.dropna(), dtype=float)
        else:
            arr = np.asarray(seq, dtype=float)

        # flatten 2D arrays that represent a single column
        if arr.ndim > 1:
            # if second dimension is 1, flatten; otherwise pick first column
            if arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                arr = arr[:, 0]

        # ensure at least length 2 for drawing math
        if arr.size < 2:
            arr = np.resize(arr, max(2, self.DEFAULT_PANEL_LENGTH))

        return arr.astype(float)

    def add_panel(self, label: str, seq=None, formula: str = None, draggable: bool = False):
        """Create InteractiveSequencePanel and insert it into layout."""
        # Normalize sequence to 1D numpy
        seq_arr = self._normalize_seq(seq)

        # Choose color
        color = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1

        # Instantiate interactive panel
        sp = InteractiveSequencePanel(self.sig_area, seq_arr, label, formula,
                                      draggable=draggable, visible_count=200,
                                      color=color)
        sp.SetMinSize((-1, 250))

        if formula:
            sp.original_seq = self.original

        if draggable:
            self.result_panel = sp

        # Insert before result panel if result exists (keeps result at bottom)
        if self.result_panel and not draggable:
            # find index of result_panel in sizer
            try:
                idx = next(i for i in range(self.sig_sizer.GetItemCount())
                           if self.sig_sizer.GetItem(i).GetWindow() is self.result_panel)
                self.sig_sizer.Insert(idx, sp, 0, wx.EXPAND | wx.ALL, 5)
            except StopIteration:
                self.sig_sizer.Add(sp, 0, wx.EXPAND | wx.ALL, 5)
        else:
            self.sig_sizer.Add(sp, 0, wx.EXPAND | wx.ALL, 5)

        self.signals.append({'panel': sp, 'formula': formula})
        self._sync_panels()
        self._refresh_legend()
        self.sig_area.FitInside()
        self.sig_area.Layout()
        sp.Refresh()

    def remove_panel(self, panel_widget):
        """Remove a panel instance from UI and internal list."""
        for i, entry in enumerate(self.signals):
            if entry['panel'] is panel_widget:
                self.sig_sizer.Detach(panel_widget)
                panel_widget.Destroy()
                del self.signals[i]
                break
        self._sync_panels()
        self._refresh_legend()
        self.sig_area.FitInside()
        self.sig_area.Layout()

    def _sync_panels(self):
        """Set sync_panels for each panel so hover/zoom/pan propagate."""
        all_panels = [entry['panel'] for entry in self.signals]
        for entry in self.signals:
            panel = entry['panel']
            panel.sync_panels = [p for p in all_panels if p is not panel]

    # ----------------------
    # Legend handling
    # ----------------------
    def _refresh_legend_clear_children(self):
        """Destroy legend children except the first (title). Defensive helper."""
        children = self.legend_panel.GetChildren()
        for i, child in enumerate(children):
            if i == 0:
                continue
            try:
                child.Destroy()
            except Exception:
                # ignore any destroy errors, continue
                pass

    def _refresh_legend_add_entry(self, label, color):
        panel = wx.Panel(self.legend_panel)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        color_box = wx.Panel(panel, size=(16, 16))
        color_box.SetBackgroundColour(color)
        name = wx.StaticText(panel, label=label)
        sizer.Add(color_box, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(name, 0, wx.ALIGN_CENTER_VERTICAL)
        panel.SetSizer(sizer)
        self.legend_sizer.Add(panel, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

    def _refresh_legend(self):
        """Completely rebuild legend to reflect active panels (no duplicates)."""
        # Clear existing entries except title
        self._refresh_legend_clear_children()

        # Add current panel list
        for entry in self.signals:
            p = entry['panel']
            self._refresh_legend_add_entry(p.title, p.color)

        self.legend_panel.Layout()

    # ----------------------
    # Panel update / rebuild
    # ----------------------
    def update_panels(self):
        """Destroy existing panels (safely) and rebuild default stack: Original, Signal 1, Result."""
        # Destroy existing panels
        for entry in self.signals[:]:
            panel = entry['panel']
            try:
                self.sig_sizer.Detach(panel)
                panel.Destroy()
            except Exception:
                pass
            try:
                self.signals.remove(entry)
            except ValueError:
                pass

        # Reset state for new panel set
        self.color_index = 0
        self.result_panel = None

        # Add Original
        if self.original is not None and self.original.size > 0:
            self.add_panel("Original", self.original, draggable=False)
        else:
            self.add_panel("Original", np.zeros(max(2, self.DEFAULT_PANEL_LENGTH)), draggable=False)

        # Add Signal 1 (orig - result)
        if self.original is not None and self.result is not None and self.original.size == self.result.size and self.original.size > 0:
            try:
                diff = (self.original - self.result)
            except Exception:
                # fallback to zeros if subtraction fails
                diff = np.zeros_like(self.original)
            self.add_panel("Signal 1", diff, formula="orig-res", draggable=False)
        else:
            self.add_panel("Signal 1", np.zeros(max(2, self.DEFAULT_PANEL_LENGTH)), draggable=False)

        # Add Result
        if self.result is not None and self.result.size > 0:
            self.add_panel("Result", self.result, draggable=True)
        else:
            self.add_panel("Result", np.zeros(max(2, self.DEFAULT_PANEL_LENGTH)), draggable=True)

        self.sig_area.FitInside()
        self.sig_area.Layout()
        # Update legend finally
        self._refresh_legend()

    # ----------------------
    # Add / Remove Component UI
    # ----------------------
    def on_add_component(self, event):
        """Prompt formula & label then add a new (empty or zero) panel sized to current original length."""
        dlg = wx.TextEntryDialog(self, "Formula (orig,res):", "Add Component", "")
        if dlg.ShowModal() != wx.ID_OK:
            dlg.Destroy()
            return
        formula = dlg.GetValue()
        dlg.Destroy()

        dlg2 = wx.TextEntryDialog(self, "Label:", "Add Component", "")
        if dlg2.ShowModal() != wx.ID_OK:
            dlg2.Destroy()
            return
        label = dlg2.GetValue()
        dlg2.Destroy()

        # create zero sequence with same length as original if available
        if self.original is not None and self.original.size > 1:
            seq = np.zeros_like(self.original)
        else:
            seq = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
        self.add_panel(label, seq, formula=formula, draggable=False)

    # ----------------------
    # File & chunk handling
    # ----------------------
    def on_load_file(self, event):
        """
        Initial file load. Calls load_file(new_file=True), receives:
            content (DataFrame/ndarray/Series), columns list, current_chunk, total_chunks
        Then it populates column dropdown and prepares the displayed data.
        """
        content, columns, chunk_idx, total = load_file(new_file=True, direction="current")
        if content is None:
            wx.MessageBox("No file selected or failed to load.", "Info", wx.OK | wx.ICON_INFORMATION)
            return

        # Save state
        self.current_file_content = content
        self.available_columns = columns or []
        self.current_chunk = chunk_idx
        self.total_chunks = total

        # Default select first column if present
        if self.available_columns:
            self.selected_column_index = 0
            set_selected_columns([self.available_columns[0]])
        else:
            self.selected_column_index = None
            set_selected_columns([])

        # Update dropdown choices
        self._update_column_choice()
        # Build arrays & panels
        self._prepare_and_show_data()

        # Enable chunk buttons if multiple chunks
        self._update_chunk_buttons()

    def on_next_chunk(self, event):
        """Load next chunk (if available) and display it."""
        content, columns, chunk_idx, total = load_file(new_file=False, direction="next")
        if content is None:
            wx.MessageBox("Failed to load next chunk.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.current_file_content = content
        self.available_columns = columns or self.available_columns
        self.current_chunk = chunk_idx
        self.total_chunks = total

        # If no selected column set (unlikely), pick first
        if self.selected_column_index is None and self.available_columns:
            self.selected_column_index = 0
            set_selected_columns([self.available_columns[0]])

        self._update_column_choice()
        self._prepare_and_show_data()
        self._update_chunk_buttons()

    def on_prev_chunk(self, event):
        """Load previous chunk (if available) and display it."""
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

    def _update_column_choice(self):
        """Update the wx.Choice control with available columns."""
        if not self.available_columns:
            self.column_choice.Clear()
            self.column_choice.SetItems([])
            self.column_choice.Refresh()
            self.chunk_info_label.SetLabel("")
            return

        # set items and preserve selection if possible
        self.column_choice.Clear()
        self.column_choice.SetItems(self.available_columns)
        if self.selected_column_index is None:
            self.selected_column_index = 0
        if 0 <= self.selected_column_index < len(self.available_columns):
            self.column_choice.SetSelection(self.selected_column_index)

        # update chunk info label
        if self.total_chunks and self.total_chunks > 1:
            self.chunk_info_label.SetLabel(f"Chunk {self.current_chunk+1}/{self.total_chunks}")
            self.SetStatusText(f"Viewing chunk {self.current_chunk+1} of {self.total_chunks}")
        else:
            self.chunk_info_label.SetLabel("")
            self.SetStatusText("Loaded")

    def on_column_choice(self, event):
        """User picked a column from the dropdown; update selected column & refresh displayed data."""
        sel = self.column_choice.GetSelection()
        if sel == wx.NOT_FOUND:
            return
        self.selected_column_index = int(sel)
        # Tell loadfile to use the selected column name from available_columns
        chosen_name = self.available_columns[self.selected_column_index]
        set_selected_columns([chosen_name])
        # Re-load current chunk (with new selection applied by load_file)
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
        """
        Convert current_file_content into a 1D numpy array, set original/result,
        then rebuild panels. This function is defensive and will accept:
          - numpy ndarray (audio or 1D)
          - pandas DataFrame (one or more columns)
          - pandas Series
        """
        content = self.current_file_content
        if content is None:
            self.original = np.array([])
            self.result = np.array([])
        elif isinstance(content, np.ndarray):
            arr = np.asarray(content, dtype=float)
            # flatten if needed
            arr = arr.flatten()
            self.original = arr
            self.result = arr.copy()
        elif isinstance(content, pd.DataFrame):
            # if column selection present, pick first selected column; otherwise pick first column
            if self.selected_column_index is None:
                # default to first column
                col_idx = 0
            else:
                col_idx = self.selected_column_index

            # guard against invalid index
            if col_idx < 0 or col_idx >= content.shape[1]:
                col_idx = 0

            series = content.iloc[:, col_idx]
            arr = np.asarray(series.dropna(), dtype=float)
            arr = arr.flatten()
            self.original = arr
            self.result = arr.copy()
        elif isinstance(content, pd.Series):
            arr = np.asarray(content.dropna(), dtype=float)
            arr = arr.flatten()
            self.original = arr
            self.result = arr.copy()
        else:
            # last resort: attempt numpy conversion
            try:
                arr = np.asarray(content, dtype=float).flatten()
                self.original = arr
                self.result = arr.copy()
            except Exception:
                self.original = np.array([])
                self.result = np.array([])

        # Ensure we always have a 1D array for panels (fallback)
        if self.original is None or self.original.size == 0:
            self.original = np.zeros(self.DEFAULT_PANEL_LENGTH, dtype=float)
            self.result = self.original.copy()

        # Finally rebuild panels
        self.update_panels()

    def _update_chunk_buttons(self):
        """Enable/disable chunk navigation buttons and update the status text."""
        if self.total_chunks and self.total_chunks > 1:
            self.btn_prev.Enable(self.current_chunk > 0)
            self.btn_next.Enable(self.current_chunk < (self.total_chunks - 1))
            self.SetStatusText(f"Chunk {self.current_chunk + 1} of {self.total_chunks}")
            self.chunk_info_label.SetLabel(f"Chunk {self.current_chunk + 1}/{self.total_chunks}")
        else:
            self.btn_prev.Disable()
            self.btn_next.Disable()
            self.SetStatusText("Loaded")
            self.chunk_info_label.SetLabel("")

    # ----------------------
    # Remove file / Reset UI
    # ----------------------
    def on_remove_file(self, event):
        """Clear loaded file and reset state and panels to defaults."""
        self.current_file_content = None
        self.available_columns = []
        self.selected_column_index = None
        self.current_chunk = 0
        self.total_chunks = 0
        set_selected_columns([])

        # clear dropdown
        self.column_choice.Clear()
        self.column_choice.SetItems([])
        self.chunk_info_label.SetLabel("")
        self.SetStatusText("File removed")

        # remove existing panels and add defaults
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
        self.result_panel = None
        self.original = np.array([])
        self.result = np.array([])
        self.update_panels()  # will create default placeholders

    # ----------------------
    # Main loop
    # ----------------------
def main():
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()


if __name__ == "__main__":
    main()
