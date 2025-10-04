"""main.py: Entry point; builds main window, manages multiple panels, links legends, updates, and supports file deletion."""

__author___ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Moye Nyuysoni Glein Perry"

# Project Imports
from InteractiveSequencePanel import InteractiveSequencePanel
from LoadFile import load_file

# Third-Party Imports
import wx
import numpy as np
import pandas as pd

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
    def __init__(self):
        super().__init__(None, title="LabelSeq (GC‑only)", size=(1200, 800))
        self.SetBackgroundColour(wx.WHITE)

        self.color_index = 0
        self.signals = []
        self.original = []
        self.result_panel = None
        self.content = []
        self.panels = []
        self.current_file = None

        self._build_ui()
        self.Centre()
        self.Show()

    # ------------------- UI Setup -------------------
    def _build_ui(self):
        main_panel = wx.Panel(self)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Legend Panel (Left)
        self.legend_panel = wx.Panel(main_panel, size=(150, -1))
        self.legend_panel.SetBackgroundColour(wx.Colour(245, 245, 245))
        self.legend_sizer = wx.BoxSizer(wx.VERTICAL)
        legend_label = wx.StaticText(self.legend_panel, label="Legend")
        font = legend_label.GetFont()
        font.MakeBold()
        legend_label.SetFont(font)
        self.legend_sizer.Add(legend_label, 0, wx.ALL, 8)
        self.legend_panel.SetSizer(self.legend_sizer)

        # Right Panel (Controls + Signal Panels)
        right_panel = wx.Panel(main_panel)
        right_sizer = wx.BoxSizer(wx.VERTICAL)

        # Top buttons
        btn_row = wx.BoxSizer(wx.HORIZONTAL)

        self.btn_add = wx.Button(right_panel, label="Add Component")
        self.btn_add.Bind(wx.EVT_BUTTON, self.on_add)
        self.btn_add.SetBackgroundColour(wx.Colour(100, 200, 255))
        self.btn_add.SetForegroundColour(wx.BLACK)

        self.btn_import = wx.Button(right_panel, label="Import Data")
        self.btn_import.Bind(wx.EVT_BUTTON, self.on_load_file)
        self.btn_import.SetBackgroundColour(wx.Colour(200, 255, 200))
        self.btn_import.SetForegroundColour(wx.BLACK)

        self.btn_delete_file = wx.Button(right_panel, label="Delete File")
        self.btn_delete_file.Bind(wx.EVT_BUTTON, self.on_delete_file)
        self.btn_delete_file.SetBackgroundColour(wx.Colour(255, 200, 200))
        self.btn_delete_file.SetForegroundColour(wx.BLACK)

        btn_row.Add(self.btn_import, 0, wx.ALL, 5)
        btn_row.Add(self.btn_delete_file, 0, wx.ALL, 5)
        btn_row.Add(self.btn_add, 0, wx.ALL, 5)

        right_sizer.Add(btn_row, 0, wx.CENTER | wx.ALL, 5)

        # Signal display area
        self.sig_area = wx.ScrolledWindow(right_panel, style=wx.VSCROLL)
        self.sig_area.SetScrollRate(0, 20)
        self.sig_sizer = wx.BoxSizer(wx.VERTICAL)
        self.sig_area.SetSizer(self.sig_sizer)
        right_sizer.Add(self.sig_area, 1, wx.EXPAND | wx.ALL, 5)

        right_panel.SetSizer(right_sizer)

        # Combine left and right
        top_sizer.Add(self.legend_panel, 0, wx.EXPAND | wx.ALL, 5)
        top_sizer.Add(right_panel, 1, wx.EXPAND | wx.ALL, 5)

        main_panel.SetSizer(top_sizer)

        # Add default panels placeholders
        self.add_panel("Original", draggable=False)
        self.add_panel("Signal 1", draggable=False)
        self.add_panel("Result", draggable=True)

    # ------------------- Panel Management -------------------
    def add_panel(self, label, seq=None, formula=None, draggable=False):
        # pick a color
        color = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1

        if seq is None:
            seq = np.zeros(self.original.shape if len(self.original) else (100,))

        sp = InteractiveSequencePanel(
            self.sig_area, seq, label, formula,
            draggable=draggable, visible_count=200, color=color
        )
        sp.SetMinSize((-1, 250))

        if formula:
            sp.original_seq = self.original

        if draggable:
            self.result_panel = sp

        self.sig_sizer.Add(sp, 0, wx.EXPAND | wx.ALL, 5)
        self.signals.append({'panel': sp, 'formula': formula})

        self.sync_panels()
        self.refresh_legend()
        self.sig_area.FitInside()
        self.sig_area.Layout()
        sp.Refresh()

    def remove_panel(self, panel):
        for i, entry in enumerate(self.signals):
            if entry['panel'] is panel:
                self.sig_sizer.Detach(panel)
                panel.Destroy()
                del self.signals[i]
                break
        self.sync_panels()
        self.refresh_legend()
        self.sig_area.FitInside()
        self.sig_area.Layout()

    def sync_panels(self):
        all_panels = [entry['panel'] for entry in self.signals]
        for entry in self.signals:
            panel = entry['panel']
            panel.sync_panels = [p for p in all_panels if p != panel]

    # ------------------- Legend -------------------
    def refresh_legend(self):
        """Only show active panel entries"""
        children = self.legend_panel.GetChildren()
        for i, child in enumerate(children):
            if i == 0:  # skip title
                continue
            child.Destroy()

        for entry in self.signals:
            panel = wx.Panel(self.legend_panel)
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            color_box = wx.Panel(panel, size=(16, 16))
            color_box.SetBackgroundColour(entry['panel'].color)
            name = wx.StaticText(panel, label=entry['panel'].title)
            sizer.Add(color_box, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
            sizer.Add(name, 0, wx.ALIGN_CENTER_VERTICAL)
            panel.SetSizer(sizer)
            self.legend_sizer.Add(panel, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        self.legend_panel.Layout()

    # ------------------- Buttons -------------------
    def on_add(self, evt):
        dlg = wx.TextEntryDialog(self, "Formula (orig,res):", "New", "")
        if dlg.ShowModal() == wx.ID_OK:
            formula = dlg.GetValue()
            dlg.Destroy()
            seq = [0.0] * len(self.original) if len(self.original) else [0.0] * 100
            ld = wx.TextEntryDialog(self, "Label:", "New", "")
            if ld.ShowModal() == wx.ID_OK:
                label = ld.GetValue()
                ld.Destroy()
                self.add_panel(label, seq, formula=formula, draggable=False)
            else:
                ld.Destroy()
        else:
            dlg.Destroy()

    def on_delete_file(self, evt):
        """Remove file data and reset panels"""
        self.current_file = None
        self.original = []
        self.result = []
        for entry in self.signals[:]:
            panel = entry['panel']
            self.sig_sizer.Detach(panel)
            panel.Destroy()
            self.signals.remove(entry)
        self.color_index = 0
        self.add_panel("Original", draggable=False)
        self.add_panel("Signal 1", draggable=False)
        self.add_panel("Result", draggable=True)

    # ------------------- File Loading -------------------
    def on_load_file(self, evt):
        self.content = load_file()
        if self.content is None:
            wx.MessageBox("No file selected or failed to load.",
                          "Info", wx.OK | wx.ICON_INFORMATION)
            return

        # Column selection for DataFrame (except audio)
        if isinstance(self.content, pd.DataFrame):
            columns = self.content.columns.tolist()
            if len(columns) > 1:
                dlg = wx.MultiChoiceDialog(
                    self, "Select columns to load", "Columns", columns
                )
                if dlg.ShowModal() == wx.ID_OK:
                    selected = dlg.GetSelections()
                    self.content = self.content.iloc[:, selected]
                dlg.Destroy()

        # Convert to NumPy
        try:
            if isinstance(self.content, np.ndarray):
                arr = np.asarray(self.content, dtype=float)
            elif isinstance(self.content, pd.DataFrame):
                arr = self.content.to_numpy(dtype=float)
            elif isinstance(self.content, pd.Series):
                arr = self.content.to_numpy(dtype=float)[:, np.newaxis]
            else:
                arr = np.array(self.content, dtype=float)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
        except Exception:
            arr = np.empty((0, 0), dtype=float)

        if arr.size == 0:
            arr = np.empty((0, 0), dtype=float)

        self.original = arr
        self.result = self.original.copy()
        self.update_panels()

    # ------------------- Panel Refresh -------------------
    def update_panels(self):
        for entry in self.signals[:]:
            panel = entry['panel']
            self.sig_sizer.Detach(panel)
            panel.Destroy()
            self.signals.remove(entry)

        self.color_index = 0
        self.result_panel = None

        if self.original is not None and self.original.size > 0:
            self.add_panel("Original", self.original, draggable=False)
        else:
            self.add_panel("Original", draggable=False)

        if self.original is not None and self.result is not None \
                and self.original.size > 0 and self.result.size > 0:
            diff = self.original - self.result
            self.add_panel("Signal 1", diff, formula="orig-res", draggable=False)
        else:
            self.add_panel("Signal 1", draggable=False)

        if self.result is not None and self.result.size > 0:
            self.add_panel("Result", self.result, draggable=True)
        else:
            self.add_panel("Result", draggable=True)

        self.sig_area.FitInside()
        self.sig_area.Layout()


if __name__ == '__main__':
    app = wx.App(False)
    MainFrame()
    app.MainLoop()
