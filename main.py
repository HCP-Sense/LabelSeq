"""main.py: Entry and main window class"""

__author___ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Moye Nyuysoni Glein Perry"

# Project Imports
from InteractiveSequencePanel import InteractiveSequencePanel
from LoadFile import  load_file

# Third-Party Imports
import wx
import numpy as np




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
        self.original =[]
        self.result_panel = None
        self.content = []
        self.panels = []




        # Top-Level Split
        main_panel = wx.Panel(self)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Legend Panel (Left Side)
        self.legend_panel = wx.Panel(main_panel, size=(150, -1))
        self.legend_panel.SetBackgroundColour(wx.Colour(245, 245, 245))
        self.legend_sizer = wx.BoxSizer(wx.VERTICAL)
        legend_label = wx.StaticText(self.legend_panel, label="Legend")
        font = legend_label.GetFont()
        font.MakeBold()
        legend_label.SetFont(font)
        self.legend_sizer.Add(legend_label, 0, wx.ALL, 8)
        self.legend_panel.SetSizer(self.legend_sizer)

        # Main Controls and Signal Panels (Right Side)
        right_panel = wx.Panel(main_panel)
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        btn_row = wx.BoxSizer(wx.HORIZONTAL)

        # First button
        btn_add = wx.Button(right_panel, label="Add Component")
        btn_add.Bind(wx.EVT_BUTTON, self.on_add)
        btn_add.SetBackgroundColour(wx.Colour(100, 200, 255))  # light blue
        btn_add.SetForegroundColour(wx.Colour(0, 0, 0))  # black text
        right_sizer.Add(btn_add, 0,wx.ALL| wx.CENTER, 5)

        # Second button
        btn_import = wx.Button(right_panel, label="Import Data")
        btn_import.Bind(wx.EVT_BUTTON, self.on_load_file)
        btn_import.SetBackgroundColour(wx.Colour(200, 255, 200))  # light green
        btn_import.SetForegroundColour(wx.Colour(0, 0, 0))  # black text
        btn_row.Add(btn_import, 0, wx.ALL, 5)

        self.sig_area = wx.ScrolledWindow(right_panel, style=wx.VSCROLL)
        self.sig_area.SetScrollRate(0, 20)
        self.sig_sizer = wx.BoxSizer(wx.VERTICAL)
        self.sig_area.SetSizer(self.sig_sizer)
        right_sizer.Add(self.sig_area, 1, wx.EXPAND | wx.ALL, 5)

        right_panel.SetSizer(right_sizer)

        # Combine Left (Legend) and Right (Panels)
        top_sizer.Add(self.legend_panel, 0, wx.EXPAND | wx.ALL, 5)
        top_sizer.Add(right_panel, 1, wx.EXPAND | wx.ALL, 5)

        main_panel.SetSizer(top_sizer)


        if self.original is not None:
            # always convert to numpy so subtraction works
            self.original = np.array(self.original)
            self.result = self.original.copy()
        else:
            self.result = None


        # default panels
        self.add_panel("Original", self.original, draggable=False)

        if self.original is not None and self.result is not None:
            # subtraction is safe because both are numpy arrays
            diff = self.original - self.result
            self.add_panel("Signal 1", diff, formula="orig-res", draggable=False)
            self.add_panel("Result", self.result, draggable=True)
        else:
            # create empty placeholders if nothing is loaded
            self.add_panel("Signal 1", draggable=False)
            self.add_panel("Result", self.result, draggable=True)

        self.Centre()
        self.Show()

    def add_panel(self, label, seq, formula=None, draggable=False):
        # Pick a color
        color = COLOR_CYCLE[self.color_index % len(COLOR_CYCLE)]
        self.color_index += 1

        # Create Sequence Panel (use SequencePanelEvents for full interactivity)
        sp = InteractiveSequencePanel(self.sig_area, seq, label, formula,
                                      draggable=draggable, visible_count=200,
                                      color=color)
        sp.SetMinSize((-1, 250))

        if formula:
            sp.original_seq = self.original

        if draggable:
            self.result_panel = sp

        # Add panel to layout (insert above result if needed)
        if self.result_panel and not draggable:
            idx = next(
                i for i in range(self.sig_sizer.GetItemCount())
                if self.sig_sizer.GetItem(i).GetWindow() is self.result_panel
            )
            self.sig_sizer.Insert(idx, sp, 0, wx.EXPAND | wx.ALL, 5)
        else:
            self.sig_sizer.Add(sp, 0, wx.EXPAND | wx.ALL, 5)

        self.signals.append({'panel': sp, 'formula': formula})

        # Sync Panels
        all_panels = [entry['panel'] for entry in self.signals]
        for entry in self.signals:
            panel = entry['panel']
            panel.sync_panels = [p for p in all_panels if p != panel]

        # Add to legend
        self.add_legend_entry(label, color)

        self.sig_area.FitInside()
        self.sig_area.Layout()
        sp.Refresh()

    def add_legend_entry(self, label, color):
        panel = wx.Panel(self.legend_panel)
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        color_box = wx.Panel(panel, size=(16, 16))
        color_box.SetBackgroundColour(color)

        name = wx.StaticText(panel, label=label)
        sizer.Add(color_box, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(name, 0, wx.ALIGN_CENTER_VERTICAL)

        panel.SetSizer(sizer)
        self.legend_sizer.Add(panel, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        self.legend_panel.Layout()

    def on_add(self, evt):
        dlg = wx.TextEntryDialog(self, "Formula (orig,res):", "New", "")
        if dlg.ShowModal() == wx.ID_OK:
            formula = dlg.GetValue()
            dlg.Destroy()
            seq = [0.0] * len(self.original)
            ld = wx.TextEntryDialog(self, "Label:", "New", "")
            if ld.ShowModal() == wx.ID_OK:
                label = ld.GetValue()
                ld.Destroy()
                self.add_panel(label, seq, formula=formula, draggable=False)
            else:
                ld.Destroy()
        else:
            dlg.Destroy()

    def remove_panel(self, panel):
        for i, entry in enumerate(self.signals):
            if entry['panel'] is panel:
                self.sig_sizer.Detach(panel)
                panel.Destroy()
                del self.signals[i]
                break

        self.sig_area.FitInside()
        self.sig_area.Layout()

    def on_load_file(self, event):
        """Load file via your load_file() and rebuild panels safely."""

        # Load content
        self.content = load_file()

        if self.content is None:
            wx.MessageBox("No file selected or failed to load.",
                          "Info", wx.OK | wx.ICON_INFORMATION)
            return

        # Convert to safe NumPy array
        try:
            if isinstance(self.content, np.ndarray):
                arr = np.asarray(self.content, dtype=float)

            elif isinstance(self.content, pd.DataFrame):
                arr = self.content.to_numpy(dtype=float)

            elif isinstance(self.content, pd.Series):
                arr = self.content.to_numpy(dtype=float)[:, np.newaxis]

            else:
                arr = np.array(self.content, dtype=float)
                # ensure at least 2D
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]

        except Exception:
            arr = np.empty((0, 0), dtype=float)

        # Fallback if empty
        if arr is None or arr.size == 0:
            arr = np.empty((0, 0), dtype=float)

        # ✅ Immediately update original and result
        self.original = arr
        self.result = self.original.copy()

        # Optional: update content for panels
        self.content = self.original

        # Now rebuild panels safely
        self.update_panels()

    def update_panels(self):
        """
        Safely rebuild the main panels based on current self.original and self.result.
        Old panels are destroyed and removed from sizer to prevent duplication.
        """
        # Remove and destroy all existing panels in the sig_sizer
        for entry in self.signals[:]:
            panel = entry['panel']
            self.sig_sizer.Detach(panel)  # remove from sizer
            panel.Destroy()  # destroy the panel
            self.signals.remove(entry)  # remove from list

        # Reset color index so colors start fresh
        self.color_index = 0
        self.result_panel = None

        # Original panel
        if self.original is not None and self.original.size > 0:
            self.add_panel("Original", self.original, draggable=False)
        else:
            self.add_panel("Original", draggable=False)

        # Signal 1 panel = difference between original and result
        if (self.original is not None and self.result is not None
                and self.original.size > 0 and self.result.size > 0):
            diff = self.original - self.result
            self.add_panel("Signal 1", diff, formula="orig-res", draggable=False)
        else:
            self.add_panel("Signal 1", draggable=False)

        # Result panel
        if self.result is not None and self.result.size > 0:
            self.add_panel("Result", self.result, draggable=True)
        else:
            self.add_panel("Result", draggable=True)

        # Refresh layout
        self.sig_area.FitInside()
        self.sig_area.Layout()


if __name__ == '__main__':
    app = wx.App(False)
    MainFrame()
    app.MainLoop()
