"""main.py: Entry and main window class"""

__author___ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Moye Nyuysoni Glein Perry"

# Project Imports
from InteractiveSequencePanel import InteractiveSequencePanel

# Third-Party Imports
import wx
import numpy as np
import librosa



file_name="HeartB/Aunlabelledtest__201108011117.wav"
file_content,sr = librosa.load(file_name, sr=None)
file_content=file_content[0:10000]
file_content=file_content*100
print(len(file_content))
print(min(file_content))
print(max(file_content))


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
        self.result_panel = None

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

        btn = wx.Button(right_panel, label="Add Component")
        btn.Bind(wx.EVT_BUTTON, self.on_add)
        right_sizer.Add(btn, 0, wx.ALL | wx.CENTER, 5)

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

        self.original = file_content
        self.result = self.original.copy()

        # default panels
        self.add_panel("Original", self.original, draggable=False)
        self.add_panel("Signal 1", self.original - self.result,
                       formula="orig-res", draggable=False)
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


if __name__ == '__main__':
    app = wx.App(False)
    MainFrame()
    app.MainLoop()
