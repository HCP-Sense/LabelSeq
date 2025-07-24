import wx
import numpy as np

class SequencePanel(wx.Panel):
    def __init__(self, parent, seq, padding=50, draggable=False,
                 title="", xlabel="Sample", ylabel="Amplitude"):
        super().__init__(parent, style=wx.NO_FULL_REPAINT_ON_RESIZE)
        self.seq = list(seq)
        self.padding = padding
        self.circle_radius = 4
        self.draggable = draggable
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.dragging = False
        self.selected_idx = None

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE,  self.on_size)

        if self.draggable:
            self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
            self.Bind(wx.EVT_MOTION,    self.on_mouse_move)
            self.Bind(wx.EVT_LEFT_UP,   self.on_left_up)

        self._bg_bmp = None

    def _create_background_bitmap(self):
        w, h = self.GetClientSize()
        if w <= 0 or h <= 0 or len(self.seq) < 2:
            return

        bmp = wx.Bitmap(w, h)
        mdc = wx.MemoryDC(bmp)
        mdc.SetBackground(wx.Brush(wx.Colour(255,255,255)))
        mdc.Clear()

        graph_w = w - 2 * self.padding
        graph_h = h - 2 * self.padding
        n_points = len(self.seq)

        scale_x = graph_w / (n_points - 1)
        max_val = max(self.seq)
        min_val = min(self.seq)
        val_range = max_val - min_val if max_val != min_val else 1
        scale_y = graph_h / val_range

        def map_point(i, v):
            x = int(self.padding + i * scale_x)
            y = int(h - self.padding - (v - min_val) * scale_y)
            return x, y

        # Draw plot box
        mdc.SetPen(wx.Pen(wx.LIGHT_GREY))
        mdc.DrawRectangle(self.padding, self.padding, graph_w, graph_h)

        # Draw Y-axis ticks and labels, track widest label
        mdc.SetPen(wx.Pen(wx.BLACK, 1))
        mdc.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT,
                            wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        max_tick_w = 0
        ticks = []
        for i in range(6):
            val = min_val + i * val_range / 5
            y = int(h - self.padding - (val - min_val) * scale_y)
            ticks.append((val, y))
            mdc.DrawLine(self.padding - 5, y, self.padding, y)
            txt = f"{val:.1f}"
            tw, th = mdc.GetTextExtent(txt)
            max_tick_w = max(max_tick_w, tw)
            mdc.DrawText(txt, self.padding - 10 - tw, y - th//2)

        # Draw X-axis ticks
        for i in range(0, n_points, max(1, n_points//10)):
            x, _ = map_point(i, min_val)
            mdc.DrawLine(x, h - self.padding, x, h - self.padding + 5)
            mdc.DrawText(str(i), x-10, h - self.padding + 8)

        # X-label
        lw, lh = mdc.GetTextExtent(self.xlabel)
        mdc.DrawText(self.xlabel, (w-lw)//2, h-25)

        # Y-label: place to the left of the widest tick label + margin
        yw, yh = mdc.GetTextExtent(self.ylabel)
        margin = 15
        ylabel_x = self.padding - max_tick_w - margin - (yh//2)
        mdc.DrawRotatedText(self.ylabel,
                            ylabel_x,
                            (h + yw)//2,
                            90)

        # Title (centered)
        mdc.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT,
                            wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        tw, th = mdc.GetTextExtent(self.title)
        mdc.DrawText(self.title, (w-tw)//2, self.padding//4)

        # Plot line + pts
        pts = [map_point(i, v) for i,v in enumerate(self.seq)]
        mdc.SetPen(wx.Pen(wx.BLUE, 1))
        for (x0,y0),(x1,y1) in zip(pts, pts[1:]):
            mdc.DrawLine(x0,y0,x1,y1)
        if self.draggable:
            mdc.SetBrush(wx.Brush(wx.BLUE))
            for x,y in pts:
                mdc.DrawCircle(x,y,self.circle_radius)

        mdc.SelectObject(wx.NullBitmap)
        self._bg_bmp = bmp

    def on_size(self, evt):
        self._create_background_bitmap()
        self.Refresh()
        evt.Skip()

    def on_paint(self, evt):
        dc = wx.BufferedPaintDC(self)
        if self._bg_bmp:
            dc.DrawBitmap(self._bg_bmp, 0, 0)

    def on_left_down(self, evt):
        x0,y0 = evt.GetPosition()
        w,h = self.GetClientSize()
        graph_w = w-2*self.padding
        graph_h = h-2*self.padding
        n = len(self.seq)
        scale_x = graph_w/(n-1)
        max_val,min_val = max(self.seq),min(self.seq)
        vr = max_val-min_val if max_val!=min_val else 1
        scale_y = graph_h/vr

        for i,v in enumerate(self.seq):
            cx = self.padding + i*scale_x
            cy = h-self.padding - (v-min_val)*scale_y
            if (x0-cx)**2 + (y0-cy)**2 <= self.circle_radius**2:
                self.selected_idx = i
                self.dragging = True
                self.CaptureMouse()
                return

    def on_mouse_move(self, evt):
        if not(self.draggable and self.dragging and evt.LeftIsDown()):
            return
        _,y = evt.GetPosition()
        h = self.GetClientSize().height
        graph_h = h-2*self.padding
        max_val,min_val = max(self.seq),min(self.seq)
        vr = max_val-min_val if max_val!=min_val else 1
        v = ((h-self.padding-y)/graph_h)*vr + min_val
        self.seq[self.selected_idx] = max(min_val, min(max_val, v))
        self._create_background_bitmap()
        self.Refresh()

    def on_left_up(self, evt):
        if self.dragging:
            self.dragging=False
            self.ReleaseMouse()
            self._create_background_bitmap()
            self.selected_idx=None
            self.Refresh()


class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="LabelSeq", size=(1000,800))
        panel = wx.Panel(self)

        # Top label only
        lbl = wx.StaticText(panel, label="LabelSeq")
        lbl.SetFont(wx.Font(14, wx.FONTFAMILY_SWISS,
                            wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))

        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.Add(lbl, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)

        # Dummy data
        original = np.random.rand(150)
        spikes   = (np.random.rand(150)-0.5)*0.4
        result   = original - spikes

        # Create three plots
        plots = [
            SequencePanel(panel, original, title="Original Signal"),
            SequencePanel(panel, spikes,    title="Spike Signal"),
            SequencePanel(panel, result,    title="Result", draggable=True),
        ]

        plot_sizer = wx.BoxSizer(wx.VERTICAL)
        for p in plots:
            plot_sizer.Add(p, 1, wx.EXPAND|wx.ALL, 5)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(top_sizer, 0, wx.EXPAND)
        main_sizer.Add(plot_sizer, 1, wx.EXPAND)
        panel.SetSizer(main_sizer)

        self.Centre()
        self.Show()


if __name__ == "__main__":
    app = wx.App(False)
    MainFrame()
    app.MainLoop()
