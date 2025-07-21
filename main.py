import wx
import numpy as np

class SequencePanel(wx.Panel):
    def __init__(self, parent, seq, scale_x=10, scale_y=200, padding=50):
        super().__init__(parent, style=wx.NO_FULL_REPAINT_ON_RESIZE)
        self.seq = list(seq)
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.padding = padding
        self.circle_radius = 5

        # Drag state
        self.dragging = False
        self.selected_idx = None

        # Double‑buffering setup
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE,  self.on_size)

        # Mouse events
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_MOTION,    self.on_mouse_move)
        self.Bind(wx.EVT_LEFT_UP,   self.on_left_up)

        # Initial static background
        self._create_background_bitmap()

    def _create_background_bitmap(self):
        """Draw full graph into self._bg_bmp."""
        w, h = self.GetClientSize()
        if w <= 0 or h <= 0:
            return

        bmp = wx.Bitmap(w, h)
        mdc = wx.MemoryDC(bmp)
        mdc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        mdc.Clear()

        # Compute and draw lines
        pen = wx.Pen(wx.BLACK, 1)
        mdc.SetPen(pen)
        pts = []
        for i, v in enumerate(self.seq):
            x = int(self.padding + i * self.scale_x)
            y = int(self.padding + (1 - v) * self.scale_y)
            pts.append((x, y))
        for (x0,y0),(x1,y1) in zip(pts, pts[1:]):
            mdc.DrawLine(x0,y0, x1,y1)

        # Draw blue circles
        mdc.SetBrush(wx.Brush(wx.BLUE))
        for x, y in pts:
            mdc.DrawCircle(x, y, self.circle_radius)

        mdc.SelectObject(wx.NullBitmap)
        self._bg_bmp = bmp

    def on_size(self, event):
        self._create_background_bitmap()
        event.Skip()

    def on_paint(self, event):
        dc = wx.BufferedPaintDC(self)
        if hasattr(self, "_bg_bmp"):
            dc.DrawBitmap(self._bg_bmp, 0, 0)

    def on_left_down(self, event):
        x0, y0 = event.GetPosition()
        # hit‑test blue circles
        for i, v in enumerate(self.seq):
            cx = self.padding + i * self.scale_x
            cy = self.padding + (1 - v) * self.scale_y
            if (x0 - cx)**2 + (y0 - cy)**2 <= self.circle_radius**2:
                self.selected_idx = i
                self.dragging = True
                self.CaptureMouse()
                return

    def on_mouse_move(self, event):
        if not (self.dragging and event.LeftIsDown()):
            return

        # Update data
        _, y = event.GetPosition()
        v = 1 - ((y - self.padding) / self.scale_y)
        self.seq[self.selected_idx] = max(0.0, min(1.0, v))

        # Rebuild static background with this point moved
        self._create_background_bitmap()

        # Immediate blit
        dc = wx.ClientDC(self)
        dc.DrawBitmap(self._bg_bmp, 0, 0)

    def on_left_up(self, event):
        if self.dragging:
            self.dragging = False
            self.ReleaseMouse()
            # Final commit: ensure everything is in the background
            self._create_background_bitmap()
            self.selected_idx = None
            self.Refresh()

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="LabelSeq Real‑Time Blue Drag", size=(1000, 600))
        seq = np.random.rand(100)
        SequencePanel(self, seq, scale_x=9, scale_y=400, padding=50)
        self.Show()

if __name__ == "__main__":
    app = wx.App(False)
    MainFrame()
    app.MainLoop()
