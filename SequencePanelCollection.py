import wx
from FormulaParser import safe_eval

class SequencePanel(wx.Panel):
    def __init__(self, parent, seq, title, formula=None,
                 draggable=False, visible_count=None):
        super().__init__(parent, style=wx.NO_FULL_REPAINT_ON_RESIZE)
        self.seq = list(seq)
        self.original_seq = None
        self.title = title
        self.formula = formula
        self.n = len(self.seq)
        self.padding = 80
        self.radius = 6
        self.draggable = draggable
        self.visible_count = visible_count or self.n
        self.zoom_factor = max(1.0, (self.n - 1) / self.visible_count)
        self.pan_offset = 0

        self.dragging = False
        self.selected_idx = None
        self.sync_panels = []
        self._pan_start = None
        self._pan_origin = 0

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: (self.Refresh(), e.Skip()))
        self.Bind(wx.EVT_LEFT_DCLICK, self.on_double_click)
        if self.draggable:
            self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
            self.Bind(wx.EVT_MOTION, self.on_mouse_move)
            self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
            self.Bind(wx.EVT_MIDDLE_DOWN, self.on_middle_down)
            self.Bind(wx.EVT_MIDDLE_UP, self.on_middle_up)
            self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)

    def to_px(self, i, v, w, h):
        gw = w - 2 * self.padding
        gh = h - 2 * self.padding
        sx = gw / (self.n - 1) * self.zoom_factor
        mx, mn = max(self.seq), min(self.seq)
        rng = (mx - mn) or 1
        mx += 0.1 * rng; mn -= 0.1 * rng; rng = mx - mn
        sy = gh / rng
        x = self.padding + i * sx + self.pan_offset
        y = h - self.padding - (v - mn) * sy
        return x, y

    def on_paint(self, evt):
        w, h = self.GetClientSize()
        if w < 2 or h < 2 or self.n < 2:
            return
        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return

        # set default font for labels
        f_wx = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        f_gc = gc.CreateFont(f_wx, wx.BLACK)
        gc.SetFont(f_gc)

        # Background
        gc.SetBrush(wx.Brush(wx.WHITE))
        gc.DrawRectangle(0, 0, w, h)

        # Axes
        gc.SetPen(wx.Pen(wx.LIGHT_GREY))
        # Y axis
        gc.StrokeLine(self.padding, self.padding,
                      self.padding, h - self.padding)
        # X axis
        gc.StrokeLine(self.padding, h - self.padding,
                      w - self.padding, h - self.padding)

        # Y ticks and labels
        mx, mn = max(self.seq), min(self.seq)
        rng = (mx - mn) or 1
        mx += 0.1 * rng; mn -= 0.1 * rng; rng = mx - mn
        for k in range(6):
            val = mn + k * (rng / 5)
            y = h - self.padding - (val - mn) * ((h - 2 * self.padding) / rng)
            gc.StrokeLine(self.padding - 5, y, self.padding, y)
            text = f"{val:.1f}"
            tw, th = gc.GetTextExtent(text)
            gc.DrawText(text, self.padding - 10 - tw, y - th / 2)

        # X ticks and labels
        step = max(1, self.n // 10)
        for i in range(0, self.n, step):
            x, _ = self.to_px(i, mn, w, h)
            if self.padding <= x <= w - self.padding:
                gc.StrokeLine(x, h - self.padding, x, h - self.padding + 5)
                label = str(i)
                tw, th = gc.GetTextExtent(label)
                gc.DrawText(label, x - tw / 2, h - self.padding + 8)

        # Title (bold)
        f2 = wx.Font(12, wx.FONTFAMILY_DEFAULT,
                     wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        fg = gc.CreateFont(f2, wx.BLACK)
        gc.SetFont(fg)
        tw, th = gc.GetTextExtent(self.title)
        gc.DrawText(self.title, (w - tw) / 2, 5)

        # Restore default font for plotting
        gc.SetFont(f_gc)

        # Plot line
        pts = [self.to_px(i, v, w, h) for i, v in enumerate(self.seq)]
        if pts:
            path = gc.CreatePath()
            path.MoveToPoint(*pts[0])
            for p in pts[1:]: path.AddLineToPoint(*p)
            gc.SetPen(wx.Pen(wx.BLUE, 1))
            gc.StrokePath(path)

        # Draggable handle
        if self.draggable and self.selected_idx is not None:
            x, y = pts[self.selected_idx]
            gc.SetBrush(wx.Brush(wx.BLUE))
            gc.DrawEllipse(x - self.radius, y - self.radius,
                           2 * self.radius, 2 * self.radius)

    def on_double_click(self, evt):
        x, y = evt.GetPosition()
        w, _ = self.GetClientSize()
        dc = wx.ClientDC(self)
        tw, th = dc.GetTextExtent(self.title)
        tx = (w - tw) // 2
        if tx <= x <= tx + tw and 0 <= y <= th:
            dlg = wx.TextEntryDialog(self, "New label:", "Rename", self.title)
            if dlg.ShowModal() == wx.ID_OK:
                self.title = dlg.GetValue()
                self.Refresh()
            dlg.Destroy()
            if self.formula:
                dlg2 = wx.TextEntryDialog(self, "Formula:", "Edit", self.formula)
                if dlg2.ShowModal() == wx.ID_OK:
                    self.formula = dlg2.GetValue()
                dlg2.Destroy()

    def on_left_down(self, evt):
        x0, y0 = evt.GetPosition()
        w, h = self.GetClientSize()
        best, idx = float('inf'), None
        for i, v in enumerate(self.seq):
            cx, cy = self.to_px(i, v, w, h)
            d = (cx - x0) ** 2 + (cy - y0) ** 2
            if d < best:
                best, idx = d, i
        if idx is not None and best < (self.radius * 3) ** 2:
            self.selected_idx = idx
            self.dragging = True
            self.CaptureMouse()

    def on_mouse_move(self, evt):
        if self.draggable and self.dragging and evt.LeftIsDown():
            x, y = evt.GetPosition()
            w, h = self.GetClientSize()
            mx, mn = max(self.seq), min(self.seq)
            rng = (mx - mn) or 1
            mx += 0.1 * rng; mn -= 0.1 * rng; rng = mx - mn
            val = ((h - self.padding - y) / (h - 2 * self.padding)) * rng + mn

            # update single point in result
            self.seq[self.selected_idx] = val

            # update each formula panel only at that index
            for sp in self.sync_panels:
                if sp.original_seq is not None and sp.formula:
                    try:
                        sp.seq[self.selected_idx] = safe_eval(
                            sp.formula,
                            {'orig': sp.original_seq[self.selected_idx], 'res': val}
                        )
                    except:
                        pass
                    sp.Refresh()
            self.Refresh()
        elif self._pan_start is not None and evt.MiddleIsDown():
            dx = evt.GetX() - self._pan_start
            self.pan_offset = self._pan_origin + dx * 2
            w, _ = self.GetClientSize()
            gw = w - 2 * self.padding
            min_off = gw * (1 - self.zoom_factor)
            self.pan_offset = min(max(self.pan_offset, min_off), 0)
            for sp in [self] + self.sync_panels:
                sp.pan_offset = self.pan_offset
                sp.Refresh()
            self.Refresh()

    def on_left_up(self, evt):
        if self.dragging:
            self.dragging = False
            try:
                self.ReleaseMouse()
            except:
                pass
            self.Refresh()

    def on_middle_down(self, evt):
        self._pan_start = evt.GetX()
        self._pan_origin = self.pan_offset

    def on_middle_up(self, evt):
        self._pan_start = None

    def on_mouse_wheel(self, evt):
        if not self.draggable:
            return
        rot = evt.GetWheelRotation() / evt.GetWheelDelta()
        f = 1 + rot * 0.1
        old_z = self.zoom_factor
        self.zoom_factor = max(1.0, min(old_z * f, 10))

        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        mx = evt.GetX() - self.padding - self.pan_offset
        self.pan_offset -= mx * (self.zoom_factor / old_z - 1)
        min_off = gw * (1 - self.zoom_factor)
        self.pan_offset = min(max(self.pan_offset, min_off), 0)

        for sp in [self] + self.sync_panels:
            sp.zoom_factor = self.zoom_factor
            sp.pan_offset = self.pan_offset
            sp.Refresh()
        self.Refresh()