__author___ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com""
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Moye Nyuysoni Glein Perry"

# Third-Party Imports
import wx


class SequencePanel(wx.Panel):
    def __init__(self, parent, sequence, title: str, formula=None,
                 draggable: bool = False, visible_count=None, color=wx.BLUE):
        super().__init__(parent, style=wx.NO_FULL_REPAINT_ON_RESIZE)

        # Core data
        self.seq = list(sequence)  # The modifiable sequence
        self.original_seq = list(sequence)  # The unmodified original sequence
        self.title = title
        self.formula = formula
        self.color = color
        self.draggable = draggable
        self.n = len(self.seq)
        self.visible_count = visible_count if visible_count else self.n

        # Visualization parameters
        self.zoom_factor = max(1.0, (self.n - 1) / self.visible_count)
        self.pan_offset = 0
        self.padding = 80
        self.radius = 6

        # Interaction state placeholders (to be handled in subclass)
        self.dragging = False
        self.selected_idx = None
        self.hover_idx = None
        self._pan_start = None
        self._pan_origin = 0

        # Panels to synchronize zoom/pan and hover
        self.sync_panels = []

        # Bind minimal events for painting and resizing
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: (self.Refresh(), e.Skip()))

    def get_y_range(self):
        """Returns combined min/max range of original and current sequence to scale overlay & main identically"""
        margin_factor = 0.15  # 15% margin
        min_margin = 1.0  # minimum fixed margin

        if self.original_seq is not None and len(self.original_seq) == self.n:
            combined = self.seq + self.original_seq
            mx = max(combined)
            mn = min(combined)
            if mx == mn:
                mx += 1
                mn -= 1
            else:
                rng = mx - mn
                margin = max(margin_factor * rng, min_margin)
                mx += margin
                mn -= margin
            return mn, mx
        else:
            mx = max(self.seq)
            mn = min(self.seq)
            if mx == mn:
                mx += 1
                mn -= 1
            else:
                rng = mx - mn
                margin = max(margin_factor * rng, min_margin)
                mx += margin
                mn -= margin
            return mn, mx

    def to_px(self, i, v, w, h, mn=None, mx=None):
        """Convert data index & value to pixel coordinates on panel"""
        if mn is None or mx is None:
            mn, mx = self.get_y_range()
        rng = mx - mn
        gw = w - 2 * self.padding
        gh = h - 2 * self.padding
        sx = gw / (self.n - 1) * self.zoom_factor
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

        # Setup font safely
        try:
            f_wx = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
            f_gc = gc.CreateFont(f_wx, wx.BLACK)
            gc.SetFont(f_gc)
        except Exception:
            pass  # Continue if font fails

        # Background white
        gc.SetBrush(wx.Brush(wx.WHITE))
        gc.DrawRectangle(0, 0, w, h)

        # Draw axes lines
        gc.SetPen(wx.Pen(wx.LIGHT_GREY))
        gc.StrokeLine(self.padding, self.padding, self.padding, h - self.padding)  # Y axis
        gc.StrokeLine(self.padding, h - self.padding, w - self.padding, h - self.padding)  # X axis

        # Draw Y ticks and labels using combined range
        mn, mx = self.get_y_range()
        rng = mx - mn
        step_val = rng / 5
        for k in range(6):
            val = mn + k * step_val
            y = h - self.padding - (val - mn) * ((h - 2 * self.padding) / rng)
            gc.StrokeLine(self.padding - 5, y, self.padding, y)
            txt = f"{val:.1f}"
            tw, th = gc.GetTextExtent(txt)
            gc.DrawText(txt, self.padding - 10 - tw, y - th / 2)

        # Draw X ticks and labels
        step_idx = max(1, self.n // 10)
        for i in range(0, self.n, step_idx):
            x, _ = self.to_px(i, mn, w, h, mn, mx)
            if self.padding <= x <= w - self.padding:
                gc.StrokeLine(x, h - self.padding, x, h - self.padding + 5)
                lbl = str(i)
                tw, th = gc.GetTextExtent(lbl)
                gc.DrawText(lbl, x - tw / 2, h - self.padding + 8)

        # Draw title centered top
        f2 = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        fg = gc.CreateFont(f2, wx.BLACK)
        gc.SetFont(fg)
        tw, _ = gc.GetTextExtent(self.title)
        gc.DrawText(self.title, (w - tw) / 2, 5)

        # Clip plot area
        gc.Clip(self.padding, self.padding, w - 2 * self.padding, h - 2 * self.padding)

        # Draw original sequence as dotted blue overlay
        if self.original_seq is not None and len(self.original_seq) == self.n:
            orig_pts = [self.to_px(i, v, w, h, mn, mx) for i, v in enumerate(self.original_seq)]
            path0 = gc.CreatePath()
            path0.MoveToPoint(*orig_pts[0])
            for pt in orig_pts[1:]:
                path0.AddLineToPoint(*pt)
            gc.SetPen(wx.Pen(wx.BLUE, 1, wx.PENSTYLE_DOT))
            gc.StrokePath(path0)

        # Draw main sequence line
        pts = [self.to_px(i, v, w, h, mn, mx) for i, v in enumerate(self.seq)]
        path1 = gc.CreatePath()
        path1.MoveToPoint(*pts[0])
        for pt in pts[1:]:
            path1.AddLineToPoint(*pt)
        gc.SetPen(wx.Pen(self.color, 1))
        gc.StrokePath(path1)

        # Draw draggable handle for selected point (only for draggable panel)
        if self.draggable and self.selected_idx is not None:
            x, y = pts[self.selected_idx]
            gc.SetBrush(wx.Brush(self.color))
            gc.DrawEllipse(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius)

        # Draw hover dot on this panel
        if self.hover_idx is not None and 0 <= self.hover_idx < self.n:
            x, y = pts[self.hover_idx]
            gc.SetBrush(wx.Brush(self.color))
            gc.DrawEllipse(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius)

        # Sync hover to other panels
        for sp in self.sync_panels:
            if sp.hover_idx != self.hover_idx:
                sp.hover_idx = self.hover_idx
                sp.Refresh()
