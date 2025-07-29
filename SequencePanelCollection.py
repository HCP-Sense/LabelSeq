import wx
from FormulaParser import safe_eval  # Your formula safe evaluation utility

class SequencePanel(wx.Panel):
    def __init__(self, parent, sequence, label, formula=None,
                 draggable=False, visible_count=None, color=wx.BLUE):
        super().__init__(parent, style=wx.NO_FULL_REPAINT_ON_RESIZE)

        # Core data
        self.seq = list(sequence)  # The modifiable sequence
        self.original_seq = list(sequence)  # The unmodified original sequence
        self.title = label
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

        # Interaction state
        self.dragging = False
        self.selected_idx = None
        self.hover_idx = None
        self._pan_start = None
        self._pan_origin = 0

        # Panels to synchronize zoom/pan and hover
        self.sync_panels = []

        # Setup UI controls (title, buttons)
        self._setup_ui()

        # Bind events for drawing and interaction
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: (self.Refresh(), e.Skip()))
        self.Bind(wx.EVT_MOTION, self.on_hover_move)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_mouse_leave)
        self.Bind(wx.EVT_LEFT_DCLICK, self.on_double_click)

        if self.draggable:
            self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
            self.Bind(wx.EVT_MOTION, self.on_drag)
            self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
            self.Bind(wx.EVT_MIDDLE_DOWN, self.on_middle_down)
            self.Bind(wx.EVT_MIDDLE_UP, self.on_middle_up)
            self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)

    def _setup_ui(self):
        """Creates the title and buttons UI at the top"""
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.title_label = wx.StaticText(self, label=self.title, style=wx.ALIGN_CENTER)
        title_font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.title_label.SetFont(title_font)
        btn_sizer.Add(self.title_label, 1, wx.ALIGN_CENTER_VERTICAL)

        btn_title = wx.Button(self, label="Edit Title", size=(90, -1))
        btn_title.Bind(wx.EVT_BUTTON, self.edit_title)
        btn_sizer.Add(btn_title, 0, wx.LEFT, 5)

        if self.formula:
            btn_formula = wx.Button(self, label="Edit Formula", size=(100, -1))
            btn_formula.Bind(wx.EVT_BUTTON, self.edit_formula)
            btn_sizer.Add(btn_formula, 0, wx.LEFT, 5)

        btn_delete = wx.Button(self, label="Delete", size=(70, -1))
        btn_delete.Bind(wx.EVT_BUTTON, self.delete_self)
        btn_sizer.Add(btn_delete, 0, wx.LEFT, 5)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(main_sizer)

    def edit_title(self, evt):
        dlg = wx.TextEntryDialog(self, "Enter new title:", "Edit Title", self.title)
        if dlg.ShowModal() == wx.ID_OK:
            self.title = dlg.GetValue()
            self.title_label.SetLabel(self.title)
            self.Refresh()
        dlg.Destroy()

    def edit_formula(self, evt):
        dlg = wx.TextEntryDialog(self, "Enter new formula:", "Edit Formula", self.formula or "")
        if dlg.ShowModal() == wx.ID_OK:
            self.formula = dlg.GetValue()
            self.Refresh()
        dlg.Destroy()

    def delete_self(self, evt):
        top = wx.GetTopLevelParent(self)
        if hasattr(top, 'remove_panel'):
            top.remove_panel(self)

    def get_y_range(self):
        """Returns combined min/max range of original and current sequence to scale overlay & main identically"""
        if self.original_seq is not None and len(self.original_seq) == self.n:
            combined = self.seq + self.original_seq
            mx = max(combined)
            mn = min(combined)
            if mx == mn:
                mx += 1
                mn -= 1
            else:
                # Add 10% margin
                rng = mx - mn
                mx += 0.1 * rng
                mn -= 0.1 * rng
            return mn, mx
        else:
            # Fallback to seq only
            mx = max(self.seq)
            mn = min(self.seq)
            if mx == mn:
                mx += 1
                mn -= 1
            else:
                rng = mx - mn
                mx += 0.1 * rng
                mn -= 0.1 * rng
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
            pass  # Just continue if font fails

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

    def on_hover_move(self, evt):
        x, _ = evt.GetPosition()
        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        step = gw / (self.n - 1) * self.zoom_factor
        idx = int((x - self.padding - self.pan_offset) / step + 0.5)
        self.hover_idx = max(0, min(idx, self.n - 1))
        self.Refresh()
        evt.Skip()

    def on_mouse_leave(self, evt):
        self.hover_idx = None
        self.Refresh()
        evt.Skip()

    def on_double_click(self, evt):
        dlg = wx.TextEntryDialog(self, "Enter new title:", "Rename", self.title)
        if dlg.ShowModal() == wx.ID_OK:
            self.title = dlg.GetValue()
            self.title_label.SetLabel(self.title)
            self.Refresh()
        dlg.Destroy()

        if self.formula:
            dlg2 = wx.TextEntryDialog(self, "Enter new formula:", "Edit Formula", self.formula)
            if dlg2.ShowModal() == wx.ID_OK:
                self.formula = dlg2.GetValue()
                self.Refresh()
            dlg2.Destroy()

    def on_left_down(self, evt):
        if not self.draggable:
            return
        x0, y0 = evt.GetPosition()
        w, h = self.GetClientSize()
        best_dist = float('inf')
        best_idx = None
        for i, v in enumerate(self.seq):
            cx, cy = self.to_px(i, v, w, h)
            dist = (cx - x0)**2 + (cy - y0)**2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None and best_dist < (self.radius * 3)**2:
            self.selected_idx = best_idx
            self.dragging = True
            self.CaptureMouse()

    def on_drag(self, evt):
        if not (self.draggable and self.dragging and evt.LeftIsDown()):
            return
        x, y = evt.GetPosition()
        w, h = self.GetClientSize()
        mn, mx = self.get_y_range()
        rng = mx - mn or 1

        # Calculate new value from mouse Y pos
        val = ((h - self.padding - y) / (h - 2 * self.padding)) * rng + mn
        # Clamp val within original y range
        val = max(min(val, mx), mn)

        self.seq[self.selected_idx] = val

        # Propagate changes to synchronized panels with formula
        for sp in self.sync_panels:
            if sp.original_seq is not None and sp.formula:
                try:
                    sp.seq[self.selected_idx] = safe_eval(
                        sp.formula,
                        {'orig': sp.original_seq[self.selected_idx], 'res': val}
                    )
                except Exception:
                    pass
                sp.Refresh()

        self.Refresh()

    def on_left_up(self, evt):
        if self.dragging:
            self.dragging = False
            self.ReleaseMouse()
            self.Refresh()

    def on_middle_down(self, evt):
        self._pan_start = evt.GetX()
        self._pan_origin = self.pan_offset

    def on_middle_up(self, evt):
        self._pan_start = None

    def on_mouse_wheel(self, evt):
        if not self.draggable:
            return
        rotation = evt.GetWheelRotation() / evt.GetWheelDelta()
        factor = 1 + rotation * 0.1
        old_zoom = self.zoom_factor
        self.zoom_factor = max(1.0, min(old_zoom * factor, 10))

        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        mouse_x = evt.GetX() - self.padding - self.pan_offset

        # Adjust pan to zoom on mouse position
        self.pan_offset -= mouse_x * (self.zoom_factor / old_zoom - 1)

        # Clamp pan offset
        min_offset = gw * (1 - self.zoom_factor)
        self.pan_offset = min(max(self.pan_offset, min_offset), 0)

        # Sync zoom/pan with linked panels
        for sp in self.sync_panels:
            sp.zoom_factor = self.zoom_factor
            sp.pan_offset = self.pan_offset
            sp.Refresh()
        self.Refresh()
