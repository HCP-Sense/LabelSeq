import wx
import numpy as np

# -----------------------------
# Basic signal-drawing panel
# -----------------------------
class SignalPanel(wx.Panel):
    def __init__(self, parent, signal, vmin, vmax, title="",
                 line_color=(0, 0, 0), draw_points=False, point_color=(0, 0, 0)):
        super().__init__(parent)
        self.signal = np.asarray(signal, dtype=float).copy()
        self.n = len(self.signal)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.title = title
        self.padding = 34

        # create pens/brushes here (safe after wx.App exists)
        self.line_color = wx.Colour(*line_color) if isinstance(line_color, tuple) else line_color
        self.pen = wx.Pen(self.line_color, 2)
        self.draw_points = draw_points
        self.point_brush = wx.Brush(wx.Colour(*point_color)) if isinstance(point_color, tuple) else wx.Brush(point_color)

        self.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)

    def set_signal(self, new_signal):
        self.signal = np.asarray(new_signal, dtype=float).copy()
        self.n = len(self.signal)
        self.Refresh()

    def on_size(self, evt):
        self.Refresh()
        evt.Skip()

    def _map_index_to_x(self, i, w):
        graph_w = max(1, w - 2 * self.padding)
        return int(self.padding + (i * graph_w) / max(1, (self.n - 1)))

    def _map_value_to_y(self, v, h):
        graph_h = max(1, h - 2 * self.padding)
        # clamp to avoid div by zero
        rng = self.vmax - self.vmin if (self.vmax - self.vmin) != 0 else 1.0
        norm = (v - self.vmin) / rng
        # invert y: higher value -> smaller y
        return int(self.padding + (1.0 - norm) * graph_h)

    def on_paint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.Clear()
        w, h = self.GetClientSize()

        # Title
        dc.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        dc.SetTextForeground(wx.BLACK)
        dc.DrawText(self.title, 6, 6)

        # Draw box for the graph area
        dc.SetPen(wx.Pen(wx.LIGHT_GREY, 1))
        dc.DrawRectangle(self.padding, self.padding, w - 2 * self.padding, h - 2 * self.padding)

        if self.signal is None or self.n < 2:
            return

        # Build points
        pts = [(self._map_index_to_x(i, w), self._map_value_to_y(v, h)) for i, v in enumerate(self.signal)]

        # Draw line
        dc.SetPen(self.pen)
        if len(pts) >= 2:
            dc.DrawLines(pts)

        # Optionally draw points as circles
        if self.draw_points:
            dc.SetBrush(self.point_brush)
            for (x, y) in pts:
                dc.DrawCircle(x, y, 5)


# ---------------------------------------
# Annotator: draggable copy of Original
# ---------------------------------------
class AnnotatorPanel(SignalPanel):
    def __init__(self, parent, signal, vmin, vmax, on_update_callback,
                 title="Annotator (drag points)", line_color=(0, 120, 0), point_color=(0, 180, 0)):
        super().__init__(parent, signal, vmin, vmax, title=title,
                         line_color=line_color, draw_points=True, point_color=point_color)
        self.on_update = on_update_callback
        self._drag_idx = None
        self._hit_radius = 8  # pixels

        # bind mouse events for dragging
        self.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self._on_left_up)
        self.Bind(wx.EVT_MOTION, self._on_motion)

    def _find_hit_index(self, pos):
        x0, y0 = pos
        w, h = self.GetClientSize()
        best = None
        best_d2 = (self._hit_radius + 1) ** 2
        for i, v in enumerate(self.signal):
            xi = self._map_index_to_x(i, w)
            yi = self._map_value_to_y(v, h)
            d2 = (xi - x0) ** 2 + (yi - y0) ** 2
            if d2 <= best_d2:
                best = i
                best_d2 = d2
        return best

    def _on_left_down(self, evt):
        i = self._find_hit_index(evt.GetPosition())
        if i is not None:
            self._drag_idx = i
            self.CaptureMouse()

    def _on_left_up(self, evt):
        if self.HasCapture():
            self.ReleaseMouse()
        self._drag_idx = None

    def _on_motion(self, evt):
        if self._drag_idx is None:
            return
        if not (evt.Dragging() and evt.LeftIsDown()):
            return
        # Convert y -> value and update that single index
        _, y = evt.GetPosition()
        w, h = self.GetClientSize()
        # clamp y to graph bounds
        y = max(self.padding, min(h - self.padding, y))
        graph_h = max(1, h - 2 * self.padding)
        frac = (y - self.padding) / graph_h  # 0 .. 1 (top->bottom)
        new_val = self.vmax - frac * (self.vmax - self.vmin)
        self.signal[self._drag_idx] = float(np.clip(new_val, self.vmin, self.vmax))
        self.Refresh()
        # callback with a copy
        if callable(self.on_update):
            self.on_update(self.signal.copy())


# ---------------------------------------
# Annotation panel: Original - Annotator
# ---------------------------------------
class AnnotationPanel(SignalPanel):
    def __init__(self, parent, signal, vmin, vmax, title="Annotation (Original - Annotator)",
                 line_color=(0, 80, 200)):
        super().__init__(parent, signal, vmin, vmax, title=title, line_color=line_color, draw_points=False)


# ---------------------------------------
# Result panel: Original + filled area
# ---------------------------------------
class ResultPanel(wx.Panel):
    def __init__(self, parent, original_signal, annotator_signal, vmin, vmax,
                 title="Result (Original with annotation area filled)"):
        super().__init__(parent)
        self.original = np.asarray(original_signal, dtype=float).copy()
        self.annotator = np.asarray(annotator_signal, dtype=float).copy()
        self.n = len(self.original)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.title = title
        self.padding = 34

        self.line_pen = wx.Pen(wx.Colour(10, 60, 160), 2)  # original line (blue)
        self.fill_brush = wx.Brush(wx.Colour(220, 60, 60))  # fill color (red)
        self.SetBackgroundColour(wx.Colour(255, 255, 255))

        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)

    def set_annotator(self, annotator_signal):
        self.annotator = np.asarray(annotator_signal, dtype=float).copy()
        self.Refresh()

    def _map_index_to_x(self, i, w):
        graph_w = max(1, w - 2 * self.padding)
        return int(self.padding + (i * graph_w) / max(1, (self.n - 1)))

    def _map_value_to_y(self, v, h):
        graph_h = max(1, h - 2 * self.padding)
        rng = self.vmax - self.vmin if (self.vmax - self.vmin) != 0 else 1.0
        norm = (v - self.vmin) / rng
        return int(self.padding + (1.0 - norm) * graph_h)

    def on_size(self, evt):
        self.Refresh()
        evt.Skip()

    def on_paint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.Clear()
        w, h = self.GetClientSize()

        # Title
        dc.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        dc.SetTextForeground(wx.BLACK)
        dc.DrawText(self.title, 6, 6)

        if self.original is None or self.annotator is None:
            return
        if len(self.original) < 2:
            return

        orig_pts = [(self._map_index_to_x(i, w), self._map_value_to_y(v, h)) for i, v in enumerate(self.original)]
        ann_pts = [(self._map_index_to_x(i, w), self._map_value_to_y(v, h)) for i, v in enumerate(self.annotator)]

        # Build polygon: original left->right then annotator right->left
        poly = []
        poly.extend(orig_pts)
        poly.extend(reversed(ann_pts))

        # Draw filled polygon (covers the area between original and annotator)
        dc.SetBrush(self.fill_brush)
        dc.SetPen(wx.Pen(self.fill_brush.GetColour(), 1))
        try:
            dc.DrawPolygon(poly)
        except Exception:
            # fallback: draw small quads between successive indices
            for i in range(self.n - 1):
                quad = [orig_pts[i], orig_pts[i + 1], ann_pts[i + 1], ann_pts[i]]
                dc.DrawPolygon(quad)

        # Draw original on top
        dc.SetPen(self.line_pen)
        dc.DrawLines(orig_pts)


# --------------------------
# Main application frame
# --------------------------
class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Annotator — Original / Annotation / Annotator / Result", size=(1000, 900))

        # Example original signal
        np.random.seed(0)
        n = 300
        t = np.linspace(0, 6.0 * np.pi, n)
        original = 0.9 * np.sin(1.6 * t) + 0.3 * np.sin(3.2 * t + 0.5) + 0.12 * np.random.randn(n)

        # Annotator starts as a copy of original (so annotation = original - annotator => zeros)
        annotator = original.copy()
        annotation = original - annotator  # zeros initially

        # Compute shared vmin/vmax so panels align vertically
        all_vals = np.hstack([original, annotator, annotation])
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        # add small margin
        margin = (vmax - vmin) * 0.08 if (vmax - vmin) != 0 else 1.0
        vmin -= margin
        vmax += margin

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        lbl = wx.StaticText(panel, label="Signal Annotator (4 panels): Original | Annotation | Annotator (draggable) | Result")
        lbl.SetFont(wx.Font(11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        vbox.Add(lbl, 0, wx.ALL | wx.EXPAND, 6)

        # Original (top) - black line
        self.original_panel = SignalPanel(panel, original, vmin, vmax, title="Original (static)", line_color=(0, 0, 0))
        vbox.Add(self.original_panel, 1, wx.EXPAND | wx.ALL, 6)

        # Annotation (second) - show original - annotator (initially zeros straight line)
        self.annotation_panel = AnnotationPanel(panel, annotation, vmin, vmax, title="Annotation (Original - Annotator)", line_color=(0, 80, 200))
        vbox.Add(self.annotation_panel, 1, wx.EXPAND | wx.ALL, 6)

        # Annotator (third) - draggable copy of original
        def annotator_updated(new_annotator_signal):
            # recompute annotation and update annotation panel and result panel
            nonlocal annotation, annotator
            annotator = np.asarray(new_annotator_signal, dtype=float).copy()
            annotation = original - annotator
            self.annotation_panel.set_signal(annotation)
            self.result_panel.set_annotator(annotator)

        self.annotator_panel = AnnotatorPanel(panel, annotator, vmin, vmax, on_update_callback=annotator_updated,
                                              title="Annotator (draggable)", line_color=(0, 140, 0), point_color=(0, 200, 0))
        vbox.Add(self.annotator_panel, 1, wx.EXPAND | wx.ALL, 6)

        # Result (bottom) - original with area between original and annotator filled (red)
        self.result_panel = ResultPanel(panel, original_signal=original, annotator_signal=annotator, vmin=vmin, vmax=vmax,
                                        title="Result (Original with annotation area filled)")
        vbox.Add(self.result_panel, 1, wx.EXPAND | wx.ALL, 6)

        panel.SetSizer(vbox)
        self.Centre()
        self.Show()


if __name__ == "__main__":
    app = wx.App(False)
    MainFrame()
    app.MainLoop()
