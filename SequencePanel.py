# =============================================================================
# SequencePanel.py  —  CLEAN BASE CLASS (2025)
# =============================================================================
"""
Base class for fast sequence visualization.
InteractiveSequencePanel extends this class.

This file contains ONLY:
  • viewport logic (zoom, pan, visible range)
  • coordinate transforms
  • basic line/axes drawing
  • hover broadcasting
  • sync panel support

NO annotation, NO spans, NO active learning.
"""

import wx
import numpy as np


class SequencePanel(wx.Panel):
    """
    Lightweight rendering engine for drawing sequences.
    Advanced interaction is implemented in InteractiveSequencePanel.
    """

    def __init__(self, parent, sequence, title: str = "",
                 formula=None, draggable=False,
                 visible_count=None, color=wx.BLUE):

        super().__init__(parent, style=wx.NO_FULL_REPAINT_ON_RESIZE)

        # ---- Core Data ----
        self.seq = np.asarray(sequence, dtype=float).flatten()
        self.original_seq = self.seq.copy()            # Used by derived panels
        self.title = title
        self.formula = formula
        self.color = color
        self.draggable = draggable
        self.n = len(self.seq)

        # ---- Viewport Parameters ----
        self.padding = 80
        self.zoom_factor = 1.0
        self.pan_offset = 0.0        # X-pan in pixels
        self.y_offset = 0.0          # Y-pan in value-units
        self.visible_count = visible_count or self.n

        # ---- Hover & Sync ----
        self.hover_idx = None
        self.selected_idx = None
        self.sync_panels = []        # list of SequencePanels
        self._dragging_pan = False
        self._pan_origin = 0

        # ---- Events ----
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, lambda evt: (self.Refresh(False), evt.Skip()))

    # =====================================================================
    # VISIBLE RANGE CALCULATION
    # =====================================================================
    def get_visible_index_range(self):
        """Returns (left_idx, right_idx) based on zoom and X-pan."""
        n = self.n
        if n <= 1:
            return 0, 1

        # number of indices visible under current zoom
        visible = int((self.n - 1) / max(self.zoom_factor, 1e-9))
        visible = max(1, visible)

        # pixel → index conversion
        w = max(1, self.GetClientSize().width)
        plot_w = max(1, w - 2 * self.padding)
        px_per_idx = (plot_w / max(1, (n - 1))) * self.zoom_factor

        # convert pan offset to index shift
        index_shift = int(round(-self.pan_offset / max(px_per_idx, 1e-9)))

        left = max(0, index_shift)
        right = min(n, left + visible)

        if right <= left:
            right = min(n, left + 1)

        return left, right

    # =====================================================================
    # Y-RANGE + VALUE OFFSET
    # =====================================================================
    def get_y_range(self):
        """Returns (min_y, max_y) BEFORE applying y_offset."""
        if self.original_seq is None or len(self.original_seq) != self.n:
            arr = self.seq
        else:
            arr = np.concatenate([self.seq, self.original_seq])

        mn = float(arr.min())
        mx = float(arr.max())
        if mx == mn:
            mx += 1
            mn -= 1

        rng = mx - mn
        margin = 0.15 * rng
        return mn - margin, mx + margin

    def get_y_display_range(self):
        """Range AFTER applying vertical offset."""
        mn, mx = self.get_y_range()
        return mn + self.y_offset, mx + self.y_offset

    # =====================================================================
    # COORDINATE TRANSFORM
    # =====================================================================
    def to_px(self, idx, value, w, h, mn=None, mx=None):
        """Convert (index, value) → (x_px, y_px)."""
        if mn is None or mx is None:
            mn, mx = self.get_y_display_range()
        rng = mx - mn

        gw = w - 2 * self.padding
        gh = h - 2 * self.padding

        sx = (gw / max(1, (self.n - 1))) * self.zoom_factor
        sy = gh / max(rng, 1e-9)

        x = self.padding + idx * sx + self.pan_offset
        y = h - self.padding - ((value - mn) * sy)

        return x, y

    # =====================================================================
    # BASIC PAINT (Override in subclasses)
    # =====================================================================
    def _on_paint(self, evt):
        """Base drawing: axes + polyline."""
        w, h = self.GetClientSize()
        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)

        # Background
        gc.SetBrush(wx.Brush(wx.WHITE))
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.DrawRectangle(0, 0, w, h)

        # Axes
        gc.SetPen(wx.Pen(wx.Colour(200, 200, 200), 1))
        gc.StrokeLine(self.padding, self.padding,
                      self.padding, h - self.padding)
        gc.StrokeLine(self.padding, h - self.padding,
                      w - self.padding, h - self.padding)

        # Compute visible range
        left, right = self.get_visible_index_range()
        if right - left <= 1:
            return

        mn, mx = self.get_y_display_range()

        # Decimation
        count = right - left
        if count > 1800:
            idxs = np.linspace(left, right - 1, 1800).astype(int)
        else:
            idxs = np.arange(left, right)

        ys = self.seq[idxs]

        # Line color
        gc.SetPen(wx.Pen(self.color, 2))

        # Build polyline
        path = gc.CreatePath()
        x0, y0 = self.to_px(idxs[0], ys[0], w, h, mn, mx)
        path.MoveToPoint(x0, y0)
        for i, v in zip(idxs[1:], ys[1:]):
            x, y = self.to_px(i, v, w, h, mn, mx)
            path.AddLineToPoint(x, y)

        gc.StrokePath(path)

    # =====================================================================
    # SYNC SYSTEM
    # =====================================================================
    def add_sync_panel(self, other):
        if other not in self.sync_panels:
            self.sync_panels.append(other)

    def broadcast_hover(self, idx):
        """Send hover index to all sync panels."""
        for p in self.sync_panels:
            p.hover_idx = idx
            p.Refresh(False)
        self.hover_idx = idx
        self.Refresh(False)

    # =====================================================================
    # ALLOW INTERACTIVE CHILDREN TO CALL REFRESH SAFELY
    # =====================================================================
    def request_refresh(self):
        """Schedules a safe refresh (subclasses may override)."""
        wx.CallAfter(self.Refresh, False)

