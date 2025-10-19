"""InteractiveSequencePanel.py (Option A: Full UI)
SequencePanel subclass adding roles (original, annotator, annotation, result),
draggable annotator points, filled result area, highlights, spans, and pan/zoom.
"""

__author__ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Development"

from SequencePanel import SequencePanel
from FormulaParser import safe_eval

import wx
import numpy as np
from typing import Optional, Callable, Any, Dict, List


class InteractiveSequencePanel(SequencePanel):
    """
    Extended SequencePanel with roles:
      - role="original"   : static original (no edit title/delete)
      - role="annotator"  : draggable points; call on_update(seq) when changed
      - role="annotation" : shows (original - annotator)
      - role="result"     : shows original and filled area between original and annotator; highlights
    """

    # ---------------------------
    # Construction
    # ---------------------------
    def __init__(self,
                 parent,
                 sequence,
                 title: str,
                 formula: Optional[str] = None,
                 draggable: bool = False,
                 visible_count: Optional[int] = None,
                 color=wx.BLUE,
                 role: str = "result",
                 on_update: Optional[Callable[[np.ndarray], Any]] = None,
                 allow_edit_title: bool = True,
                 allow_delete: bool = True):
        super().__init__(parent, sequence, title, formula, draggable, visible_count, color)

        # Role + callbacks
        self.role = (role or "result").lower()
        # Keep both names for compatibility
        self.on_update = on_update
        self.on_update_callback = on_update

        # allow flags (UI)
        self.allow_edit_title = allow_edit_title
        self.allow_delete = allow_delete

        # External references
        self.annotator_ref = None  # set by main on result/annotation panels when annotator exists
        self.original_seq = getattr(self, "original_seq", None)

        # Legend mapping: label -> wx.Colour or RGB tuple
        self.annotation_legend: Dict[str, Any] = {}

        # Interaction state (safe defaults for all roles)
        self._undo_stack: List[np.ndarray] = []
        self._redo_stack: List[np.ndarray] = []
        self.play_idx: int = 0
        self.spans: List[Dict[str, Any]] = []  # {"start":int, "end":int, "label":str|None, "value":float|None}
        self._span_edit = None         # {"index":int, "which":"left"|"right"|"body", "last_x":px}
        self._span_idx0 = None         # temp start index during Shift+drag creation
        self._has_focus = False
        self._space_pan_active = False
        self._hit_radius = max(6, int(self.radius * 1.5)) if hasattr(self, "radius") else 8

        # Top UI (buttons) — only in Option A
        self._setup_ui()

        # Event bindings
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_mouse_leave)
        self.Bind(wx.EVT_LEFT_DCLICK, self.on_double_click)
        self.Bind(wx.EVT_MOTION, self.on_mouse_motion)

        if self.role == "annotator" or self.draggable:
            self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
            self.Bind(wx.EVT_LEFT_UP, self.on_left_up)

        self.Bind(wx.EVT_MIDDLE_DOWN, self.on_middle_down)
        self.Bind(wx.EVT_MIDDLE_UP, self.on_middle_up)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_middle_down)   # right as pan start
        self.Bind(wx.EVT_RIGHT_UP, self.on_middle_up)

        self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)

        # Focus + keyboard
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(wx.EVT_SET_FOCUS, self.on_focus)
        self.Bind(wx.EVT_KILL_FOCUS, self.on_kill_focus)
        try:
            self.SetFocusable(True)
        except Exception:
            pass

    # ---------------------------
    # UI (title row)
    # ---------------------------
    def _setup_ui(self):
        top = self.GetSizer()
        if top is None:
            top = wx.BoxSizer(wx.VERTICAL)
            self.SetSizer(top)
        top.Clear()

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.title_label = wx.StaticText(self, label=self.title, style=wx.ALIGN_CENTER)
        self.title_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        row.Add(self.title_label, 1, wx.ALIGN_CENTER_VERTICAL)

        if self.allow_edit_title and self.role != "original":
            btn_title = wx.Button(self, label="Edit Title", size=(90, -1))
            btn_title.Bind(wx.EVT_BUTTON, self.edit_title)
            row.Add(btn_title, 0, wx.LEFT, 5)

        if self.formula:
            btn_formula = wx.Button(self, label="Edit Formula", size=(100, -1))
            btn_formula.Bind(wx.EVT_BUTTON, self.edit_formula)
            row.Add(btn_formula, 0, wx.LEFT, 5)

        if self.allow_delete and self.role not in ("original", "result"):
            btn_delete = wx.Button(self, label="Delete", size=(70, -1))
            btn_delete.Bind(wx.EVT_BUTTON, self.delete_self)
            row.Add(btn_delete, 0, wx.LEFT, 5)

        top.Add(row, 0, wx.EXPAND | wx.ALL, 5)
        self.Layout()

    # ---------------------------
    # Title / Formula / Delete
    # ---------------------------
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

    # ---------------------------
    # Focus handlers
    # ---------------------------
    def on_focus(self, evt):
        self._has_focus = True
        try:
            evt.Skip()
        except Exception:
            pass

    def on_kill_focus(self, evt):
        self._has_focus = False
        try:
            evt.Skip()
        except Exception:
            pass

    # ---------------------------
    # Undo/Redo helpers
    # ---------------------------
    def _push_undo(self):
        if self.role == "annotator":
            try:
                self._undo_stack.append(np.asarray(self.seq, dtype=float).copy())
                if len(self._undo_stack) > 100:
                    self._undo_stack.pop(0)
                self._redo_stack.clear()
            except Exception:
                pass

    def _undo(self):
        if self.role != "annotator" or not self._undo_stack:
            return
        prev = self._undo_stack.pop()
        self._redo_stack.append(np.asarray(self.seq, dtype=float).copy())
        self.seq = prev.copy()
        cb = getattr(self, "on_update", None)
        if callable(cb):
            try:
                cb(self.seq.copy())
            except Exception:
                pass
        self.Refresh()

    def _redo(self):
        if self.role != "annotator" or not self._redo_stack:
            return
        nxt = self._redo_stack.pop()
        self._undo_stack.append(np.asarray(self.seq, dtype=float).copy())
        self.seq = nxt.copy()
        cb = getattr(self, "on_update", None)
        if callable(cb):
            try:
                cb(self.seq.copy())
            except Exception:
                pass
        self.Refresh()

    # ---------------------------
    # Span helpers & hit testing
    # ---------------------------
    def _x_to_index(self, x_px, w):
        plot_w = max(1, w - 2 * self.padding)
        rel = (x_px - self.padding - self.pan_offset) / (plot_w * self.zoom_factor)
        left, right = self.get_visible_index_range()
        span = max(1, right - left)
        rel = max(0.0, min(1.0, rel))
        return int(left + rel * span)

    def _normalize_span(self, s: Dict[str, Any]):
        a, b = int(s.get("start", 0)), int(s.get("end", 0))
        if b < a:
            a, b = b, a
        a = max(0, min(self.n - 1, a))
        b = max(0, min(self.n - 1, b))
        s["start"], s["end"] = a, b
        return s

    def _hit_span_edge(self, x_px, y_px, w, h, tol=4):
        # returns (index, which) where which in {"left","right","body"} or (None,None)
        left_idx, right_idx = self.get_visible_index_range()
        for i, s in enumerate(self.spans or []):
            a, b = int(s.get("start", 0)), int(s.get("end", 0))
            if b < a:
                a, b = b, a
            if b <= left_idx or a >= right_idx:
                continue
            ax, _ = self.to_px(a, self.seq[a if 0 <= a < self.n else 0], w, h)
            bx, _ = self.to_px(b, self.seq[b if 0 <= b < self.n else self.n - 1], w, h)
            if abs(x_px - ax) <= tol:
                return i, "left"
            if abs(x_px - bx) <= tol:
                return i, "right"
            if min(ax, bx) + tol < x_px < max(ax, bx) - tol:
                return i, "body"
        return None, None

    # ---------------------------
    # Mouse / interaction handlers
    # ---------------------------
    def on_mouse_motion(self, evt):
        # Space + left => pan mode
        if wx.GetKeyState(wx.WXK_SPACE) and evt.Dragging() and evt.LeftIsDown():
            if not self._space_pan_active:
                self._space_pan_active = True
                self._pan_start = evt.GetX()
                self._pan_origin = self.pan_offset
            dx = evt.GetX() - self._pan_start
            self._apply_pan(dx)
            return

        # Middle/right panning
        if (evt.MiddleIsDown() or evt.RightIsDown()) and getattr(self, "_pan_start", None) is not None:
            dx = evt.GetX() - self._pan_start
            self._apply_pan(dx)
            return

        # Span editing
        if getattr(self, "_span_edit", None) and evt.Dragging() and evt.LeftIsDown():
            w, h = self.GetClientSize()
            x = evt.GetX()
            ed = self._span_edit
            i = ed["index"]
            which = ed["which"]
            s = self.spans[i]
            if which == "left":
                s["start"] = self._x_to_index(x, w)
            elif which == "right":
                s["end"] = self._x_to_index(x, w)
            elif which == "body":
                dx_px = x - ed["last_x"]
                ed["last_x"] = x
                plot_w = max(1, w - 2 * self.padding)
                left, right = self.get_visible_index_range()
                span = max(1, right - left)
                di = int(dx_px * span / (plot_w * self.zoom_factor))
                s["start"] += di
                s["end"] += di
            self._normalize_span(s)
            self.Refresh()
            return

        # Annotator dragging
        if (self.role == "annotator" or self.draggable) and getattr(self, "dragging", False) and evt.LeftIsDown():
            self._handle_drag(evt)
            return

        # Hover
        self._handle_hover(evt)

    def _apply_pan(self, dx):
        new_offset = self._pan_origin + dx
        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        min_offset = gw * (1 - self.zoom_factor)
        self.pan_offset = min(max(new_offset, min_offset), 0)
        # sync
        for sp in self.sync_panels:
            sp.pan_offset = self.pan_offset
            sp.Refresh()
        self.Refresh()

    def on_mouse_leave(self, evt):
        self.hover_idx = None
        self.Refresh()
        try:
            evt.Skip()
        except Exception:
            pass

    def on_double_click(self, evt):
        # rename if allowed
        if self.role != "original" and self.allow_edit_title:
            dlg = wx.TextEntryDialog(self, "Enter new title:", "Rename", self.title)
            if dlg.ShowModal() == wx.ID_OK:
                self.title = dlg.GetValue()
                if hasattr(self, "title_label"):
                    self.title_label.SetLabel(self.title)
                self.Refresh()
            dlg.Destroy()
        # formula edit if present
        if self.formula:
            dlg2 = wx.TextEntryDialog(self, "Enter new formula:", "Edit Formula", self.formula)
            if dlg2.ShowModal() == wx.ID_OK:
                self.formula = dlg2.GetValue()
                self.Refresh()
            dlg2.Destroy()

    def on_left_down(self, evt):
        try:
            self.SetFocus()
        except Exception:
            pass

        w, h = self.GetClientSize()
        x = evt.GetX()
        y = evt.GetY()

        # Span edge/body hit to edit
        idx, which = self._hit_span_edge(x, y, w, h)
        if which is not None:
            self._span_edit = {"index": idx, "which": which, "last_x": x}
            try:
                self.CaptureMouse()
            except Exception:
                pass
            return

        # Start creating span with Shift
        if evt.ShiftDown():
            self._span_idx0 = self._x_to_index(x, w)
            self._span_edit = None
            return

        self._span_edit = None

        # Annotator: pick nearest point to drag
        if self.role == "annotator":
            x0, y0 = evt.GetPosition()
            best_dist = float("inf")
            best_idx = None
            for i, v in enumerate(self.seq):
                try:
                    cx, cy = self.to_px(i, v, w, h)
                except Exception:
                    continue
                dist = (cx - x0) ** 2 + (cy - y0) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None and best_dist < (self._hit_radius ** 2):
                self.selected_idx = best_idx
                self.dragging = True
                self._push_undo()
                try:
                    self.CaptureMouse()
                except Exception:
                    pass
                return

        # Generic draggable panels (if any)
        if self.draggable and self.role != "annotator":
            x0, y0 = evt.GetPosition()
            best_dist = float("inf")
            best_idx = None
            for i, v in enumerate(self.seq):
                cx, cy = self.to_px(i, v, w, h)
                dist = (cx - x0) ** 2 + (cy - y0) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None and best_dist < (self.radius * 3) ** 2:
                self.selected_idx = best_idx
                self.dragging = True
                try:
                    self.CaptureMouse()
                except Exception:
                    pass

    def on_left_up(self, evt):
        w, h = self.GetClientSize()

        # Finish span edit
        if getattr(self, "_span_edit", None):
            self._span_edit = None
            if self.HasCapture():
                try:
                    self.ReleaseMouse()
                except Exception:
                    pass
            self.Refresh()
            return

        # Finish span creation (Shift drag)
        if getattr(self, "_span_idx0", None) is not None and evt.ShiftDown():
            i0 = int(self._span_idx0)
            i1 = self._x_to_index(evt.GetX(), w)
            if i1 != i0:
                s = {"start": i0, "end": i1, "label": None, "value": None}
                self._normalize_span(s)
                self.spans.append(s)
                # Hook: allow panel to react (annotation panel fills values, etc)
                cb = getattr(self, "on_span_created", None)
                if callable(cb):
                    try:
                        cb(s)
                    except Exception:
                        pass
            self._span_idx0 = None
            self.Refresh()
            return

        # Finish dragging
        if getattr(self, "dragging", False):
            self.dragging = False
            if self.HasCapture():
                try:
                    self.ReleaseMouse()
                except Exception:
                    pass
            if self.role == "annotator":
                self._trigger_on_update()
            self.Refresh()

    def on_middle_down(self, evt):
        self._pan_start = evt.GetX()
        self._pan_origin = self.pan_offset

    def on_middle_up(self, evt):
        self._pan_start = None
        self._space_pan_active = False

    def on_mouse_wheel(self, evt):
        rotation = evt.GetWheelRotation() / max(1, evt.GetWheelDelta())
        factor = 1 + rotation * 0.1
        old_zoom = self.zoom_factor
        self.zoom_factor = max(1.0, min(old_zoom * factor, 20.0))

        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        mouse_x = evt.GetX() - self.padding - self.pan_offset

        # keep zoom centered under mouse
        try:
            self.pan_offset -= mouse_x * (self.zoom_factor / old_zoom - 1)
        except ZeroDivisionError:
            pass

        # clamp
        min_offset = gw * (1 - self.zoom_factor)
        self.pan_offset = min(max(self.pan_offset, min_offset), 0)

        # sync
        for sp in self.sync_panels:
            sp.zoom_factor = self.zoom_factor
            sp.pan_offset = self.pan_offset
            sp.Refresh()

        self.Refresh()

    # ---------------------------
    # Drag handling (annotator)
    # ---------------------------
    def _handle_drag(self, evt):
        if self.selected_idx is None:
            return

        x, y = evt.GetPosition()
        w, h = self.GetClientSize()
        mn, mx = self.get_y_range()
        rng = mx - mn or 1.0
        val = ((h - self.padding - y) / (h - 2 * self.padding)) * rng + mn
        val = max(min(val, mx), mn)

        seq_np = np.asarray(self.seq, dtype=float)
        if 0 <= self.selected_idx < seq_np.size:
            seq_np[self.selected_idx] = float(val)
            self.seq = seq_np

        # Propagate to synced panels via formula
        for sp in self.sync_panels:
            if getattr(sp, "original_seq", None) is not None and sp.formula:
                try:
                    sp.seq[self.selected_idx] = safe_eval(
                        sp.formula,
                        {'orig': sp.original_seq[self.selected_idx], 'res': val}
                    )
                except Exception:
                    pass
                sp.Refresh()

        # Special propagation to annotation/result
        annot = np.asarray(self.seq, dtype=float)
        for sp in self.sync_panels:
            role = getattr(sp, "role", None)
            if role == "annotation" and getattr(sp, "original_seq", None) is not None:
                orig = np.asarray(sp.original_seq, dtype=float)
                minlen = min(len(orig), len(annot))
                new_ann = orig[:minlen] - annot[:minlen]
                padded = np.zeros(getattr(sp, "n", minlen), dtype=float)
                padded[:minlen] = new_ann
                sp.seq = padded.tolist()
                sp.Refresh()
            if role == "result" and getattr(sp, "original_seq", None) is not None:
                orig = np.asarray(sp.original_seq, dtype=float)
                minlen = min(len(orig), len(annot))
                new_res = 2 * orig[:minlen] - annot[:minlen]
                padded = np.zeros(getattr(sp, "n", minlen), dtype=float)
                padded[:minlen] = new_res
                sp.seq = padded.tolist()
                sp.annotator_ref = self
                sp.Refresh()

        self.Refresh()
        self._trigger_on_update()

    def _trigger_on_update(self):
        cb = getattr(self, "on_update", None) or getattr(self, "on_update_callback", None)
        if callable(cb):
            try:
                cb(np.asarray(self.seq, dtype=float).copy())
            except Exception:
                pass

    # ---------------------------
    # Hover
    # ---------------------------
    def _handle_hover(self, evt):
        x, _ = evt.GetPosition()
        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        if self.n <= 1:
            self.hover_idx = None
            return
        step = (gw / max(1, (self.n - 1))) * self.zoom_factor
        idx = int((x - self.padding - self.pan_offset) / step + 0.5)
        self.hover_idx = max(0, min(idx, self.n - 1))
        self.Refresh()
        try:
            evt.Skip()
        except Exception:
            pass

    # ---------------------------
    # Painting (role visuals)
    # ---------------------------
    def on_paint(self, evt):
        w, h = self.GetClientSize()
        if w < 2 or h < 2 or self.n < 2:
            return

        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return

        # Background
        gc.SetBrush(wx.Brush(wx.WHITE))
        gc.DrawRectangle(0, 0, w, h)

        # Axes
        gc.SetPen(wx.Pen(wx.LIGHT_GREY))
        gc.StrokeLine(self.padding, self.padding, self.padding, h - self.padding)
        gc.StrokeLine(self.padding, h - self.padding, w - self.padding, h - self.padding)

        # Ranges
        mn, mx = self.get_y_range()
        rng = (mx - mn) or 1.0

        # Title
        try:
            f2 = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE.NORMAL, wx.FONTWEIGHT.BOLD)
            gf = gc.CreateFont(f2, wx.BLACK)
            gc.SetFont(gf)
            tw, _ = gc.GetTextExtent(self.title)
            gc.DrawText(self.title, (w - tw) / 2, 5)
        except Exception:
            pass

        # Clip to plot
        gc.Clip(self.padding, self.padding, w - 2 * self.padding, h - 2 * self.padding)

        # Convert to pts
        seq_arr = np.asarray(self.seq, dtype=float)
        pts = [self.to_px(i, v, w, h, mn, mx) for i, v in enumerate(seq_arr)]

        # Original overlay
        if getattr(self, "original_seq", None) is not None and len(self.original_seq) == self.n:
            orig_arr = np.asarray(self.original_seq, dtype=float)
            orig_pts = [self.to_px(i, v, w, h, mn, mx) for i, v in enumerate(orig_arr)]
            p = gc.CreatePath()
            p.MoveToPoint(*orig_pts[0])
            for pt in orig_pts[1:]:
                p.AddLineToPoint(*pt)
            gc.SetPen(wx.Pen(wx.BLUE, 1, wx.PENSTYLE_DOT))
            gc.StrokePath(p)

        # Role-specific drawing
        if self.role == "annotation":
            if len(pts) >= 2:
                p = gc.CreatePath()
                p.MoveToPoint(*pts[0])
                for pt in pts[1:]:
                    p.AddLineToPoint(*pt)
                gc.SetPen(wx.Pen(self.color, 1))
                gc.StrokePath(p)

        elif self.role == "annotator":
            if len(pts) >= 2:
                p = gc.CreatePath()
                p.MoveToPoint(*pts[0])
                for pt in pts[1:]:
                    p.AddLineToPoint(*pt)
                gc.SetPen(wx.Pen(self.color, 1))
                gc.StrokePath(p)
            for (x, y) in pts:
                gc.SetBrush(wx.Brush(self.color))
                gc.DrawEllipse(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius)

        elif self.role == "result":
            annot_pts = None
            if getattr(self, "annotator_ref", None) is not None:
                try:
                    ann_arr = np.asarray(self.annotator_ref.seq, dtype=float)
                    minlen = min(len(seq_arr), len(ann_arr))
                    orig = np.asarray(self.original_seq, dtype=float) if getattr(self, "original_seq", None) is not None else seq_arr
                    orig = orig[:minlen]
                    ann = ann_arr[:minlen]
                    orig_pts2 = [self.to_px(i, v, w, h, mn, mx) for i, v in enumerate(orig)]
                    ann_pts2 = [self.to_px(i, v, w, h, mn, mx) for i, v in enumerate(ann)]
                    if len(orig_pts2) >= 2 and len(ann_pts2) >= 2:
                        path_fill = gc.CreatePath()
                        path_fill.MoveToPoint(*orig_pts2[0])
                        for pnt in orig_pts2[1:]:
                            path_fill.AddLineToPoint(*pnt)
                        for pnt in reversed(ann_pts2):
                            path_fill.AddLineToPoint(*pnt)
                        path_fill.CloseSubpath()
                        gc.SetBrush(wx.Brush(wx.Colour(220, 60, 60, 80)))
                        try:
                            gc.FillPath(path_fill)
                        except Exception:
                            gc.SetPen(wx.Pen(wx.Colour(220, 60, 60), 1))
                            gc.StrokePath(path_fill)
                        path_orig = gc.CreatePath()
                        path_orig.MoveToPoint(*orig_pts2[0])
                        for pnt in orig_pts2[1:]:
                            path_orig.AddLineToPoint(*pnt)
                        gc.SetPen(wx.Pen(wx.Colour(10, 60, 160), 1))
                        gc.StrokePath(path_orig)
                        annot_pts = ann_pts2
                except Exception:
                    annot_pts = None
            if annot_pts is None:
                if len(pts) >= 2:
                    p = gc.CreatePath()
                    p.MoveToPoint(*pts[0])
                    for pt in pts[1:]:
                        p.AddLineToPoint(*pt)
                    gc.SetPen(wx.Pen(self.color, 1))
                    gc.StrokePath(p)

        else:
            if len(pts) >= 2:
                p = gc.CreatePath()
                p.MoveToPoint(*pts[0])
                for pt in pts[1:]:
                    p.AddLineToPoint(*pt)
                gc.SetPen(wx.Pen(self.color, 1))
                gc.StrokePath(p)

        # Draw playhead
        if getattr(self, "play_idx", None) is not None and 0 <= self.play_idx < self.n:
            x_px, _ = self.to_px(self.play_idx, seq_arr[self.play_idx], w, h, mn, mx)
            gc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1, wx.PENSTYLE_DOT))
            gc.StrokeLine(x_px, self.padding, x_px, h - self.padding)

        # Draw spans (yellow translucent)
        if getattr(self, 'spans', None):
            gc.SetBrush(wx.Brush(wx.Colour(200, 200, 0, 64)))
            gc.SetPen(wx.Pen(wx.Colour(200, 200, 0, 64)))
            left_idx, right_idx = self.get_visible_index_range()
            for s in self.spans:
                a = int(s.get("start", 0))
                b = int(s.get("end", 0))
                A = max(left_idx, min(right_idx, a))
                B = max(left_idx, min(right_idx, b))
                if B <= A:
                    continue
                x0, _ = self.to_px(A, mn, w, h, mn, mx)
                x1, _ = self.to_px(B, mn, w, h, mn, mx)
                gc.DrawRectangle(x0, self.padding, x1 - x0, h - 2 * self.padding)

        # Selected & hover handles
        if getattr(self, "selected_idx", None) is not None and 0 <= self.selected_idx < len(pts):
            x, y = pts[self.selected_idx]
            gc.SetBrush(wx.Brush(self.color))
            gc.DrawEllipse(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius)

        if getattr(self, "hover_idx", None) is not None and 0 <= self.hover_idx < len(pts):
            x, y = pts[self.hover_idx]
            gc.SetBrush(wx.Brush(self.color))
            gc.DrawEllipse(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius)

        # Sync hover across panels
        for sp in self.sync_panels:
            if getattr(sp, "hover_idx", None) != getattr(self, "hover_idx", None):
                sp.hover_idx = self.hover_idx
                sp.Refresh()

    # ---------------------------
    # Public legend API
    # ---------------------------
    def set_annotation_legend(self, legend_map: dict):
        self.annotation_legend = dict(legend_map or {})

    def get_annotation_legend(self) -> dict:
        return self.annotation_legend.copy()

    def set_annotator_ref(self, annotator_panel):
        self.annotator_ref = annotator_panel

    # ---------------------------
    # Keyboard
    # ---------------------------
    def on_key_down(self, evt):
        key = evt.GetKeyCode()
        mods = evt.GetModifiers()

        # Delete a span containing the playhead
        if key in (wx.WXK_DELETE, wx.WXK_NUMPAD_DELETE):
            if getattr(self, "spans", None) and getattr(self, "play_idx", None) is not None:
                for i, s in enumerate(list(self.spans)):
                    a = int(s.get("start", 0))
                    b = int(s.get("end", a))
                    if a > b:
                        a, b = b, a
                    if a <= self.play_idx <= b:
                        try:
                            self.spans.pop(i)
                            self.Refresh()
                            return
                        except Exception:
                            pass

        # Undo/redo
        if (mods & wx.MOD_CONTROL) and key == ord('Z'):
            self._undo(); return
        if (mods & wx.MOD_CONTROL) and key == ord('Y'):
            self._redo(); return

        # Playhead + nudge
        if key in (wx.WXK_LEFT, wx.WXK_RIGHT, wx.WXK_UP, wx.WXK_DOWN):
            if getattr(self, "play_idx", None) is None:
                self.play_idx = 0
            if key == wx.WXK_LEFT and self.play_idx > 0:
                self.play_idx -= 1
            elif key == wx.WXK_RIGHT and self.play_idx < (self.n - 1):
                self.play_idx += 1
            elif key in (wx.WXK_UP, wx.WXK_DOWN) and getattr(self, "draggable", False):
                delta = 1.0 if not (mods & wx.MOD_SHIFT) else 0.1
                if key == wx.WXK_DOWN:
                    delta = -delta
                try:
                    self.seq[self.play_idx] = float(self.seq[self.play_idx]) + delta
                    cb = getattr(self, "on_update", None)
                    if callable(cb):
                        cb(np.asarray(self.seq, dtype=float).copy())
                except Exception:
                    pass
            self.Refresh()
            return

        try:
            evt.Skip()
        except Exception:
            pass
