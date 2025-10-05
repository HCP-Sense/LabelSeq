"""InteractiveSequencePanel.py: SequencePanel subclass adding roles (original, annotator, annotation, result),
draggable annotator points, filled result area, highlights, and improved pan/zoom navigation."""

__author___ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Moye Nyuysoni Glein Perry"

from SequencePanel import SequencePanel
from FormulaParser import safe_eval

import wx
import numpy as np
from typing import Optional, Callable, List, Any


class InteractiveSequencePanel(SequencePanel):
    """
    Extended SequencePanel with roles:
      - role="original"   : static original (no edit title/delete)
      - role="annotator"  : draggable points; call on_update(seq) when changed
      - role="annotation" : shows (original - annotator)
      - role="result"     : shows original and filled area between original and annotator; highlights annotations
    """

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
        """
        Note: keep signature compatible with the rest of your code; role/on_update are new.
        """
        # call base ctor with expected args (do NOT forward role/on_update)
        super().__init__(parent, sequence, title, formula, draggable, visible_count, color)

        # Role + callbacks
        self.role = str(role).lower() if role is not None else "result"
        self.on_update_callback = on_update

        # allow flags (UI)
        self.allow_edit_title = allow_edit_title
        self.allow_delete = allow_delete

        # annotator reference (set externally by main if needed)
        # annotator_ref should be a reference to the annotator panel object (so we can read annotator_ref.seq)
        self.annotator_ref = None

        # annotation legend info (exposed for main to include in the legend)
        # mapping label -> wx.Colour used for highlights in result
        self.annotation_legend = {"Positive": wx.RED, "Negative": wx.Colour(0, 120, 255)}

        # small hit radius for point selection (in pixels)
        self._hit_radius = max(6, int(self.radius * 1.5))

        # Setup UI controls (title, buttons) with role-aware visibility
        self._setup_ui()

        # Rebind / augment events for improved navigation & interaction
        # keep original bindings from SequencePanel; add or override where appropriate
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_mouse_leave)
        self.Bind(wx.EVT_LEFT_DCLICK, self.on_double_click)
        self.Bind(wx.EVT_MOTION, self.on_mouse_motion)

        # dragging/selection only for annotator or if panel.draggable True
        if self.role == "annotator" or self.draggable:
            self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
            self.Bind(wx.EVT_LEFT_UP, self.on_left_up)

        # Pan with middle or right button (and also space+left drag)
        self.Bind(wx.EVT_MIDDLE_DOWN, self.on_middle_down)
        self.Bind(wx.EVT_MIDDLE_UP, self.on_middle_up)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_middle_down)   # treat right as pan start
        self.Bind(wx.EVT_RIGHT_UP, self.on_middle_up)

        # Mouse wheel for zoom
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)

        # internal pan tracking for space+left panning
        self._space_pan_active = False

    # ---------------------------
    # UI Setup (role-aware)
    # ---------------------------
    def _setup_ui(self):
        """Creates the title and buttons UI at the top, but hides buttons depending on role/flags."""
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.title_label = wx.StaticText(self, label=self.title, style=wx.ALIGN_CENTER)
        title_font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.title_label.SetFont(title_font)
        btn_sizer.Add(self.title_label, 1, wx.ALIGN_CENTER_VERTICAL)

        # EDIT TITLE: hide for 'original' if not explicitly allowed
        if self.allow_edit_title and self.role != "original":
            btn_title = wx.Button(self, label="Edit Title", size=(90, -1))
            btn_title.Bind(wx.EVT_BUTTON, self.edit_title)
            btn_sizer.Add(btn_title, 0, wx.LEFT, 5)

        # Formula button (keep original behaviour)
        if self.formula:
            btn_formula = wx.Button(self, label="Edit Formula", size=(100, -1))
            btn_formula.Bind(wx.EVT_BUTTON, self.edit_formula)
            btn_sizer.Add(btn_formula, 0, wx.LEFT, 5)

        # DELETE: disallow for 'original'; also disallow for 'result' if requested
        if self.allow_delete and self.role not in ("original", "result"):
            btn_delete = wx.Button(self, label="Delete", size=(70, -1))
            btn_delete.Bind(wx.EVT_BUTTON, self.delete_self)
            btn_sizer.Add(btn_delete, 0, wx.LEFT, 5)

        main_sizer = self.GetSizer()
        if main_sizer is None:
            main_sizer = wx.BoxSizer(wx.VERTICAL)
            self.SetSizer(main_sizer)

        # Replace top row (safe: clear existing then add)
        # Note: do NOT destroy other children (plot area) — SequencePanel draws below
        main_sizer.Clear()
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)
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
    # Mouse / interaction handlers
    # ---------------------------
    def on_mouse_motion(self, evt):
        # Space + left => pan mode (alternative)
        keys = wx.GetKeyState(wx.WXK_SPACE)
        if keys and evt.Dragging() and evt.LeftIsDown():
            if not self._space_pan_active:
                self._space_pan_active = True
                self._pan_start = evt.GetX()
                self._pan_origin = self.pan_offset

            dx = evt.GetX() - self._pan_start
            self._apply_pan(dx)
            return

        # Middle or right panning
        if (evt.MiddleIsDown() or evt.RightIsDown()) and getattr(self, "_pan_start", None) is not None:
            dx = evt.GetX() - self._pan_start
            self._apply_pan(dx)
            return

        # Left-dragging: annotation/point editing
        if (self.role == "annotator" or self.draggable) and getattr(self, "dragging", False) and evt.LeftIsDown():
            self._handle_drag(evt)
            return

        # otherwise treat as hover move
        self._handle_hover(evt)

    def _apply_pan(self, dx):
        """Common pan logic; dx is pixel delta from pan start."""
        new_offset = self._pan_origin + dx
        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        min_offset = gw * (1 - self.zoom_factor)
        self.pan_offset = min(max(new_offset, min_offset), 0)

        # Sync pan offset with linked panels
        for sp in self.sync_panels:
            sp.pan_offset = self.pan_offset
            sp.Refresh()

        self.Refresh()

    def on_mouse_leave(self, evt):
        self.hover_idx = None
        self.Refresh()
        evt.Skip()

    def on_double_click(self, evt):
        # double click: rename title (unless original locked)
        if self.role != "original":
            dlg = wx.TextEntryDialog(self, "Enter new title:", "Rename", self.title)
            if dlg.ShowModal() == wx.ID_OK:
                self.title = dlg.GetValue()
                self.title_label.SetLabel(self.title)
                self.Refresh()
            dlg.Destroy()
        # allow formula edit on double click if present
        if self.formula:
            dlg2 = wx.TextEntryDialog(self, "Enter new formula:", "Edit Formula", self.formula)
            if dlg2.ShowModal() == wx.ID_OK:
                self.formula = dlg2.GetValue()
                self.Refresh()
            dlg2.Destroy()

    def on_left_down(self, evt):
        # if annotator role => click on nearest point to start dragging
        if self.role == "annotator":
            x0, y0 = evt.GetPosition()
            w, h = self.GetClientSize()
            best_dist = float('inf')
            best_idx = None

            # compute points using to_px helper
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
                self.CaptureMouse()
                return

        # fallback for draggable panels (existing behavior)
        if self.draggable and not self.role == "annotator":
            # reuse previous selection algorithm (closest point)
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

    def on_left_up(self, evt):
        if getattr(self, "dragging", False):
            self.dragging = False
            try:
                self.ReleaseMouse()
            except Exception:
                pass
            # after finishing a drag, inform listeners
            if self.role == "annotator":
                self._trigger_on_update()
            self.Refresh()

    def on_middle_down(self, evt):
        # start pan (works for middle or right)
        self._pan_start = evt.GetX()
        self._pan_origin = self.pan_offset

    def on_middle_up(self, evt):
        self._pan_start = None
        self._space_pan_active = False

    def on_mouse_wheel(self, evt):
        # allow zooming for all panels; keep behavior centered on mouse
        rotation = evt.GetWheelRotation() / evt.GetWheelDelta()
        factor = 1 + rotation * 0.1
        old_zoom = self.zoom_factor
        self.zoom_factor = max(1.0, min(old_zoom * factor, 20.0))  # allow higher zoom

        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        mouse_x = evt.GetX() - self.padding - self.pan_offset

        # Adjust pan to keep zoom centered
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

    # ---------------------------
    # Drag handling (point editing)
    # ---------------------------
    def _handle_drag(self, evt):
        """Update the value under selected_idx based on mouse Y and propagate to other panels."""
        if self.selected_idx is None:
            return

        x, y = evt.GetPosition()
        w, h = self.GetClientSize()
        mn, mx = self.get_y_range()
        rng = mx - mn or 1.0

        # convert vertical position to panel value
        val = ((h - self.padding - y) / (h - 2 * self.padding)) * rng + mn
        val = max(min(val, mx), mn)

        # update own sequence (SequencePanel stores seq as list)
        try:
            self.seq[self.selected_idx] = float(val)
        except Exception:
            # if seq is numpy array
            self.seq = list(self.seq)
            self.seq[self.selected_idx] = float(val)

        # Propagate changes:
        # 1) Existing formula-based propagation (if panels use formulas)
        for sp in self.sync_panels:
            if getattr(sp, "original_seq", None) is not None and sp.formula:
                try:
                    sp.seq[self.selected_idx] = safe_eval(
                        sp.formula,
                        {'orig': sp.original_seq[self.selected_idx], 'res': val}
                    )
                except Exception:
                    # ignore formula errors
                    pass
                sp.Refresh()

        # 2) Special propagation for annotator -> annotation/result panels
        if self.role == "annotator":
            # read annotator as numpy for arithmetic
            annot = np.asarray(self.seq, dtype=float)
            for sp in self.sync_panels:
                if getattr(sp, "role", None) == "annotation":
                    # compute annotation = original - annotator
                    if getattr(sp, "original_seq", None) is not None:
                        orig = np.asarray(sp.original_seq, dtype=float)
                        minlen = min(len(orig), len(annot))
                        new_ann = orig[:minlen] - annot[:minlen]
                        # convert back to list with same length as sp.seq
                        # if sp.seq length differs, create a new list sized to sp.n
                        target_len = sp.n if getattr(sp, "n", None) else len(new_ann)
                        # fill or pad with zeros if needed
                        padded = np.zeros(target_len, dtype=float)
                        padded[:len(new_ann)] = new_ann
                        sp.seq = padded.tolist()
                        sp.Refresh()
                if getattr(sp, "role", None) == "result":
                    # compute result = original + annotation (as you requested)
                    # annotation = orig - annot => result = orig + (orig - annot) = 2*orig - annot
                    if getattr(sp, "original_seq", None) is not None:
                        orig = np.asarray(sp.original_seq, dtype=float)
                        minlen = min(len(orig), len(annot))
                        new_res = 2 * orig[:minlen] - annot[:minlen]
                        target_len = sp.n if getattr(sp, "n", None) else len(new_res)
                        padded = np.zeros(target_len, dtype=float)
                        padded[:len(new_res)] = new_res
                        sp.seq = padded.tolist()
                        # store a reference to annotator for drawing (so result can fill area)
                        sp.annotator_ref = self
                        sp.Refresh()

        # local refresh
        self.Refresh()

        # optionally call on_update callback while dragging (live update)
        if self.role == "annotator":
            self._trigger_on_update()

    def _trigger_on_update(self):
        """Call the on_update callback with a numpy copy of current annotator sequence."""
        if callable(self.on_update_callback):
            try:
                arr = np.asarray(self.seq, dtype=float).copy()
                self.on_update_callback(arr)
            except Exception:
                pass

    # ---------------------------
    # Hover handling
    # ---------------------------
    def _handle_hover(self, evt):
        """Compute hover index and refresh (shared behavior)."""
        x, _ = evt.GetPosition()
        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        # protect division by zero and n<2
        if self.n <= 1:
            self.hover_idx = None
            return
        step = gw / (self.n - 1) * self.zoom_factor
        idx = int((x - self.padding - self.pan_offset) / step + 0.5)
        self.hover_idx = max(0, min(idx, self.n - 1))
        self.Refresh()
        evt.Skip()

    # ---------------------------
    # Painting (override to handle role-specific visuals)
    # ---------------------------
    def on_paint(self, evt):
        """
        Custom paint: start from SequencePanel drawing approach but add:
          - annotator: show points prominently
          - annotation: draw original overlay and annotation line
          - result: draw filled polygon between original and annotator, plus highlights
        """
        w, h = self.GetClientSize()
        if w < 2 or h < 2 or self.n < 2:
            return

        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return

        # Fonts
        try:
            f_wx = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
            f_gc = gc.CreateFont(f_wx, wx.BLACK)
            gc.SetFont(f_gc)
        except Exception:
            pass

        # Background
        gc.SetBrush(wx.Brush(wx.WHITE))
        gc.DrawRectangle(0, 0, w, h)

        # Axes (basic)
        gc.SetPen(wx.Pen(wx.LIGHT_GREY))
        gc.StrokeLine(self.padding, self.padding, self.padding, h - self.padding)  # Y axis
        gc.StrokeLine(self.padding, h - self.padding, w - self.padding, h - self.padding)  # X axis

        # Y ticks & labels based on combined range if original exists
        mn, mx = self.get_y_range()
        rng = mx - mn
        step_val = rng / 5 if rng != 0 else 1.0
        for k in range(6):
            val = mn + k * step_val
            y = h - self.padding - (val - mn) * ((h - 2 * self.padding) / (rng or 1.0))
            gc.StrokeLine(self.padding - 5, y, self.padding, y)
            txt = f"{val:.1f}"
            tw, th = gc.GetTextExtent(txt)
            gc.DrawText(txt, self.padding - 10 - tw, y - th / 2)

        # X ticks
        step_idx = max(1, self.n // 10)
        for i in range(0, self.n, step_idx):
            try:
                x, _ = self.to_px(i, mn, w, h, mn, mx)
            except Exception:
                continue
            if self.padding <= x <= w - self.padding:
                gc.StrokeLine(x, h - self.padding, x, h - self.padding + 5)
                lbl = str(i)
                tw, th = gc.GetTextExtent(lbl)
                gc.DrawText(lbl, x - tw / 2, h - self.padding + 8)

        # Title
        f2 = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        fg = gc.CreateFont(f2, wx.BLACK)
        gc.SetFont(fg)
        tw, _ = gc.GetTextExtent(self.title)
        gc.DrawText(self.title, (w - tw) / 2, 5)

        # Clip to plot area
        gc.Clip(self.padding, self.padding, w - 2 * self.padding, h - 2 * self.padding)

        # Convert seqs to points for drawing
        # Use numpy arrays for arithmetic
        seq_arr = np.asarray(self.seq, dtype=float)
        pts = []
        # handle possible length mismatches gracefully
        for i, v in enumerate(seq_arr):
            x, y = self.to_px(i, v, w, h, mn, mx)
            pts.append((x, y))

        # Draw original overlay (if present and differs)
        if getattr(self, "original_seq", None) is not None and len(self.original_seq) == self.n:
            orig_arr = np.asarray(self.original_seq, dtype=float)
            orig_pts = [self.to_px(i, v, w, h, mn, mx) for i, v in enumerate(orig_arr)]
            path0 = gc.CreatePath()
            path0.MoveToPoint(*orig_pts[0])
            for pt in orig_pts[1:]:
                path0.AddLineToPoint(*pt)
            gc.SetPen(wx.Pen(wx.BLUE, 1, wx.PENSTYLE_DOT))
            gc.StrokePath(path0)
        else:
            orig_pts = None

        # Role-specific drawing
        if self.role == "annotation":
            # annotation should represent original - annotator; seq already set by main/annotator propagation
            # Draw annotation as main colored line
            if len(pts) >= 2:
                path_ann = gc.CreatePath()
                path_ann.MoveToPoint(*pts[0])
                for pt in pts[1:]:
                    path_ann.AddLineToPoint(*pt)
                gc.SetPen(wx.Pen(self.color, 1))
                gc.StrokePath(path_ann)

        elif self.role == "annotator":
            # Draw annotator line and points (draggable)
            if len(pts) >= 2:
                path_ann = gc.CreatePath()
                path_ann.MoveToPoint(*pts[0])
                for pt in pts[1:]:
                    path_ann.AddLineToPoint(*pt)
                gc.SetPen(wx.Pen(self.color, 1))
                gc.StrokePath(path_ann)

            # Draw points as filled circles
            for i, (x, y) in enumerate(pts):
                gc.SetBrush(wx.Brush(self.color))
                gc.DrawEllipse(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius)

        elif self.role == "result":
            # For result, draw filled polygon between original and annotator (if annotator_ref exists)
            # If annotator_ref present, compute annotator_pts that line up with original points
            annot_pts = None
            if getattr(self, "annotator_ref", None) is not None:
                try:
                    ann_arr = np.asarray(self.annotator_ref.seq, dtype=float)
                    # align lengths
                    minlen = min(len(seq_arr), len(ann_arr))
                    orig = np.asarray(self.original_seq, dtype=float) if getattr(self, "original_seq", None) is not None else seq_arr
                    orig = orig[:minlen]
                    ann = ann_arr[:minlen]
                    # build pixel points
                    orig_pts2 = [self.to_px(i, v, w, h, mn, mx) for i, v in enumerate(orig)]
                    ann_pts2 = [self.to_px(i, v, w, h, mn, mx) for i, v in enumerate(ann)]
                    if len(orig_pts2) >= 2 and len(ann_pts2) >= 2:
                        path_fill = gc.CreatePath()
                        path_fill.MoveToPoint(*orig_pts2[0])
                        for p in orig_pts2[1:]:
                            path_fill.AddLineToPoint(*p)
                        # go back along annotator pts reversed
                        for p in reversed(ann_pts2):
                            path_fill.AddLineToPoint(*p)
                        # close
                        path_fill.CloseSubpath()
                        gc.SetBrush(wx.Brush(wx.Colour(220, 60, 60, 80)))  # semi-transparent red-like
                        try:
                            gc.FillPath(path_fill)
                        except Exception:
                            # fallback: stroke polygon instead
                            gc.SetPen(wx.Pen(wx.Colour(220, 60, 60), 1))
                            gc.StrokePath(path_fill)
                        # draw original line on top
                        path_orig = gc.CreatePath()
                        path_orig.MoveToPoint(*orig_pts2[0])
                        for p in orig_pts2[1:]:
                            path_orig.AddLineToPoint(*p)
                        gc.SetPen(wx.Pen(wx.Colour(10, 60, 160), 1))
                        gc.StrokePath(path_orig)
                        annot_pts = ann_pts2
                except Exception:
                    annot_pts = None
            # fallback: if no annotator_ref or error -> draw seq as normal
            if annot_pts is None:
                if len(pts) >= 2:
                    path1 = gc.CreatePath()
                    path1.MoveToPoint(*pts[0])
                    for pt in pts[1:]:
                        path1.AddLineToPoint(*pt)
                    gc.SetPen(wx.Pen(self.color, 1))
                    gc.StrokePath(path1)

            # Highlights: compute annotation values (orig - annot) if annotator present, else zeros
            if getattr(self, "annotator_ref", None) is not None and getattr(self, "original_seq", None) is not None:
                try:
                    orig_arr = np.asarray(self.original_seq, dtype=float)
                    ann_arr = np.asarray(self.annotator_ref.seq, dtype=float)
                    minlen = min(len(orig_arr), len(ann_arr))
                    diff = orig_arr[:minlen] - ann_arr[:minlen]
                    if minlen > 0:
                        rng = np.nanmax(diff) - np.nanmin(diff)
                        thresh = 0.08 * (rng if rng != 0 else (np.nanmax(np.abs(orig_arr[:minlen])) or 1.0))
                        # draw small markers where abs(diff) > thresh
                        for i, d in enumerate(diff):
                            if abs(d) >= thresh:
                                # choose color for sign
                                c = self.annotation_legend["Positive"] if d > 0 else self.annotation_legend["Negative"]
                                x, y = self.to_px(i, orig_arr[i], w, h, mn, mx)
                                gc.SetBrush(wx.Brush(c))
                                gc.DrawEllipse(x - (self.radius/1.8), y - (self.radius/1.8), self.radius*1.6, self.radius*1.6)
                except Exception:
                    pass

        else:
            # default drawing (generic)
            if len(pts) >= 2:
                path1 = gc.CreatePath()
                path1.MoveToPoint(*pts[0])
                for pt in pts[1:]:
                    path1.AddLineToPoint(*pt)
                gc.SetPen(wx.Pen(self.color, 1))
                gc.StrokePath(path1)

        # Draw selected/hover handles (common)
        if self.selected_idx is not None and 0 <= self.selected_idx < len(pts):
            x, y = pts[self.selected_idx]
            gc.SetBrush(wx.Brush(self.color))
            gc.DrawEllipse(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius)

        if self.hover_idx is not None and 0 <= self.hover_idx < len(pts):
            x, y = pts[self.hover_idx]
            # draw hover differently (thin white border)
            gc.SetBrush(wx.Brush(self.color))
            gc.DrawEllipse(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius)

        # finally sync hover to other panels (so they highlight)
        for sp in self.sync_panels:
            if getattr(sp, "hover_idx", None) != self.hover_idx:
                sp.hover_idx = self.hover_idx
                sp.Refresh()

    # ---------------------------
    # Helpers for legend / external access
    # ---------------------------
    def get_annotation_legend(self) -> dict:
        """
        Return mapping label->wx.Colour representing annotation highlights.
        Main can call this on the result panel and add legend entries under an "Annotations" heading.
        """
        return self.annotation_legend.copy()

    def set_annotator_ref(self, annotator_panel):
        """Set a reference to the annotator panel so 'annotation' and 'result' roles can read live annotator.seq"""
        self.annotator_ref = annotator_panel

    # ensure we still expose the same public API as before
    # (edit_title, edit_formula, delete_self etc. are provided above)
