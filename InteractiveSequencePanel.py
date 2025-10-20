"""
InteractiveSequencePanel.py
Extended SequencePanel with roles:
  - original  : read-only
  - annotator : draggable points, feeds active learning
  - annotation: span-based labeling + freeze, draggable only when unfrozen
  - result    : passive display, shows filled area & active learning suggestions

Includes:
  - X and Y panning
  - Zoom
  - Hover sync
  - Playhead sync
  - Fast rendering with decimation (MAX_VISIBLE_POINTS = 1000 for speed)
  - Active Learning Engine built in (point/span/hybrid)
  - Hybrid storage of AL suggestions:
      * span mode  -> store spans with textual labels
      * point mode -> store numeric values into annotation array
"""

__author__ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Development (Optimized with Dragging=2 and Fast Rendering)"

# ======================================
# IMPORTS
# ======================================
import wx
import numpy as np
from typing import Optional, Callable, Any, Dict, List, Tuple
from SequencePanel import SequencePanel
from FormulaParser import safe_eval

# ======================================
# PERFORMANCE CONSTANT
# ======================================
MAX_VISIBLE_POINTS = 1000  # Fast decimation limit for rendering across ALL panels

# ======================================
# DEBUG FLAG (you can turn this off later easily)
# ======================================
DEBUG_MODE = True


def debug_print(msg: str):
    """Print debug messages only when DEBUG_MODE is True."""
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}")


# =============================================================================
# ACTIVE LEARNING ENGINE (as provided earlier, minimal modifications for safety)
# =============================================================================
_SKLEARN_OK = True
try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    _SKLEARN_OK = False


class ActiveLearningEngine:
    """
    This class handles active learning suggestions based on labeled points or spans.
    It supports:
        - point mode
        - span mode
        - hybrid (automatic decision)
    """
    def __init__(self, mode: str = "hybrid", threshold: float = 0.85, window: int = 3):
        self.mode = mode.lower()
        self.threshold = float(threshold)
        self.window = max(1, int(window))
        self._signal = None
        self._X = None

        self._point_labels: Dict[int, str] = {}
        self._span_labels: List[Tuple[int, int, str]] = []

        self._label_to_int: Dict[str, int] = {}
        self._int_to_label: Dict[int, str] = {}

        self._clf_knn = None
        self._clf_rf = None

        self.last_suggestions: List[int] = []

    def fit(self, signal: np.ndarray):
        """Fit the active learning engine to a base signal."""
        signal = np.asarray(signal, dtype=float).flatten()
        if signal.size < 2:
            signal = np.pad(signal, (0, 2 - signal.size), mode='edge')
        self._signal = signal
        self._X = self._make_features(signal)
        self._clf_knn = None
        self._clf_rf = None
        debug_print("ActiveLearningEngine: Signal fitted.")

    def _make_features(self, x: np.ndarray) -> np.ndarray:
        """Generate feature matrix."""
        n = len(x)
        X = np.zeros((n, 8), dtype=float)
        X[:, 0] = x
        X[:, 1] = np.diff(x, prepend=x[0])
        X[:, 2] = np.diff(X[:, 1], prepend=X[:, 1][0])

        w = self.window
        for i in range(n):
            a = max(0, i - w)
            b = min(n, i + w + 1)
            seg = x[a:b]
            X[i, 3] = np.mean(seg)
            X[i, 4] = np.std(seg) if len(seg) > 1 else 0.0
            X[i, 5] = np.min(seg)
            X[i, 6] = np.max(seg)
            if len(seg) >= 2:
                t = np.arange(len(seg))
                t_mean = np.mean(t)
                denom = np.sum((t - t_mean) ** 2) or 1.0
                slope = np.sum((t - t_mean) * (seg - np.mean(seg))) / denom
            else:
                slope = 0.0
            X[i, 7] = slope
        return X

    def set_mode(self, mode: str):
        self.mode = mode.lower()
        debug_print(f"ActiveLearningEngine: Mode set to {self.mode}")

    def set_threshold(self, th: float):
        self.threshold = float(th)
        debug_print(f"ActiveLearningEngine: Threshold set to {self.threshold}")

    def add_point(self, idx: int, label: str):
        """Add a labeled point."""
        if self._signal is None:
            return
        idx = int(np.clip(idx, 0, len(self._signal) - 1))
        self._point_labels[idx] = label
        debug_print(f"ActiveLearningEngine: Point labeled at index {idx} with '{label}'")

    def add_span(self, start: int, end: int, label: str):
        """Add a labeled span."""
        if self._signal is None:
            return
        a = int(np.clip(min(start, end), 0, len(self._signal) - 1))
        b = int(np.clip(max(start, end), 0, len(self._signal) - 1))
        self._span_labels.append((a, b, label))
        debug_print(f"ActiveLearningEngine: Span labeled [{a}, {b}] with '{label}'")

    # (The rest of ActiveLearningEngine will continue in Part 2 or Part 3)



# =============================================================================
# INTERACTIVE SEQUENCE PANEL CLASS - PARTIAL (Beginning)
# =============================================================================
class InteractiveSequencePanel(SequencePanel):
    def __init__(
        self,
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
        allow_delete: bool = True,
        al_mode: str = "hybrid",
        al_threshold: float = 0.85
    ):
        """
        role options:
         - original  : read-only base signal
         - annotator : draggable points to adjust signal
         - annotation: derived, supports spans & freeze
         - result    : passive visualization, shaded area only
        """
        super().__init__(parent, sequence, title, formula, draggable, visible_count, color)

        # === Debug Mode Flag ===
        self.debug = DEBUG_MODE

        # === Performance: Fast rendering settings ===
        self.MAX_VISIBLE_POINTS = MAX_VISIBLE_POINTS  # applies to all rendering logic

        # === Core Attributes ===
        self.role = role.lower().strip()
        self.on_update = on_update
        self.allow_edit_title = allow_edit_title
        self.allow_delete = allow_delete
        self.original_seq = getattr(self, "original_seq", None)
        self.annotator_ref = None

        # Visualization state
        self.play_idx = 0
        self.hover_idx = None
        self.spans: List[Dict[str, Any]] = []
        self.selected_idx = None
        self.dragging = False  # Will be set to preview-only behavior
        self.frozen = False

        # Pan & Zoom State
        self.zoom_factor = 1.0
        self.pan_offset = 0
        self.y_offset = 0.0
        self._space_pan_active = False
        self._ypan_active = False

        # Undo/Redo
        self._undo_stack: List[np.ndarray] = []
        self._redo_stack: List[np.ndarray] = []

        # Sync Panels
        self.sync_panels: List["InteractiveSequencePanel"] = []

        # === Unified refresh throttling system ===
        self._refresh_pending = False

        def _request_refresh_local():
            if self._refresh_pending:
                return
            self._refresh_pending = True
            def _do():
                self._refresh_pending = False
                try:
                    if self.debug:
                        debug_print(f"Refreshing panel '{self.title}'")
                    self.Refresh(False)
                except Exception:
                    pass
            wx.CallAfter(_do)

        self._request_refresh = _request_refresh_local  # Assign as instance method

        # === Dragging = preview only, apply on release ===
        self.drag_preview_mode = True  # this enables dragging=2 behavior

        if self.debug:
            debug_print(f"InteractiveSequencePanel initialized with role={self.role}, title={self.title}")
        # ------------------------------------------------------
        # Active Learning Integration (instance)
        # ------------------------------------------------------
        self.al_engine = ActiveLearningEngine(mode=al_mode, threshold=al_threshold, window=3)
        base_signal = (
            np.asarray(self.original_seq, dtype=float)
            if self.original_seq is not None
            else np.asarray(self.seq, dtype=float)
        )
        self.al_engine.fit(base_signal)

        self.al_mode = al_mode
        self.al_threshold = al_threshold

        # Overlay for suggestions (preview only)
        self._suggest_indices: List[int] = []
        self._suggest_labels: List[str] = []

        # Callback hooks to MainFrame if available
        self._active_labels_callback = None
        self._active_spans_callback = None
        top = wx.GetTopLevelParent(self)
        if hasattr(top, "on_active_labels") and callable(getattr(top, "on_active_labels")):
            self._active_labels_callback = getattr(top, "on_active_labels")
        if hasattr(top, "on_active_labels_spans") and callable(getattr(top, "on_active_labels_spans")):
            self._active_spans_callback = getattr(top, "on_active_labels_spans")

        # ------------------------------------------------------
        # Build UI & Bind Events Role-Safely
        # ------------------------------------------------------
        self._setup_ui()
        self._bind_role_safe_events()

    # ==============================================================
    # Expose legend map if Result wants to display annotation colors
    # ==============================================================
    def get_annotation_legend(self) -> Dict[str, Any]:
        return getattr(self, "annotation_legend", {})

    def set_annotation_legend(self, legend_map: Dict[str, Any]):
        self.annotation_legend = dict(legend_map or {})
        self._request_refresh()

    # ==============================================================
    # Role-Safe Event Binding
    # ==============================================================
    def _bind_role_safe_events(self):
        """Bind only handlers that safely exist per role."""

        bindings_common = {
            wx.EVT_MOUSEWHEEL: "on_mouse_wheel",
            wx.EVT_MOTION: "on_mouse_motion",
            wx.EVT_LEAVE_WINDOW: "on_mouse_leave",
            wx.EVT_SET_FOCUS: "on_focus",
            wx.EVT_KILL_FOCUS: "on_kill_focus",
            wx.EVT_KEY_DOWN: "on_key_down",
            wx.EVT_MIDDLE_DOWN: "on_middle_down",
            wx.EVT_MIDDLE_UP: "on_middle_up",
            wx.EVT_RIGHT_DOWN: "on_right_down",
            wx.EVT_RIGHT_UP: "on_right_up",
        }
        for evt_type, handler_name in bindings_common.items():
            if hasattr(self, handler_name):
                self.Bind(evt_type, getattr(self, handler_name))

        if self.role in ("annotator", "annotation"):
            edit_bindings = {
                wx.EVT_LEFT_DOWN: "on_left_down",
                wx.EVT_LEFT_UP: "on_left_up",
                wx.EVT_LEFT_DCLICK: "on_double_click",
            }
            for evt_type, handler_name in edit_bindings.items():
                if hasattr(self, handler_name):
                    self.Bind(evt_type, getattr(self, handler_name))

        try:
            self.SetFocusable(True)
        except Exception:
            pass

    # ==============================================================
    # UI (title row) + AL controls
    # ==============================================================
    def _setup_ui(self):
        top = self.GetSizer()
        if top is None:
            top = wx.BoxSizer(wx.VERTICAL)
            self.SetSizer(top)
        else:
            top.Clear()

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.title_label = wx.StaticText(self, label=self.title, style=wx.ALIGN_CENTER)
        self.title_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        row.Add(self.title_label, 1, wx.ALIGN_CENTER_VERTICAL)

        # Edit Title (not for original)
        if self.allow_edit_title and self.role != "original":
            btn_title = wx.Button(self, label="Edit Title", size=(90, -1))
            btn_title.Bind(wx.EVT_BUTTON, self.edit_title)
            row.Add(btn_title, 0, wx.LEFT, 6)

        # Annotation: Freeze + Edit Formula
        if self.role == "annotation":
            self.frozen = False
            self.btn_freeze = wx.ToggleButton(self, label="🔓 Unfrozen", size=(110, -1))
            self.btn_freeze.Bind(wx.EVT_TOGGLEBUTTON, self.toggle_freeze)
            row.Add(self.btn_freeze, 0, wx.LEFT, 6)

            btn_formula = wx.Button(self, label="Edit Formula", size=(110, -1))
            btn_formula.Bind(wx.EVT_BUTTON, self.edit_formula)
            row.Add(btn_formula, 0, wx.LEFT, 6)

        # Result: Export button (no editing)
        if self.role == "result":
            btn_export = wx.Button(self, label="Export Result", size=(120, -1))
            def _do_export(evt):
                top_frame = wx.GetTopLevelParent(self)
                if hasattr(top_frame, "on_export_result_for_panel"):
                    top_frame.on_export_result_for_panel(self)
                elif hasattr(top_frame, "on_export_result"):
                    top_frame.on_export_result(evt)
            btn_export.Bind(wx.EVT_BUTTON, _do_export)
            row.Add(btn_export, 0, wx.LEFT, 6)

        # Delete (not for original/result/annotator)
        if self.allow_delete and self.role not in ("original", "result", "annotator"):
            btn_delete = wx.Button(self, label="Delete", size=(80, -1))
            btn_delete.Bind(wx.EVT_BUTTON, self.delete_self)
            row.Add(btn_delete, 0, wx.LEFT, 6)

        # Active Learning controls visible for annotator/annotation
        if self.role in ("annotator", "annotation"):
            row.AddSpacer(10)
            row.Add(wx.StaticText(self, label="AL:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)

            self.al_mode_choice = wx.Choice(self, choices=["point", "span", "hybrid"])
            try:
                self.al_mode_choice.SetStringSelection(self.al_mode)
            except Exception:
                self.al_mode_choice.SetSelection(2)  # hybrid
            self.al_mode_choice.Bind(wx.EVT_CHOICE, self._on_al_mode_changed)
            row.Add(self.al_mode_choice, 0, wx.RIGHT, 6)

            row.Add(wx.StaticText(self, label="τ:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
            self.al_thresh_ctrl = wx.SpinCtrlDouble(self, min=0.50, max=0.99,
                                                    initial=float(self.al_threshold), inc=0.01, size=(80, -1))
            self.al_thresh_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_al_threshold_changed)
            row.Add(self.al_thresh_ctrl, 0, wx.RIGHT, 6)

            btn_suggest = wx.Button(self, label="Suggest", size=(90, -1))
            btn_suggest.Bind(wx.EVT_BUTTON, self._on_al_suggest_clicked)
            row.Add(btn_suggest, 0, wx.LEFT, 4)

        top.Add(row, 0, wx.EXPAND | wx.ALL, 5)
        self.Layout()

    # ==============================================================
    # Title / Formula / Freeze / Delete
    # ==============================================================
    def edit_title(self, evt):
        dlg = wx.TextEntryDialog(self, "Enter new title:", "Edit Title", self.title)
        if dlg.ShowModal() == wx.ID_OK:
            self.title = dlg.GetValue()
            self.title_label.SetLabel(self.title)
            self.Layout()
            self._request_refresh()
            top = wx.GetTopLevelParent(self)
            if hasattr(top, "_refresh_legend"):
                top._refresh_legend()
        dlg.Destroy()

    def edit_formula(self, evt):
        dlg = wx.TextEntryDialog(self, "Enter new formula:", "Edit Formula", self.formula or "")
        if dlg.ShowModal() == wx.ID_OK:
            self.formula = dlg.GetValue()
            self._request_refresh()
        dlg.Destroy()

    def toggle_freeze(self, evt=None):
        self.frozen = bool(self.btn_freeze.GetValue())
        self.btn_freeze.SetLabel("🔒 Frozen" if self.frozen else "🔓 Unfrozen")
        self._request_refresh()

    def delete_self(self, evt):
        top = wx.GetTopLevelParent(self)
        if hasattr(top, "remove_panel"):
            top.remove_panel(self)

    # ==============================================================
    # Undo/Redo helpers
    # ==============================================================
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
            try: cb(self.seq.copy())
            except Exception: pass
        self._request_refresh()

    def _redo(self):
        if self.role != "annotator" or not self._redo_stack:
            return
        nxt = self._redo_stack.pop()
        self._undo_stack.append(np.asarray(self.seq, dtype=float).copy())
        self.seq = nxt.copy()
        cb = getattr(self, "on_update", None)
        if callable(cb):
            try: cb(self.seq.copy())
            except Exception: pass
        self._request_refresh()

    # ==============================================================
    # Span helpers & hit testing
    # ==============================================================
    def _x_to_index(self, x_px, w):
        plot_w = max(1, w - 2 * self.padding)
        rel = (x_px - self.padding - self.pan_offset) / (plot_w * self.zoom_factor)
        left, right = self.get_visible_index_range()
        span = max(1, right - left)
        rel = max(0.0, min(1.0, rel))
        return int(left + rel * span)

    def _normalize_span(self, s: Dict[str, Any]):
        a, b = int(s.get("start", 0)), int(s.get("end", 0))
        if b < a: a, b = b, a
        a = max(0, min(self.n - 1, a))
        b = max(0, min(self.n - 1, b))
        s["start"], s["end"] = a, b
        return s

    def _hit_span_edge(self, x_px, _y_px, w, _h, tol=4):
        left_idx, right_idx = self.get_visible_index_range()
        for i, s in enumerate(self.spans or []):
            a, b = int(s.get("start", 0)), int(s.get("end", 0))
            if b < a: a, b = b, a
            if b <= left_idx or a >= right_idx:
                continue
            ax, _ = self.to_px(a, self.seq[a if 0 <= a < self.n else 0], w, self.GetClientSize().height)
            bx, _ = self.to_px(b, self.seq[b if 0 <= b < self.n else self.n - 1], w, self.GetClientSize().height)
            if abs(x_px - ax) <= tol: return i, "left"
            if abs(x_px - bx) <= tol: return i, "right"
            if min(ax, bx) + tol < x_px < max(ax, bx) - tol: return i, "body"
        return None, None

    # ==============================================================
    # Mouse / interaction handlers (optimized)
    # ==============================================================
    def on_mouse_motion(self, evt):
        # A+Left => Y-pan (all roles)
        if evt.Dragging() and evt.LeftIsDown() and wx.GetKeyState(ord('A')):
            if not self._ypan_active:
                self._ypan_active = True
                self._ypan_start_y = evt.GetY()
                self._ypan_origin = self.y_offset
            dy = evt.GetY() - (self._ypan_start_y or evt.GetY())
            self._apply_pan_y(dy, origin=self._ypan_origin)
            return

        # Space+Left => X-pan (all roles)
        if wx.GetKeyState(wx.WXK_SPACE) and evt.Dragging() and evt.LeftIsDown():
            if not self._space_pan_active:
                self._space_pan_active = True
                self._pan_start = evt.GetX()
                self._pan_origin = self.pan_offset
            dx = evt.GetX() - self._pan_start
            self._apply_pan_x(dx)
            return

        # Middle drag => X-pan
        if evt.MiddleIsDown() and getattr(self, "_pan_start", None) is not None:
            dx = evt.GetX() - self._pan_start
            self._apply_pan_x(dx)
            return

        # Right drag => Y-pan
        if evt.RightIsDown() and getattr(self, "_ypan_start_y", None) is not None:
            dy = evt.GetY() - self._ypan_start_y
            self._apply_pan_y(dy, origin=self._ypan_origin)
            return

        # Span editing (annotation only)
        if self.role == "annotation" and getattr(self, "_span_edit", None) and evt.Dragging() and evt.LeftIsDown():
            if getattr(self, "frozen", False):
                return
            w, _h = self.GetClientSize()
            x = evt.GetX()
            s = self.spans[self._span_edit["index"]]
            which = self._span_edit["which"]
            if which == "left":
                s["start"] = self._x_to_index(x, w)
            elif which == "right":
                s["end"] = self._x_to_index(x, w)
            elif which == "body":
                dx = x - self._span_edit["last_x"]
                self._span_edit["last_x"] = x
                left, right = self.get_visible_index_range()
                span_total = max(1, right - left)
                gw = max(1, w - 2 * self.padding)
                di = int(dx * span_total / (gw * self.zoom_factor))
                s["start"] += di; s["end"] += di
            self._normalize_span(s)
            self._request_refresh()
            return

        # Point dragging (annotator only) -> preview-only
        if self.role == "annotator" and getattr(self, "dragging", False) and evt.LeftIsDown():
            self._handle_drag(evt)  # preview only, no propagation
            return

        # Otherwise: hover
        self._handle_hover(evt)

    def on_left_down(self, evt):
        try: self.SetFocus()
        except Exception: pass

        w, _h = self.GetClientSize()
        x = evt.GetX()

        # Annotation: hit test span edges/body
        if self.role == "annotation":
            idx, which = self._hit_span_edge(x, evt.GetY(), w, _h)
            if which is not None and not getattr(self, "frozen", False):
                self._span_edit = {"index": idx, "which": which, "last_x": x}
                try: self.CaptureMouse()
                except Exception: pass
                return
            # Start span creation with Shift
            if evt.ShiftDown() and not getattr(self, "frozen", False):
                self._span_idx0 = self._x_to_index(x, w)
                self._span_edit = None
                return

        # Avoid starting a drag if A is held (Y-pan)
        if wx.GetKeyState(ord('A')):
            return

        # Annotator: select nearest point for dragging
        if self.role == "annotator":
            x0, y0 = evt.GetPosition()
            best_d = float("inf"); best_i = None
            for i, v in enumerate(self.seq):
                try:
                    cx, cy = self.to_px(i, v, *self.GetClientSize())
                except Exception:
                    continue
                d = (cx - x0)**2 + (cy - y0)**2
                if d < best_d:
                    best_d = d; best_i = i
            hit_radius = max(6, int(self.radius * 1.5)) if hasattr(self, "radius") else 8
            if best_i is not None and best_d < (hit_radius ** 2):
                self.selected_idx = best_i
                self.dragging = True
                self._push_undo()
                try: self.CaptureMouse()
                except Exception: pass

    def on_left_up(self, evt):
        # End Y-pan if active
        if self._ypan_active:
            self._ypan_active = False
            self._ypan_start_y = None

        # Finish span edit (annotation)
        if self.role == "annotation" and getattr(self, "_span_edit", None):
            self._span_edit = None
            if self.HasCapture():
                try: self.ReleaseMouse()
                except Exception: pass
            self._request_refresh()
            return

        # Finish span creation (annotation)
        if self.role == "annotation" and getattr(self, "_span_idx0", None) is not None and evt.ShiftDown() and not getattr(self, "frozen", False):
            w, _h = self.GetClientSize()
            i0 = int(self._span_idx0)
            i1 = self._x_to_index(evt.GetX(), w)
            if i1 != i0:
                s = {"start": i0, "end": i1, "label": self.title, "value": None}
                self._normalize_span(s)
                self.spans.append(s)
                # feed AL (span/hybrid modes)
                if self.al_engine.mode in ("span", "hybrid"):
                    self.al_engine.add_span(s["start"], s["end"], self.title)
            self._span_idx0 = None
            self._request_refresh()
            return

        # Finish dragging (annotator) -> apply once + propagate
        if self.role == "annotator" and getattr(self, "dragging", False):
            self.dragging = False
            if self.HasCapture():
                try: self.ReleaseMouse()
                except Exception: pass

            if self.selected_idx is not None and self.al_engine.mode in ("point", "hybrid"):
                self.al_engine.add_point(int(self.selected_idx), self.title)

            # Single propagation to peers
            try:
                annot = np.asarray(self.seq, dtype=float)
                for sp in self.sync_panels:
                    role = getattr(sp, "role", None)
                    if role == "annotation" and getattr(sp, "original_seq", None) is not None:
                        orig = np.asarray(sp.original_seq, dtype=float)
                        m = min(len(orig), len(annot))
                        new_ann = orig[:m] - annot[:m]
                        if getattr(sp, "n", None) is not None:
                            padded = np.zeros(getattr(sp, "n"), dtype=float)
                            padded[:m] = new_ann
                            sp.seq = padded.tolist()
                        else:
                            sp.seq = new_ann.tolist()
                        try:
                            sp._request_refresh()
                        except Exception:
                            sp.Refresh(False)
                    if role == "result" and getattr(sp, "original_seq", None) is not None:
                        orig = np.asarray(sp.original_seq, dtype=float)
                        m = min(len(orig), len(annot))
                        new_res = 2 * orig[:m] - annot[:m]
                        if getattr(sp, "n", None) is not None:
                            padded = np.zeros(getattr(sp, "n"), dtype=float)
                            padded[:m] = new_res
                            sp.seq = padded.tolist()
                        else:
                            sp.seq = new_res.tolist()
                        sp.annotator_ref = self
                        try:
                            sp._request_refresh()
                        except Exception:
                            sp.Refresh(False)
            except Exception:
                pass

            # Fire single update callback (let the model recompute once)
            if callable(self.on_update):
                try:
                    self.on_update(np.asarray(self.seq, dtype=float).copy())
                except Exception:
                    pass

            self._request_refresh()

    def on_double_click(self, evt):
        # convenience: rename; edit formula if annotation
        if self.role != "original" and self.allow_edit_title:
            self.edit_title(evt)
        if self.role == "annotation":
            self.edit_formula(evt)

    def on_mouse_leave(self, evt):
        self.hover_idx = None
        self._request_refresh()
        if evt is not None:
            evt.Skip()

    def on_middle_down(self, evt):
        self._pan_start = evt.GetX()
        self._pan_origin = self.pan_offset

    def on_middle_up(self, evt):
        self._pan_start = None
        self._space_pan_active = False

    def on_right_down(self, evt):
        self._ypan_active = True
        self._ypan_start_y = evt.GetY()
        self._ypan_origin = self.y_offset

    def on_right_up(self, evt):
        self._ypan_active = False
        self._ypan_start_y = None

    def on_mouse_wheel(self, evt):
        # Shift+wheel => Y-pan, else X-zoom around mouse
        rotation = evt.GetWheelRotation() / max(1, evt.GetWheelDelta())

        if evt.ShiftDown():
            mn, mx = self.get_y_range()
            rng = max(mx - mn, 1e-9)
            self.y_offset -= rotation * (0.08 * rng)  # 8% per notch
            for sp in getattr(self, "sync_panels", []):
                if hasattr(sp, "y_offset"):
                    sp.y_offset = self.y_offset
                    try: sp._request_refresh()
                    except Exception: sp.Refresh(False)
            self._request_refresh()
            return

        factor = 1 + rotation * 0.1
        old_zoom = self.zoom_factor
        self.zoom_factor = max(1.0, min(old_zoom * factor, 40.0))

        w, _ = self.GetClientSize()
        gw = max(1, w - 2 * self.padding)
        mouse_x = evt.GetX() - self.padding - self.pan_offset
        try:
            self.pan_offset -= mouse_x * (self.zoom_factor / old_zoom - 1)
        except ZeroDivisionError:
            pass

        min_offset = gw * (1 - self.zoom_factor)
        self.pan_offset = min(max(self.pan_offset, min_offset), 0)

        for sp in self.sync_panels:
            sp.zoom_factor = self.zoom_factor
            sp.pan_offset = self.pan_offset
            try: sp._request_refresh()
            except Exception: sp.Refresh(False)

        self._request_refresh()

    # ==============================================================
    # Pan helpers (sync X/Y to peers) - optimized
    # ==============================================================
    def _apply_pan_x(self, dx):
        new_offset = getattr(self, "_pan_origin", 0) + dx
        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        min_offset = gw * (1 - self.zoom_factor)
        self.pan_offset = min(max(new_offset, min_offset), 0)
        for sp in self.sync_panels:
            sp.pan_offset = self.pan_offset
            try: sp._request_refresh()
            except Exception: sp.Refresh(False)
        self._request_refresh()

    def _apply_pan_y(self, dy, origin=None):
        mn, mx = self.get_y_range()
        rng = max(mx - mn, 1e-9)
        h = max(1, self.GetClientSize().height - 2 * self.padding)
        units_per_px = rng / h
        base = self.y_offset if origin is None else origin
        self.y_offset = base + dy * units_per_px
        for sp in self.sync_panels:
            if hasattr(sp, "y_offset"):
                sp.y_offset = self.y_offset
                try: sp._request_refresh()
                except Exception: sp.Refresh(False)
        self._request_refresh()

    # ==============================================================
    # Drag handling (annotator) - preview only during move
    # ==============================================================
    def _handle_drag(self, evt):
        if self.selected_idx is None:
            return
        _x, y = evt.GetPosition()
        w, h = self.GetClientSize()
        mn, mx = self._get_y_display_range()
        rng = mx - mn or 1.0
        val = ((h - self.padding - y) / (h - 2 * self.padding)) * rng + mn
        val = max(min(val, mx), mn)

        seq_np = np.asarray(self.seq, dtype=float)
        if 0 <= self.selected_idx < seq_np.size:
            seq_np[self.selected_idx] = float(val)
            self.seq = seq_np

        # No propagation during drag (performance)
        self._request_refresh()

    def _trigger_on_update(self):
        cb = getattr(self, "on_update", None) or getattr(self, "on_update_callback", None)
        if callable(cb):
            try: cb(np.asarray(self.seq, dtype=float).copy())
            except Exception: pass

    # ==============================================================
    # Hover handler (sync hover to peers) - optimized
    # ==============================================================
    def _handle_hover(self, evt):
        x, _ = evt.GetPosition()
        w, _ = self.GetClientSize()
        gw = max(1, w - 2 * self.padding)
        if self.n <= 1:
            self.hover_idx = None
        else:
            step = (gw / max(1, (self.n - 1))) * self.zoom_factor
            idx = int((x - self.padding - self.pan_offset) / step + 0.5)
            self.hover_idx = max(0, min(idx, self.n - 1))

        for sp in self.sync_panels:
            sp.hover_idx = self.hover_idx
            try: sp._request_refresh()
            except Exception: pass
        self._request_refresh()
        try:
            evt.Skip()
        except Exception:
            pass

    # ==============================================================
    # Rendering (fast/decimated) + markers + sync (no repaint loop)
    # ==============================================================
    def _get_y_display_range(self):
        mn, mx = self.get_y_range()
        return mn + self.y_offset, mx + self.y_offset

    def on_paint(self, evt):
        w, h = self.GetClientSize()
        if w < 2 or h < 2 or self.n < 2:
            return

        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return

        # ✅ Ensure a default font is always set
        try:
            font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
            gc.SetFont(font, wx.BLACK)
        except Exception:
            # absolute fallback to avoid crashes
            pass

        # Background
        gc.SetBrush(wx.Brush(wx.WHITE))
        gc.DrawRectangle(0, 0, w, h)

        # Axes
        gc.SetPen(wx.Pen(wx.LIGHT_GREY))
        gc.StrokeLine(self.padding, self.padding, self.padding, h - self.padding)          # Y-axis
        gc.StrokeLine(self.padding, h - self.padding, w - self.padding, h - self.padding)  # X-axis

        # Y-range with pan
        mn, mx = self._get_y_display_range()
        rng = max(mx - mn, 1e-9)

        # Y ticks
        step_val = rng / 5.0
        for i in range(6):
            val = mn + i * step_val
            yy = h - self.padding - (val - mn) * ((h - 2 * self.padding) / rng)
            gc.StrokeLine(self.padding - 4, yy, self.padding, yy)
            lbl = f"{val:.2f}"
            tw, th = gc.GetTextExtent(lbl)
            gc.DrawText(lbl, self.padding - 10 - tw, yy - th / 2)

        # X ticks based on visible range
        left_i, right_i = self.get_visible_index_range()
        max_labels = 10
        step_idx = max(1, (right_i - left_i) // max_labels)
        for i in range(left_i, right_i, step_idx):
            xx, _ = self.to_px(i, mn, w, h, mn, mx)
            gc.StrokeLine(xx, h - self.padding, xx, h - self.padding + 4)
            lbl = str(i)
            tw, th = gc.GetTextExtent(lbl)
            gc.DrawText(lbl, xx - tw / 2, h - self.padding + 6)

        # Title
        try:
            title_font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
            gc.SetFont(gc.CreateFont(title_font, wx.BLACK))
            tw, _ = gc.GetTextExtent(self.title)
            gc.DrawText(self.title, (w - tw) / 2, 5)
        except Exception:
            pass

        # Clip to plot area
        gc.Clip(self.padding, self.padding, w - 2 * self.padding, h - 2 * self.padding)

        # Decimation (keep ~MAX_VISIBLE_POINTS pts)
        seq_arr = np.asarray(self.seq, dtype=float)
        vis_len = max(1, right_i - left_i)
        stride = max(1, vis_len // min(vis_len, self.MAX_VISIBLE_POINTS))
        idxs = np.arange(left_i, right_i, stride, dtype=int)

        # Points to draw for this panel
        pts = [self.to_px(i, seq_arr[i], w, h, mn, mx) for i in idxs]

        # Original overlay (dotted) if available and aligned length
        if getattr(self, "original_seq", None) is not None and len(self.original_seq) == self.n:
            orig_arr = np.asarray(self.original_seq, dtype=float)
            o_pts = [self.to_px(i, orig_arr[i], w, h, mn, mx) for i in idxs]
            if o_pts:
                path0 = gc.CreatePath()
                path0.MoveToPoint(*o_pts[0])
                for pt in o_pts[1:]:
                    path0.AddLineToPoint(*pt)
                gc.SetPen(wx.Pen(wx.BLUE, 1, wx.PENSTYLE_DOT))
                gc.StrokePath(path0)

        # Draw role visuals (line, markers, or shaded region for result)
        self._render_role_visual(gc, pts, w, h, mn, mx, idxs)

        # Active Learning overlay markers (from set_active_suggestions)
        if self._suggest_indices:
            gc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1))
            for ii in self._suggest_indices:
                if left_i <= ii < right_i:
                    xx, yy = self.to_px(ii, seq_arr[min(ii, self.n - 1)], w, h, mn, mx)
                    gc.StrokeLine(xx, self.padding, xx, h - self.padding)
                    gc.DrawEllipse(xx - 2, yy - 2, 4, 4)

        # Spans (annotation only)
        self._render_spans(gc, w, h, mn, mx)

        # Playhead
        if getattr(self, "play_idx", None) is not None and left_i <= self.play_idx < right_i:
            x_px, _ = self.to_px(self.play_idx, seq_arr[self.play_idx], w, h, mn, mx)
            gc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1, wx.PENSTYLE_DOT))
            gc.StrokeLine(x_px, self.padding, x_px, h - self.padding)

        # Hover marker
        if getattr(self, "hover_idx", None) is not None and left_i <= self.hover_idx < right_i:
            xh, yh = self.to_px(self.hover_idx, seq_arr[self.hover_idx], w, h, mn, mx)
            gc.SetBrush(wx.Brush(wx.GREEN))
            gc.DrawEllipse(xh - 4, yh - 4, 8, 8)

        # Selected point (annotator only)
        if self.role == "annotator" and getattr(self, "selected_idx", None) is not None:
            if left_i <= self.selected_idx < right_i:
                xs, ys = self.to_px(self.selected_idx, seq_arr[self.selected_idx], w, h, mn, mx)
                gc.SetBrush(wx.Brush(wx.BLUE))
                gc.DrawEllipse(xs - 4, ys - 4, 8, 8)

        # Sync hover & playhead to peers (no forced repaint to avoid loops)
        for sp in self.sync_panels:
            sp.hover_idx = self.hover_idx
            sp.play_idx = self.play_idx
        # Peers repaint on their own; avoid forced Refresh here.

    # --------------------------------------------------------------
    def _render_role_visual(self, gc, pts, w, h, mn, mx, indices):
        """Draw per-role main visual."""
        if not pts:
            return
        path = gc.CreatePath()
        path.MoveToPoint(*pts[0])
        for pt in pts[1:]:
            path.AddLineToPoint(*pt)

        if self.role == "annotator":
            gc.SetPen(wx.Pen(self.color, 1))
            gc.StrokePath(path)
            for (x, y) in pts:
                gc.SetBrush(wx.Brush(self.color))
                gc.DrawEllipse(x - self.radius, y - self.radius, 2 * self.radius, 2 * self.radius)

        elif self.role == "annotation":
            gc.SetPen(wx.Pen(self.color, 1))
            gc.StrokePath(path)

        elif self.role == "result":
            self._render_result_shaded(gc, w, h, mn, mx, indices)

        else:
            gc.SetPen(wx.Pen(self.color, 1))
            gc.StrokePath(path)

    # --------------------------------------------------------------
    def _render_result_shaded(self, gc, w, h, mn, mx, indices):
        """Shaded area between original and annotator lines in result panel."""
        if getattr(self, "annotator_ref", None) is None:
            seq_arr = np.asarray(self.seq, dtype=float)
            if indices.size == 0:
                return
            path = gc.CreatePath()
            x0, y0 = self.to_px(int(indices[0]), seq_arr[int(indices[0])], w, h, mn, mx)
            path.MoveToPoint(x0, y0)
            for i in indices[1:]:
                x, y = self.to_px(int(i), seq_arr[int(i)], w, h, mn, mx)
                path.AddLineToPoint(x, y)
            gc.SetPen(wx.Pen(self.color, 1))
            gc.StrokePath(path)
            return

        seq_arr = np.asarray(self.seq, dtype=float)
        ann_arr = np.asarray(self.annotator_ref.seq, dtype=float)
        m = min(len(seq_arr), len(ann_arr))
        if m <= 1:
            return

        orig_arr = self.original_seq[:m] if self.original_seq is not None else seq_arr[:m]
        ann_arr = ann_arr[:m]

        pts_orig = [self.to_px(int(i), float(orig_arr[int(i)]), w, h, mn, mx) for i in indices if int(i) < m]
        pts_ann  = [self.to_px(int(i), float(ann_arr[int(i)]),  w, h, mn, mx) for i in indices if int(i) < m]

        if not pts_orig or not pts_ann:
            return

        path_fill = gc.CreatePath()
        path_fill.MoveToPoint(*pts_orig[0])
        for p in pts_orig[1:]:
            path_fill.AddLineToPoint(*p)
        for p in reversed(pts_ann):
            path_fill.AddLineToPoint(*p)
        path_fill.CloseSubpath()

        gc.SetBrush(wx.Brush(wx.Colour(200, 50, 50, 80)))
        try:
            gc.FillPath(path_fill)
        except Exception:
            gc.SetPen(wx.Pen(wx.RED, 1))
            gc.StrokePath(path_fill)

        path_orig = gc.CreatePath()
        path_orig.MoveToPoint(*pts_orig[0])
        for p in pts_orig[1:]:
            path_orig.AddLineToPoint(*p)
        gc.SetPen(wx.Pen(wx.Colour(10, 60, 160), 1))
        gc.StrokePath(path_orig)

    # --------------------------------------------------------------
    def _render_spans(self, gc, w, h, mn, mx):
        """Background highlighted spans (annotation role only)."""
        if self.role != "annotation" or not getattr(self, "spans", None):
            return
        left_i, right_i = self.get_visible_index_range()
        gc.SetBrush(wx.Brush(wx.Colour(255, 255, 0, 60)))
        gc.SetPen(wx.Pen(wx.Colour(200, 200, 0, 80)))

        for s in self.spans:
            a, b = int(s.get("start", 0)), int(s.get("end", 0))
            if b < a: a, b = b, a
            A = max(left_i, min(right_i, a))
            B = max(left_i, min(right_i, b))
            if B <= A:
                continue
            x0, _ = self.to_px(A, mn, w, h, mn, mx)
            x1, _ = self.to_px(B, mn, w, h, mn, mx)
            gc.DrawRectangle(x0, self.padding, x1 - x0, h - 2 * self.padding)
    # ==============================================================
    # ACTIVE LEARNING: Suggestion Logic
    # ==============================================================
    def set_active_suggestions(self, idxs: List[int], labels: List[str]):
        """Overlay suggestions on this panel."""
        self._suggest_indices = list(map(int, idxs or []))
        self._suggest_labels = list(map(str, labels or []))
        self._request_refresh()

    def _clear_last_suggestions(self):
        self._suggest_indices = []
        self._suggest_labels = []
        top = wx.GetTopLevelParent(self)
        try:
            if hasattr(top, "result_panel") and top.result_panel:
                top.result_panel.set_active_suggestions([], [])
        except Exception:
            pass
        self._request_refresh()

    def _group_indices_to_spans(self, indices: List[int]) -> List[Tuple[int, int]]:
        """Group sorted indices into continuous spans [start, end]."""
        if not indices:
            return []
        spans = []
        start = prev = indices[0]
        for i in indices[1:]:
            if i == prev + 1:
                prev = i
                continue
            spans.append((start, prev))
            start = prev = i
        spans.append((start, prev))
        return spans

    def _run_active_learning_preview(self):
        """Run AL suggest and apply preview overlay only."""
        base_signal = (
            np.asarray(self.original_seq, dtype=float)
            if self.original_seq is not None
            else np.asarray(self.seq, dtype=float)
        )
        self.al_engine.fit(base_signal)
        idxs, labs = self.al_engine.suggest()
        if self.debug:
            debug_print(f"AL preview: {len(idxs)} suggestions")

        top = wx.GetTopLevelParent(self)
        try:
            if hasattr(top, "result_panel") and top.result_panel:
                top.result_panel.set_active_suggestions(idxs.tolist(), labs.tolist())
        except Exception:
            pass

        self.set_active_suggestions(idxs.tolist(), labs.tolist())

    def _on_al_suggest_clicked(self, evt):
        """Handle Suggest button click with confirm dialog."""
        self._run_active_learning_preview()
        idxs = self._suggest_indices or []
        labs = self._suggest_labels or []
        if not idxs:
            wx.MessageBox("No confident suggestions above threshold.", "Active Learning",
                          wx.OK | wx.ICON_INFORMATION)
            return

        mode = (self.al_mode or "hybrid").lower()
        groups = self._group_indices_to_spans(sorted(idxs))
        covered = sum((b - a + 1) for a, b in groups)
        use_spans = (mode == "span" or (mode == "hybrid" and covered >= 0.6 * len(idxs)))

        if self.debug:
            debug_print(f"AL suggest clicked. Mode={mode}, use_spans={use_spans}")

        top = wx.GetTopLevelParent(self)
        if use_spans:
            spans = groups
            span_labels = []
            for a, b in spans:
                seg = [(i, l) for i, l in zip(idxs, labs) if a <= i <= b]
                if seg:
                    cnt = {}
                    for _, l in seg:
                        cnt[l] = cnt.get(l, 0) + 1
                    lab = max(cnt.items(), key=lambda kv: kv[1])[0]
                else:
                    lab = self.title
                span_labels.append(lab)

            if callable(self._active_spans_callback):
                self._active_spans_callback(spans, span_labels, self)
                return

            msg = f"Apply {len(spans)} suggested spans?"
            if wx.MessageBox(msg, "Confirm", wx.YES_NO | wx.ICON_QUESTION) != wx.YES:
                return

            # Apply locally if no callback
            target = self if self.role == "annotation" else None
            if not target:
                for sp in self.sync_panels:
                    if getattr(sp, "role", "") == "annotation" and not getattr(sp, "frozen", False):
                        target = sp
                        break

            if target:
                for (a, b), lab in zip(spans, span_labels):
                    s = {"start": int(a), "end": int(b), "label": str(lab), "value": None}
                    target._normalize_span(s)
                    target.spans.append(s)
                    if hasattr(target, "al_engine"):
                        target.al_engine.add_span(s["start"], s["end"], s["label"])
                target._request_refresh()

        else:
            # Apply point numeric labels
            if callable(self._active_labels_callback):
                self._active_labels_callback(list(idxs), list(labs), self)
                return

            msg = f"Apply {len(idxs)} point labels (value=1.0)?"
            if wx.MessageBox(msg, "Confirm", wx.YES_NO | wx.ICON_QUESTION) != wx.YES:
                return

            target = None
            if self.role == "annotation":
                target = self
            else:
                for sp in self.sync_panels:
                    if getattr(sp, "role", "") == "annotation":
                        target = sp
                        break

            if target:
                ann = np.asarray(target.seq, dtype=float)
                for i in idxs:
                    if 0 <= i < ann.size:
                        ann[i] = 1.0
                target.seq = ann
                target._request_refresh()

                for sp in self.sync_panels:
                    if getattr(sp, "role", "") == "result" and getattr(sp, "original_seq", None) is not None:
                        orig = np.asarray(sp.original_seq, dtype=float)
                        m = min(len(orig), len(ann))
                        sp.seq = (orig[:m] + ann[:m]).tolist()
                        sp._request_refresh()

        self._request_refresh()

    def _apply_last_suggestions(self):
        """Apply suggestions without showing dialog (used by Ctrl+Enter)."""
        if not (self._suggest_indices and self._suggest_labels):
            return

        mode = (self.al_mode or "hybrid").lower()
        idxs = sorted(self._suggest_indices)
        labs = list(self._suggest_labels)
        groups = self._group_indices_to_spans(idxs)
        covered = sum((b - a + 1) for a, b in groups)
        use_spans = (mode == "span" or (mode == "hybrid" and covered >= 0.6 * len(idxs)))

        if use_spans and callable(self._active_spans_callback):
            spans = self._group_indices_to_spans(idxs)
            span_labels = []
            for a, b in spans:
                seg = [(i, l) for i, l in zip(idxs, labs) if a <= i <= b]
                if seg:
                    cnt = {}
                    for _, l in seg:
                        cnt[l] = cnt.get(l, 0) + 1
                    lab = max(cnt.items(), key=lambda kv: kv[1])[0]
                else:
                    lab = self.title
                span_labels.append(lab)
            self._active_spans_callback(spans, span_labels, self)
        elif callable(self._active_labels_callback):
            self._active_labels_callback(list(idxs), list(labs), self)

    # ==============================================================
    # Keyboard Shortcuts
    # ==============================================================
    def on_key_down(self, evt):
        key = evt.GetKeyCode()
        mods = evt.GetModifiers()

        # Delete span at playhead (annotation only)
        if key in (wx.WXK_DELETE, wx.WXK_NUMPAD_DELETE):
            if self.role == "annotation" and self.spans and self.play_idx is not None and not self.frozen:
                for i, s in enumerate(list(self.spans)):
                    a, b = int(s.get("start", 0)), int(s.get("end", 0))
                    if a > b: a, b = b, a
                    if a <= self.play_idx <= b:
                        self.spans.pop(i)
                        self._request_refresh()
                        return

        # Undo / Redo
        if self.role == "annotator" and (mods & wx.MOD_CONTROL) and key == ord('Z'):
            self._undo()
            return
        if self.role == "annotator" and (mods & wx.MOD_CONTROL) and key == ord('Y'):
            self._redo()
            return

        # Navigation
        if key == wx.WXK_LEFT:
            self.move_playhead(-1, broadcast=True); return
        if key == wx.WXK_RIGHT:
            self.move_playhead(+1, broadcast=True); return
        if key == wx.WXK_HOME:
            self.set_playhead(0, broadcast=True); self.center_on_playhead(); return
        if key == wx.WXK_END:
            self.set_playhead(self.n - 1, broadcast=True); self.center_on_playhead(); return
        if key == wx.WXK_PAGEUP:
            self.move_playhead(-(max(1, self.n // 20)), broadcast=True); self.center_on_playhead(); return
        if key == wx.WXK_PAGEDOWN:
            self.move_playhead(+(max(1, self.n // 20)), broadcast=True); self.center_on_playhead(); return

        # Nudge value at playhead (annotator only)
        if self.role == "annotator" and key in (wx.WXK_UP, wx.WXK_DOWN) and not self.frozen:
            delta = (0.1 if (mods & wx.MOD_SHIFT) else 1.0)
            if key == wx.WXK_DOWN:
                delta = -delta
            try:
                if self.play_idx is None:
                    self.play_idx = 0
                self.seq[self.play_idx] = float(self.seq[self.play_idx]) + delta
                cb = getattr(self, "on_update", None)
                if callable(cb):
                    cb(np.asarray(self.seq, dtype=float).copy())
            except Exception:
                pass
            self._request_refresh()
            return

        # Active Learning shortcuts
        if (mods & wx.MOD_CONTROL) and key == ord('L'):
            self._run_active_learning_preview()
            return

        if (mods & wx.MOD_CONTROL) and key in (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER):
            self._apply_last_suggestions()
            return

        if key == wx.WXK_ESCAPE:
            self._clear_last_suggestions()
            return

        evt.Skip()

    # ==============================================================
    # Playhead Helpers
    # ==============================================================
    def move_playhead(self, di: int, broadcast=False):
        self.play_idx = int(max(0, min(self.n - 1, (self.play_idx or 0) + di)))
        if broadcast:
            for sp in self.sync_panels:
                sp.play_idx = self.play_idx
                try: sp._request_refresh()
                except Exception: pass
        self._request_refresh()

    def set_playhead(self, i: int, broadcast=False):
        self.play_idx = int(max(0, min(self.n - 1, i)))
        if broadcast:
            for sp in self.sync_panels:
                sp.play_idx = self.play_idx
                try: sp._request_refresh()
                except Exception: pass
        self._request_refresh()

    def center_on_playhead(self):
        if self.play_idx is None:
            return
        w, _ = self.GetClientSize()
        gw = max(1, w - 2 * self.padding)
        left, right = self.get_visible_index_range()
        vis = max(1, right - left)
        step = (gw / max(1, (self.n - 1))) * self.zoom_factor
        x_play = self.padding + self.play_idx * step + self.pan_offset
        target = self.padding + gw / 2
        dx = target - x_play
        self._pan_origin = self.pan_offset
        self._apply_pan_x(dx)
# ==============================================================
# ACTIVE LEARNING UI EVENT HANDLERS
# ==============================================================
    def _on_al_mode_changed(self, evt):
        """Triggered when user changes active learning mode from UI"""
        try:
            new_mode = self.al_mode_choice.GetStringSelection()
        except Exception:
            new_mode = getattr(self, "al_mode", "hybrid")
        self.al_mode = new_mode
        if self.debug:
            debug_print(f"AL mode changed to: {self.al_mode}")
        if hasattr(self, "al_engine"):
            self.al_engine.set_mode(self.al_mode)

    def _on_al_threshold_changed(self, evt):
        """Triggered when user changes AL confidence threshold"""
        try:
            new_th = float(self.al_thresh_ctrl.GetValue())
        except Exception:
            new_th = getattr(self, "al_threshold", 0.85)
        self.al_threshold = new_th
        if self.debug:
            debug_print(f"AL threshold changed to: {self.al_threshold}")
        if hasattr(self, "al_engine"):
            self.al_engine.set_threshold(new_th)

    def _on_al_suggest_clicked(self, evt):
        """Trigger AL interactive suggestion with UI"""
        if self.debug:
            debug_print("Suggest button clicked")
        self._run_active_learning_preview()
        self._confirm_apply_suggestions()

# ============================
# End of InteractiveSequencePanel.py
# ============================
