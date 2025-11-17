# =============================================================================
# InteractiveSequencePanel.py
# =============================================================================
"""
InteractiveSequencePanel
Extended SequencePanel with roles:
  - original  : read-only
  - annotator : draggable points, feeds active learning
  - annotation: span-based labeling + freeze, draggable only when unfrozen
  - result    : passive display, shows filled area & active learning suggestions

Includes:
  - X and Y panning
  - Zoom
  - Hover sync
  - (Timeline bar removed — no playhead line drawn)
  - Fast rendering with decimation (MAX_VISIBLE_POINTS = 1000 for speed)
  - Active Learning Engine built in (point/span/hybrid)
  - Hybrid storage of AL suggestions:
      * span mode  -> store spans with textual labels
      * point mode -> store numeric values into annotation array

Region mode:
  - When 'region' mode is selected, NO draggable point markers are shown.
  - Left-drag creates a highlighted span selection (no Shift needed).

Y-axis sync:
  - All panels in the sync group share the same y-range,
    driven by the 'original' panel, so all show the same scale.
"""

__author__ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Development (Optimized with Dragging Modes + Fast Rendering)"

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
MAX_VISIBLE_POINTS = 1000  # Decimation target for visible samples

# ======================================
# DEBUG FLAG
# ======================================
DEBUG_MODE = True
def debug_print(msg: str):
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}")

# ======================================
# Optional sklearn import (soft dependency)
# ======================================
_SKLEARN_OK = True
try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    _SKLEARN_OK = False


# =============================================================================
# ACTIVE LEARNING ENGINE
# =============================================================================
class ActiveLearningEngine:
    """
    Handles active learning suggestions based on labeled points or spans.
    Supports 'point', 'span', and 'hybrid'.
    """
    def __init__(self, mode: str = "hybrid", threshold: float = 0.85, window: int = 3):
        self.mode = mode.lower()
        self.threshold = float(threshold)
        self.window = max(1, int(window))

        self._signal: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None

        self._point_labels: Dict[int, str] = {}
        self._span_labels: List[Tuple[int, int, str]] = []

        self._label_to_int: Dict[str, int] = {}
        self._int_to_label: Dict[int, str] = {}

        self._clf_knn = None
        self._clf_rf = None

        self.last_suggestions: List[int] = []

    # ---------------- Configuration ----------------
    def set_mode(self, mode: str):
        self.mode = (mode or "hybrid").lower()

    def set_threshold(self, th: float):
        self.threshold = float(th)

    # ---------------- Signal Fit ----------------
    def fit(self, signal: np.ndarray):
        signal = np.asarray(signal, dtype=float).flatten()
        if signal.size < 2:
            signal = np.pad(signal, (0, 2 - signal.size), mode='edge')
        self._signal = signal
        self._X = self._make_features(signal)
        self._clf_knn = None
        self._clf_rf = None
        debug_print("ActiveLearningEngine: Signal fitted.")

    def _make_features(self, x: np.ndarray) -> np.ndarray:
        """
        Generate features per index:
        [ value, diff1, diff2, local_mean, local_std, local_min, local_max, slope ]
        """
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

    # ---------------- Label Collection ----------------
    def add_point(self, idx: int, label: str):
        if self._signal is None:
            return
        idx = int(np.clip(idx, 0, len(self._signal) - 1))
        self._point_labels[idx] = label

    def add_span(self, start: int, end: int, label: str):
        if self._signal is None:
            return
        a = int(np.clip(min(start, end), 0, len(self._signal) - 1))
        b = int(np.clip(max(start, end), 0, len(self._signal) - 1))
        self._span_labels.append((a, b, label))

    def _get_labeled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._signal is None or self._X is None:
            return np.empty((0, 8)), np.empty((0,), dtype=int)

        X = self._X
        n = len(X)

        idxs: List[int] = []
        labels: List[str] = []

        if self.mode in ("point", "hybrid"):
            for idx, lab in self._point_labels.items():
                if 0 <= idx < n:
                    idxs.append(idx)
                    labels.append(lab)

        if self.mode in ("span", "hybrid"):
            for a, b, lab in self._span_labels:
                for i in range(a, b + 1):
                    idxs.append(i)
                    labels.append(lab)

        if not idxs:
            return np.empty((0, 8)), np.empty((0,), dtype=int)

        unique = sorted(set(labels))
        self._label_to_int = {lab: i for i, lab in enumerate(unique)}
        self._int_to_label = {i: lab for lab, i in self._label_to_int.items()}
        y = np.array([self._label_to_int[l] for l in labels], dtype=int)
        X_lab = X[np.array(idxs, dtype=int)]
        return X_lab, y

    # ---------------- Suggestion ----------------
    def suggest(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._signal is None or self._X is None:
            return np.array([], dtype=int), np.array([], dtype=str)

        X_lab, y_lab = self._get_labeled_data()
        if X_lab.size == 0:
            return np.array([], dtype=int), np.array([], dtype=str)

        n_classes = len(set(y_lab))

        if _SKLEARN_OK:
            if self._clf_knn is None:
                self._clf_knn = Pipeline([
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
                ])
            self._clf_knn.fit(X_lab, y_lab)
            proba_knn = self._clf_knn.predict_proba(self._X)
            proba = proba_knn

            if n_classes >= 2 and len(y_lab) > 20:
                if self._clf_rf is None:
                    self._clf_rf = Pipeline([
                        ("scaler", StandardScaler()),
                        ("rf", RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1))
                    ])
                self._clf_rf.fit(X_lab, y_lab)
                proba_rf = self._clf_rf.predict_proba(self._X)
                proba = 0.6 * proba_knn + 0.4 * proba_rf

            conf = np.max(proba, axis=1)
            pred_labels = np.argmax(proba, axis=1)
        else:
            # cosine-ish similarity fallback
            X_norm = self._X / (np.linalg.norm(self._X, axis=1, keepdims=True) + 1e-8)
            X_lab_norm = X_lab / (np.linalg.norm(X_lab, axis=1, keepdims=True) + 1e-8)
            sim = X_norm.dot(X_lab_norm.T)
            conf = np.max(sim, axis=1)
            pred_labels = y_lab[np.argmax(sim, axis=1)]

        # filter confident & not already labeled
        labeled_points = set(self._point_labels.keys())
        span_ranges = [(a, b) for a, b, _ in self._span_labels]

        def already_labeled(i):
            if i in labeled_points:
                return True
            for a, b in span_ranges:
                if a <= i <= b:
                    return True
            return False

        mask = (conf >= self.threshold)
        idxs = np.where(mask)[0]
        idxs = np.array([i for i in idxs if not already_labeled(i)], dtype=int)

        if idxs.size == 0:
            self.last_suggestions = []
            return idxs, np.array([], dtype=str)

        labs = np.array([self._int_to_label[int(pred_labels[i])] for i in idxs], dtype=str)
        self.last_suggestions = idxs.tolist()
        return idxs, labs


# =============================================================================
# InteractiveSequencePanel
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
        super().__init__(parent, sequence, title, formula, draggable, visible_count, color)

        # Debug
        self.debug = DEBUG_MODE

        # Performance cap for rendering
        self.MAX_VISIBLE_POINTS = MAX_VISIBLE_POINTS

        # Core
        self.role = role.lower().strip()
        self.on_update = on_update
        self.allow_edit_title = allow_edit_title
        self.allow_delete = allow_delete

        # Refs
        self.original_seq = getattr(self, "original_seq", None)  # set by SequencePanel creator
        self.annotator_ref = None  # for result shading

        # State
        self.play_idx = 0  # kept for external sync, but NOT drawn (timeline removed)
        self.hover_idx: Optional[int] = None
        self.spans: List[Dict[str, Any]] = []
        self.selected_idx: Optional[int] = None
        self.dragging = False
        self.frozen = False

        # Annotation mode toggle (point | region)
        self.annotation_mode = "point"  # default

        # Pan & Zoom
        self.zoom_factor = 1.0
        self.pan_offset = 0
        self.y_offset = 0.0
        self._space_pan_active = False
        self._ypan_active = False

        # Undo/Redo for numeric edits (annotator)
        self._undo_stack: List[np.ndarray] = []
        self._redo_stack: List[np.ndarray] = []

        # Undo/Redo for spans (annotation)
        self._span_undo_stack: List[List[Dict[str, Any]]] = []
        self._span_redo_stack: List[List[Dict[str, Any]]] = []

        # Peers
        self.sync_panels: List["InteractiveSequencePanel"] = []

        # Dragging helpers / transient state
        self._region_dragging = False
        self._region_start = None
        self._region_current = None
        self._span_edit = None      # {"index": idx, "which": "left/right/body", "last_x": x}
        self._span_idx0 = None      # legacy Shift-drag start for spans

        # Throttled refresh
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

        self._request_refresh = _request_refresh_local

        # Global drag mode from MainFrame (default preview)
        self.drag_mode = getattr(wx.GetTopLevelParent(self), "drag_mode", "preview")
        # preview state during drag
        self._drag_preview_idx: Optional[int] = None
        self._drag_preview_value: Optional[float] = None

        if self.debug:
            debug_print(f"InteractiveSequencePanel initialized with role={self.role}, title={self.title}")

        # Active Learning
        self.al_engine = ActiveLearningEngine(mode=al_mode, threshold=al_threshold, window=3)
        base_signal = (
            np.asarray(self.original_seq, dtype=float)
            if self.original_seq is not None
            else np.asarray(self.seq, dtype=float)
        )
        self.al_engine.fit(base_signal)
        self.al_mode = al_mode
        self.al_threshold = al_threshold

        # Suggest overlay
        self._suggest_indices: List[int] = []
        self._suggest_labels: List[str] = []

        # Callbacks to MainFrame (optional)
        self._active_labels_callback = None
        self._active_spans_callback = None
        top = wx.GetTopLevelParent(self)
        if hasattr(top, "on_active_labels") and callable(getattr(top, "on_active_labels")):
            self._active_labels_callback = getattr(top, "on_active_labels")
        if hasattr(top, "on_active_labels_spans") and callable(getattr(top, "on_active_labels_spans")):
            self._active_spans_callback = getattr(top, "on_active_labels_spans")

        # UI & Events
        self._setup_ui()
        self._bind_role_safe_events()

    # ----------------------------------------------------------
    # Legend passthrough
    # ----------------------------------------------------------
    def get_annotation_legend(self) -> Dict[str, Any]:
        return getattr(self, "annotation_legend", {})

    def set_annotation_legend(self, legend_map: Dict[str, Any]):
        self.annotation_legend = dict(legend_map or {})
        self._request_refresh()

    # ----------------------------------------------------------
    # Y-Range Sync (ALL panels use same Y-range as 'original')
    # ----------------------------------------------------------
    def _group_y_range(self) -> Tuple[float, float]:
        """
        Compute a shared y-range for all panels in the group:
        - Prefer 'original' panel's data
        - Fallback to this panel's data
        """
        # 1) Find original panel
        orig_panel = None
        if self.role == "original":
            orig_panel = self
        else:
            for sp in self.sync_panels:
                if getattr(sp, "role", "") == "original":
                    orig_panel = sp
                    break

        def _range_from_seq(seq):
            arr = np.asarray(seq, dtype=float)
            if arr.size == 0:
                return 0.0, 1.0
            mn = float(np.nanmin(arr))
            mx = float(np.nanmax(arr))
            if not np.isfinite(mn) or not np.isfinite(mx):
                return 0.0, 1.0
            if mn == mx:
                mn -= 0.5
                mx += 0.5
            return mn, mx

        # Prefer original panel's sequence
        if orig_panel is not None and getattr(orig_panel, "seq", None) is not None:
            return _range_from_seq(orig_panel.seq)

        # Fallback: this panel's sequence
        return _range_from_seq(self.seq)

    def get_y_range(self) -> Tuple[float, float]:
        """
        Override SequencePanel.get_y_range to ensure
        all synced panels share the same Y-range.
        """
        return self._group_y_range()

    # ----------------------------------------------------------
    # SIMPLE FREEZE HANDLER (GLOBAL FREEZE FOR ANNOTATION)
    # ----------------------------------------------------------
    def set_frozen(self, flag: bool):
        """
        Simple freeze: when True, this annotation panel cannot be edited AND
        should not be automatically updated from annotator changes.

        - Blocks:
          * span edits
          * region creation
          * delete-span
        - Keeps:
          * pan, zoom, hover (pure visualization)
        """
        self.frozen = bool(flag)

        # Clear all transient edit states
        self.dragging = False
        self.selected_idx = None
        self.hover_idx = None

        self._region_dragging = False
        self._region_start = None
        self._region_current = None

        self._span_edit = None
        self._span_idx0 = None

        self._drag_preview_idx = None
        self._drag_preview_value = None

        # Keep toggle button in sync (if exists)
        if hasattr(self, "btn_freeze") and self.btn_freeze:
            try:
                self.btn_freeze.SetValue(self.frozen)
                self.btn_freeze.SetLabel("🔒 Frozen" if self.frozen else "🔓 Unfrozen")
            except Exception:
                pass

        self._request_refresh()

    # ----------------------------------------------------------
    # Role-Safe Event Binding
    # ----------------------------------------------------------
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
            wx.EVT_PAINT: "on_paint",
        }
        for evt_type, handler_name in bindings_common.items():
            if hasattr(self, handler_name):
                self.Bind(evt_type, getattr(self, handler_name))

        # Left click interactions are only meaningful on annotator/annotation
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

    # ----------------------------------------------------------
    # UI (title row) + Mode + AL controls
    # ----------------------------------------------------------
    def _setup_ui(self):
        top = self.GetSizer()
        if top is None:
            top = wx.BoxSizer(wx.VERTICAL)
            self.SetSizer(top)
        else:
            top.Clear()

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.title_label = wx.StaticText(self, label=self.title, style=wx.ALIGN_CENTER)
        self.title_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT,
                                         wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        row.Add(self.title_label, 1, wx.ALIGN_CENTER_VERTICAL)

        # Edit Title (not for original)
        if self.allow_edit_title and self.role != "original":
            btn_title = wx.Button(self, label="Edit Title", size=(90, -1))
            btn_title.Bind(wx.EVT_BUTTON, self.edit_title)
            row.Add(btn_title, 0, wx.LEFT, 6)

        # Mode switch (point / region) for annotator & annotation
        if self.role in ("annotator", "annotation"):
            row.Add(wx.StaticText(self, label="Mode:"),
                    0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
            self.mode_choice = wx.Choice(self, choices=["point", "region"])
            try:
                self.mode_choice.SetStringSelection(self.annotation_mode)
            except Exception:
                self.mode_choice.SetSelection(0)
            self.mode_choice.Bind(wx.EVT_CHOICE, self._on_mode_changed)
            row.Add(self.mode_choice, 0, wx.LEFT, 4)

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
            row.Add(wx.StaticText(self, label="AL:"),
                    0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)

            self.al_mode_choice = wx.Choice(self, choices=["point", "span", "hybrid"])
            try:
                self.al_mode_choice.SetStringSelection(self.al_mode)
            except Exception:
                self.al_mode_choice.SetSelection(2)  # hybrid
            self.al_mode_choice.Bind(wx.EVT_CHOICE, self._on_al_mode_changed)
            row.Add(self.al_mode_choice, 0, wx.RIGHT, 6)

            row.Add(wx.StaticText(self, label="τ:"),
                    0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
            self.al_thresh_ctrl = wx.SpinCtrlDouble(
                self, min=0.50, max=0.99,
                initial=float(self.al_threshold),
                inc=0.01, size=(80, -1)
            )
            self.al_thresh_ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_al_threshold_changed)
            row.Add(self.al_thresh_ctrl, 0, wx.RIGHT, 6)

            btn_suggest = wx.Button(self, label="Suggest", size=(90, -1))
            btn_suggest.Bind(wx.EVT_BUTTON, self._on_al_suggest_clicked)
            row.Add(btn_suggest, 0, wx.LEFT, 4)

        top.Add(row, 0, wx.EXPAND | wx.ALL, 5)
        self.Layout()

    # ----------------------------------------------------------
    # Title / Formula / Freeze / Delete
    # ----------------------------------------------------------
    def edit_title(self, evt):
        dlg = wx.TextEntryDialog(self, "Enter new title:", "Edit Title", self.title)
        if dlg.ShowModal() == wx.ID_OK:
            self.title = dlg.GetValue()
            self.title_label.SetLabel(self.title)
            self.Layout()
            self._request_refresh()
            # refresh legend in MainFrame if provided
            top = wx.GetTopLevelParent(self)
            if hasattr(top, "_refresh_legend"):
                try:
                    top._refresh_legend()
                except Exception:
                    pass
        dlg.Destroy()

    def edit_formula(self, evt):
        dlg = wx.TextEntryDialog(self, "Enter new formula:",
                                 "Edit Formula", self.formula or "")
        if dlg.ShowModal() == wx.ID_OK:
            self.formula = dlg.GetValue()
            self._request_refresh()
        dlg.Destroy()

    def toggle_freeze(self, evt=None):
        """
        Called when the annotation panel freeze toggle is pressed.
        Uses set_frozen() so all transient edit state is cleared.
        """
        value = bool(self.btn_freeze.GetValue())
        self.set_frozen(value)

    def delete_self(self, evt):
        top = wx.GetTopLevelParent(self)
        if hasattr(top, "remove_panel"):
            top.remove_panel(self)

    # ----------------------------------------------------------
    # Mode switching
    # ----------------------------------------------------------
    def _on_mode_changed(self, evt=None):
        try:
            self.annotation_mode = self.mode_choice.GetStringSelection()
        except Exception:
            self.annotation_mode = "point"
        if self.debug:
            debug_print(f"[{self.title}] annotation_mode -> {self.annotation_mode}")
        # If switching to region, ensure no lingering 'point' drag state
        if self.annotation_mode == "region":
            self.dragging = False
            self.selected_idx = None
            self._drag_preview_idx = None
            self._drag_preview_value = None
        self._request_refresh()

    # ----------------------------------------------------------
    # Undo/Redo helpers (numeric, annotator)
    # ----------------------------------------------------------
    def _push_undo(self):
        """
        Snapshot annotator sequence before a numeric change.
        (Span undo is handled separately in annotation panel.)
        """
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
        # Propagation to peers will be handled in Part 2/_propagate_from_annotator
        cb = getattr(self, "on_update", None)
        if callable(cb):
            try:
                cb(self.seq.copy())
            except Exception:
                pass
        self._request_refresh()

    def _redo(self):
        if self.role != "annotator" or not self._redo_stack:
            return
        nxt = self._redo_stack.pop()
        self._undo_stack.append(np.asarray(self.seq, dtype=float).copy())
        self.seq = nxt.copy()
        # Propagation to peers will be handled in Part 2/_propagate_from_annotator
        cb = getattr(self, "on_update", None)
        if callable(cb):
            try:
                cb(self.seq.copy())
            except Exception:
                pass
        self._request_refresh()

    # ----------------------------------------------------------
    # Span Undo/Redo (annotation only) - will be used for regions
    # ----------------------------------------------------------
    def _push_span_undo(self):
        if self.role != "annotation":
            return
        try:
            snapshot = [dict(s) for s in self.spans]
            self._span_undo_stack.append(snapshot)
            if len(self._span_undo_stack) > 200:
                self._span_undo_stack.pop(0)
            self._span_redo_stack.clear()
        except Exception:
            pass

    def _span_undo(self):
        if self.role != "annotation" or not self._span_undo_stack:
            return
        current = [dict(s) for s in self.spans]
        prev = self._span_undo_stack.pop()
        self._span_redo_stack.append(current)
        self.spans = [dict(s) for s in prev]
        self._request_refresh()
        # Legend refresh in MainFrame
        top = wx.GetTopLevelParent(self)
        if hasattr(top, "_refresh_legend"):
            try:
                top._refresh_legend()
            except Exception:
                pass

    def _span_redo(self):
        if self.role != "annotation" or not self._span_redo_stack:
            return
        current = [dict(s) for s in self.spans]
        nxt = self._span_redo_stack.pop()
        self._span_undo_stack.append(current)
        self.spans = [dict(s) for s in nxt]
        self._request_refresh()
        top = wx.GetTopLevelParent(self)
        if hasattr(top, "_refresh_legend"):
            try:
                top._refresh_legend()
            except Exception:
                pass

    # ----------------------------------------------------------
    # Span helpers & hit testing
    # ----------------------------------------------------------
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

    def _hit_span_edge(self, x_px, _y_px, w, _h, tol=4):
        left_idx, right_idx = self.get_visible_index_range()
        for i, s in enumerate(self.spans or []):
            a, b = int(s.get("start", 0)), int(s.get("end", 0))
            if b < a:
                a, b = b, a
            if b <= left_idx or a >= right_idx:
                continue
            ax, _ = self.to_px(a, self.seq[a if 0 <= a < self.n else 0],
                               w, self.GetClientSize().height)
            bx, _ = self.to_px(b, self.seq[b if 0 <= b < self.n else self.n - 1],
                               w, self.GetClientSize().height)
            if abs(x_px - ax) <= tol:
                return i, "left"
            if abs(x_px - bx) <= tol:
                return i, "right"
            if min(ax, bx) + tol < x_px < max(ax, bx) - tol:
                return i, "body"
        return None, None

    # ==============================================================
    # Mouse / interaction handlers
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
            if self.frozen:
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
                s["start"] += di
                s["end"] += di
            self._normalize_span(s)
            self._request_refresh()
            return

        # Region dragging (annotator or annotation) when mode == region
        if self.role in ("annotator", "annotation") and self.annotation_mode == "region":
            if getattr(self, "_region_dragging", False) and evt.LeftIsDown():
                if self.role == "annotation" and self.frozen:
                    return
                w, _ = self.GetClientSize()
                self._region_current = self._x_to_index(evt.GetX(), w)
                self._request_refresh()
                return

        # Point dragging (annotator only) when mode == "point"
        if self.role == "annotator" and self.annotation_mode == "point" and getattr(self, "dragging", False) and evt.LeftIsDown():
            self._handle_drag(evt)  # preview/realtime handled inside
            return

        # Otherwise: hover
        self._handle_hover(evt)

    def on_left_down(self, evt):
        try:
            self.SetFocus()
        except Exception:
            pass

        w, _h = self.GetClientSize()
        x = evt.GetX()

        # Region mode (annotator/annotation): start region drag
        if self.role in ("annotator", "annotation") and self.annotation_mode == "region":
            if self.role == "annotation" and self.frozen:
                return
            self._region_dragging = True
            self._region_start = self._x_to_index(x, w)
            self._region_current = self._region_start
            try:
                self.CaptureMouse()
            except Exception:
                pass
            self._request_refresh()
            return

        # Annotation: hit test span edges/body for editing
        if self.role == "annotation":
            idx, which = self._hit_span_edge(x, evt.GetY(), w, _h)
            if which is not None:
                if self.frozen:
                    # When frozen, allow selection but no move (for delete via key)
                    self._span_edit = {"index": idx, "which": which, "last_x": x}
                    return
                # editable span drag
                self._push_span_undo()
                self._span_edit = {"index": idx, "which": which, "last_x": x}
                try:
                    self.CaptureMouse()
                except Exception:
                    pass
                return
            # If not region mode, allow Shift new-span shortcut as legacy
            if self.annotation_mode != "region" and evt.ShiftDown() and not self.frozen:
                self._span_idx0 = self._x_to_index(x, w)
                self._span_edit = None
                return

        # Avoid starting a drag if A is held (Y-pan)
        if wx.GetKeyState(ord('A')):
            return

        # Point mode (annotator): select nearest point for dragging
        if self.role == "annotator" and self.annotation_mode == "point":
            x0, y0 = evt.GetPosition()
            best_d = float("inf")
            best_i = None

            # restrict search to visible range
            left, right = self.get_visible_index_range()
            left = max(0, left)
            right = min(self.n, max(left + 1, right))
            for i in range(left, right):
                v = self.seq[i]
                try:
                    cx, cy = self.to_px(i, v, *self.GetClientSize())
                except Exception:
                    continue
                d = (cx - x0) ** 2 + (cy - y0) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i

            hit_radius = 8
            if hasattr(self, "radius"):
                hit_radius = max(6, int(self.radius * 1.5))
            if best_i is not None and best_d < (hit_radius ** 2):
                self.selected_idx = best_i
                self.dragging = True
                self._push_undo()
                # Initialize preview state
                self._drag_preview_idx = best_i
                self._drag_preview_value = float(self.seq[best_i])
                try:
                    self.CaptureMouse()
                except Exception:
                    pass

    def on_left_up(self, evt):
        # End Y-pan if active
        if self._ypan_active:
            self._ypan_active = False
            self._ypan_start_y = None

        # Finish span edit (annotation)
        if self.role == "annotation" and getattr(self, "_span_edit", None):
            # No extra change here, span already modified
            self._span_edit = None
            if self.HasCapture():
                try:
                    self.ReleaseMouse()
                except Exception:
                    pass
            self._request_refresh()
            return

        # Finish region selection (annotator/annotation)
        if getattr(self, "_region_dragging", False):
            self._region_dragging = False
            if self.HasCapture():
                try:
                    self.ReleaseMouse()
                except Exception:
                    pass

            a = int(self._region_start)
            b = int(self._region_current)
            if a != b:
                s = {"start": min(a, b), "end": max(a, b),
                     "label": self.title, "value": None}
                self._normalize_span(s)

                # Store spans:
                # - If this is an annotation panel: store locally (with undo)
                # - If this is annotator: forward span to the annotation panel
                if self.role == "annotation":
                    if not self.frozen:
                        self._push_span_undo()
                        self.spans.append(s)
                        if self.al_engine.mode in ("span", "hybrid"):
                            self.al_engine.add_span(s["start"], s["end"], self.title)
                elif self.role == "annotator":
                    # self keeps copy as well (for highlighting if desired)
                    self.spans.append(s)
                    if self.al_engine.mode in ("span", "hybrid"):
                        self.al_engine.add_span(s["start"], s["end"], self.title)

                    # Forward region to annotation peer(s)
                    for sp in self.sync_panels:
                        if getattr(sp, "role", "") == "annotation" and not sp.frozen:
                            sp._push_span_undo()
                            sp.spans.append(dict(s))
                            if hasattr(sp, "al_engine"):
                                sp.al_engine.add_span(s["start"], s["end"], s["label"])
                            try:
                                sp._request_refresh()
                            except Exception:
                                sp.Refresh(False)

                # legend refresh
                top = wx.GetTopLevelParent(self)
                if hasattr(top, "_refresh_legend"):
                    try:
                        top._refresh_legend()
                    except Exception:
                        pass

            self._region_start = None
            self._region_current = None
            self._request_refresh()
            return

        # Finish legacy span creation via Shift (annotation only)
        if (
            self.role == "annotation"
            and getattr(self, "_span_idx0", None) is not None
            and evt.ShiftDown()
            and not self.frozen
        ):
            w, _h = self.GetClientSize()
            i0 = int(self._span_idx0)
            i1 = self._x_to_index(evt.GetX(), w)
            if i1 != i0:
                self._push_span_undo()
                s = {"start": i0, "end": i1,
                     "label": self.title, "value": None}
                self._normalize_span(s)
                self.spans.append(s)
                if self.al_engine.mode in ("span", "hybrid"):
                    self.al_engine.add_span(s["start"], s["end"], self.title)
            self._span_idx0 = None
            self._request_refresh()
            # legend refresh
            top = wx.GetTopLevelParent(self)
            if hasattr(top, "_refresh_legend"):
                try:
                    top._refresh_legend()
                except Exception:
                    pass
            return

        # Finish dragging (annotator point mode) -> apply once + propagate
        if self.role == "annotator" and self.annotation_mode == "point" and getattr(self, "dragging", False):
            self.dragging = False
            if self.HasCapture():
                try:
                    self.ReleaseMouse()
                except Exception:
                    pass

            applied = False
            if self.selected_idx is not None:
                # If preview mode, commit the pending value now
                if (
                    self.drag_mode == "preview"
                    and self._drag_preview_idx == self.selected_idx
                    and self._drag_preview_value is not None
                ):
                    seq_np = np.asarray(self.seq, dtype=float)
                    if 0 <= self.selected_idx < seq_np.size:
                        seq_np[self.selected_idx] = float(self._drag_preview_value)
                        self.seq = seq_np
                        applied = True
                elif self.drag_mode == "realtime":
                    applied = True  # already applied during motion

                # AL point memory
                if applied and self.al_engine.mode in ("point", "hybrid"):
                    self.al_engine.add_point(int(self.selected_idx), self.title)

            # Clear preview state
            self._drag_preview_idx = None
            self._drag_preview_value = None

            # Propagate once to annotation + result
            if applied:
                self._propagate_from_annotator()

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
                    try:
                        sp._request_refresh()
                    except Exception:
                        sp.Refresh(False)
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
            try:
                sp._request_refresh()
            except Exception:
                sp.Refresh(False)

        self._request_refresh()

    # ==============================================================
    # Pan helpers (sync X/Y to peers)
    # ==============================================================
    def _apply_pan_x(self, dx):
        new_offset = getattr(self, "_pan_origin", 0) + dx
        w, _ = self.GetClientSize()
        gw = max(1, w - 2 * self.padding)
        min_offset = gw * (1 - self.zoom_factor)
        self.pan_offset = min(max(new_offset, min_offset), 0)
        for sp in self.sync_panels:
            sp.pan_offset = self.pan_offset
            try:
                sp._request_refresh()
            except Exception:
                sp.Refresh(False)
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
                try:
                    sp._request_refresh()
                except Exception:
                    sp.Refresh(False)
        self._request_refresh()

    # ==============================================================
    # Propagation from annotator -> annotation & result
    # ==============================================================
    def _propagate_from_annotator(self):
        """
        Called when the annotator sequence has changed (drag/undo/redo/nudge).
        - Updates annotation panel(s) numerically: annotation = original - annotator
        - Updates result panel(s): result = original + annotation
        - Respects annotation.frozen (no overwrite when frozen)
        """
        if self.role != "annotator":
            return

        annot = np.asarray(self.seq, dtype=float)

        # Find annotation and result panels
        annotation_panels = [sp for sp in self.sync_panels if getattr(sp, "role", "") == "annotation"]
        result_panels = [sp for sp in self.sync_panels if getattr(sp, "role", "") == "result"]

        # Update annotation panels (unless frozen)
        for ap in annotation_panels:
            if ap.frozen:
                continue
            if getattr(ap, "original_seq", None) is None:
                continue
            orig = np.asarray(ap.original_seq, dtype=float)
            m = min(len(orig), len(annot))
            if m <= 0:
                continue
            new_ann = orig[:m] - annot[:m]
            if getattr(ap, "n", None) is not None:
                padded = np.zeros(ap.n, dtype=float)
                padded[:m] = new_ann
                ap.seq = padded.tolist()
            else:
                ap.seq = new_ann.tolist()
            try:
                ap._request_refresh()
            except Exception:
                ap.Refresh(False)

        # Determine primary annotation sequence for results
        primary_ann = None
        primary_orig = None
        if annotation_panels:
            ap0 = annotation_panels[0]
            if getattr(ap0, "original_seq", None) is not None:
                primary_orig = np.asarray(ap0.original_seq, dtype=float)
                primary_ann = np.asarray(ap0.seq, dtype=float)
        else:
            # Fallback: use any result's original_seq with current annot
            for rp in result_panels:
                if getattr(rp, "original_seq", None) is not None:
                    primary_orig = np.asarray(rp.original_seq, dtype=float)
                    # Equivalent of annotation = original - annotator
                    m = min(len(primary_orig), len(annot))
                    if m > 0:
                        primary_ann = primary_orig[:m] - annot[:m]
                    break

        # Update result panels using: result = original + annotation
        if primary_orig is not None and primary_ann is not None:
            for rp in result_panels:
                orig = np.asarray(rp.original_seq, dtype=float) if getattr(rp, "original_seq", None) is not None else primary_orig
                ann = primary_ann
                m = min(len(orig), len(ann))
                if m <= 0:
                    continue
                new_res = orig[:m] + ann[:m]
                if getattr(rp, "n", None) is not None:
                    padded = np.zeros(rp.n, dtype=float)
                    padded[:m] = new_res
                    rp.seq = padded.tolist()
                else:
                    rp.seq = new_res.tolist()
                # Keep a reference to annotator for shaded diff
                rp.annotator_ref = self
                try:
                    rp._request_refresh()
                except Exception:
                    rp.Refresh(False)

        # Fire single update callback (let the model recompute once)
        if callable(self.on_update):
            try:
                self.on_update(np.asarray(self.seq, dtype=float).copy())
            except Exception:
                pass

    # ==============================================================
    # Drag handling (annotator) - preview vs realtime
    # ==============================================================
    def _handle_drag(self, evt):
        if self.selected_idx is None:
            return

        # Compute value under cursor
        _x, y = evt.GetPosition()
        w, h = self.GetClientSize()
        mn, mx = self._get_y_display_range()
        rng = mx - mn or 1.0
        val = ((h - self.padding - y) / max(1, (h - 2 * self.padding))) * rng + mn
        val = max(min(val, mx), mn)

        if self.drag_mode == "preview":
            # do NOT mutate seq while dragging
            self._drag_preview_idx = int(self.selected_idx)
            self._drag_preview_value = float(val)
            # Only this panel repaints now (no propagation during drag)
            self._request_refresh()
            return

        # Realtime: update local sequence only (no propagation during drag)
        if self.drag_mode == "realtime":
            seq_np = np.asarray(self.seq, dtype=float)
            if 0 <= self.selected_idx < seq_np.size:
                seq_np[self.selected_idx] = float(val)
                self.seq = seq_np
            # keep UI snappy: repaint just this panel
            self._request_refresh()

    def _trigger_on_update(self):
        cb = getattr(self, "on_update", None) or getattr(self, "on_update_callback", None)
        if callable(cb):
            try:
                cb(np.asarray(self.seq, dtype=float).copy())
            except Exception:
                pass

    # ==============================================================
    # Hover handler (sync hover to peers)
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
            try:
                sp._request_refresh()
            except Exception:
                pass
        self._request_refresh()
        try:
            evt.Skip()
        except Exception:
            pass

    # ==============================================================
    # Focus (optional visual cue; avoid heavy work)
    # ==============================================================
    def on_focus(self, evt):
        evt.Skip()

    def on_kill_focus(self, evt):
        evt.Skip()

    # ==============================================================
    # Rendering (fast/decimated) + markers (no repaint loops)
    # ==============================================================
    def _get_y_display_range(self):
        mn, mx = self.get_y_range()
        return mn + self.y_offset, mx + self.y_offset

    def on_paint(self, evt):
        w, h = self.GetClientSize()
        if w < 2 or h < 2 or self.n < 2:
            dc = wx.AutoBufferedPaintDC(self)
            dc.Clear()
            return

        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return

        # Ensure a default font is always set
        try:
            font = wx.Font(10, wx.FONTFAMILY_DEFAULT,
                           wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
            gc.SetFont(font, wx.BLACK)
        except Exception:
            pass

        # Background
        gc.SetBrush(wx.Brush(wx.WHITE))
        gc.DrawRectangle(0, 0, w, h)

        # Axes
        gc.SetPen(wx.Pen(wx.LIGHT_GREY))
        gc.StrokeLine(self.padding, self.padding,
                      self.padding, h - self.padding)          # Y-axis
        gc.StrokeLine(self.padding, h - self.padding,
                      w - self.padding, h - self.padding)      # X-axis

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
            title_font = wx.Font(12, wx.FONTFAMILY_DEFAULT,
                                 wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
            gc.SetFont(title_font, wx.BLACK)
            tw, _ = gc.GetTextExtent(self.title)
            gc.DrawText(self.title, (w - tw) / 2, 5)
        except Exception:
            pass

        # Clip to plot area
        gc.Clip(self.padding, self.padding,
                w - 2 * self.padding, h - 2 * self.padding)

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

        # Draw role visuals (line, markers (point mode only), or shaded region for result)
        self._render_role_visual(gc, pts, w, h, mn, mx, idxs)

        # Active Learning overlay markers (from set_active_suggestions)
        if self._suggest_indices:
            gc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 1))
            for ii in self._suggest_indices:
                if left_i <= ii < right_i:
                    xx, yy = self.to_px(ii, seq_arr[min(ii, self.n - 1)], w, h, mn, mx)
                    gc.StrokeLine(xx, self.padding, xx, h - self.padding)
                    gc.DrawEllipse(xx - 2, yy - 2, 4, 4)

        # Spans for this panel
        self._render_spans(gc, w, h, mn, mx)

        # Result panel: also show spans from all annotation panels in their color
        if self.role == "result":
            self._render_annotation_spans_in_result(gc, w, h, mn, mx)

        # Hover marker
        if getattr(self, "hover_idx", None) is not None and left_i <= self.hover_idx < right_i:
            xh, yh = self.to_px(self.hover_idx, seq_arr[self.hover_idx], w, h, mn, mx)
            gc.SetBrush(wx.Brush(wx.Colour(0, 160, 0)))
            gc.DrawEllipse(xh - 3, yh - 3, 6, 6)

        # Selected point (annotator, point mode) + preview indicator
        if self.role == "annotator" and self.annotation_mode == "point" and self.selected_idx is not None:
            if left_i <= self.selected_idx < right_i:
                xs, ys = self.to_px(self.selected_idx, seq_arr[self.selected_idx], w, h, mn, mx)
                gc.SetBrush(wx.Brush(wx.BLUE))
                gc.DrawEllipse(xs - 4, ys - 4, 8, 8)

        # Preview marker if in preview mode (annotator, point mode)
        if self.role == "annotator" and self.annotation_mode == "point" and self.drag_mode == "preview":
            if getattr(self, "_drag_preview_idx", None) is not None and self._drag_preview_value is not None:
                i = int(self._drag_preview_idx)
                if left_i <= i < right_i:
                    xp, yp = self.to_px(i, float(self._drag_preview_value), w, h, mn, mx)
                    # draw hollow circle + crosshair to indicate preview
                    gc.SetPen(wx.Pen(wx.Colour(30, 144, 255), 2))
                    gc.SetBrush(wx.Brush(wx.Colour(30, 144, 255, 40)))
                    gc.DrawEllipse(xp - 5, yp - 5, 10, 10)
                    gc.StrokeLine(xp, yp - 8, xp, yp + 8)
                    gc.StrokeLine(xp - 8, yp, xp + 8, yp)

        # Sync hover & playhead to peers (no forced repaint to avoid loops)
        for sp in self.sync_panels:
            sp.hover_idx = self.hover_idx
            sp.play_idx = self.play_idx

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
            # Only draw the line; no per-point markers (points shown via hover/selection)
            gc.SetPen(wx.Pen(self.color, 1))
            gc.StrokePath(path)

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
            # fallback: draw own line
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

        # Filled polygon between orig and annotator
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

        # Draw the original line on top for clarity
        path_orig = gc.CreatePath()
        path_orig.MoveToPoint(*pts_orig[0])
        for p in pts_orig[1:]:
            path_orig.AddLineToPoint(*p)
        gc.SetPen(wx.Pen(wx.Colour(10, 60, 160), 1))
        gc.StrokePath(path_orig)

    # --------------------------------------------------------------
    def _render_spans(self, gc, w, h, mn, mx):
        """
        Background highlighted spans for THIS panel only.
        - Annotation (and annotator region) store spans in self.spans
        - Result panel gets additional spans from all annotation panels
          via _render_annotation_spans_in_result()
        """
        left_i, right_i = self.get_visible_index_range()

        # stored spans
        # stored spans
        if getattr(self, "spans", None):

            # base = panel color
            base = self.color if isinstance(self.color, wx.Colour) \
                else wx.Colour(255, 0, 0)

            fill_col = wx.Colour(base.Red(), base.Green(), base.Blue(), 60)
            edge_col = wx.Colour(base.Red(), base.Green(), base.Blue(), 160)

            gc.SetBrush(wx.Brush(fill_col))
            gc.SetPen(wx.Pen(edge_col, 1))

            for s in self.spans:
                a, b = int(s.get("start", 0)), int(s.get("end", 0))
                if b < a: a, b = b, a
                A = max(left_i, min(right_i, a))
                B = max(left_i, min(right_i, b))
                if B <= A:
                    continue

                x0, _ = self.to_px(A, mn, w, h, mn, mx)
                x1, _ = self.to_px(B, mn, w, h, mn, mx)

                gc.DrawRectangle(x0, self.padding, max(1, x1 - x0), h - 2 * self.padding)

        # live preview during region drag
        if self.role in ("annotator", "annotation") and self.annotation_mode == "region":
            if getattr(self, "_region_dragging", False) and self._region_start is not None and self._region_current is not None:
                a = min(self._region_start, self._region_current)
                b = max(self._region_start, self._region_current)
                A = max(left_i, min(right_i, a))
                B = max(left_i, min(right_i, b))
                if B > A:
                    x0, _ = self.to_px(A, mn, w, h, mn, mx)
                    x1, _ = self.to_px(B, mn, w, h, mn, mx)
                    gc.SetBrush(wx.Brush(wx.Colour(30, 144, 255, 60)))
                    gc.SetPen(wx.Pen(wx.Colour(30, 144, 255, 160)))
                    gc.DrawRectangle(x0, self.padding,
                                     max(1, x1 - x0), h - 2 * self.padding)

    # --------------------------------------------------------------
    def _render_annotation_spans_in_result(self, gc, w, h, mn, mx):
        """
        In result panel:
        draw spans from ALL annotation panels in their own color,
        so each annotation has a clear color in the result.
        """
        left_i, right_i = self.get_visible_index_range()

        for sp in self.sync_panels:
            if getattr(sp, "role", "") != "annotation":
                continue
            spans = getattr(sp, "spans", [])
            if not spans:
                continue

            # Use the annotation panel's color for its spans
            color = getattr(sp, "color", wx.Colour(255, 255, 0))
            brush = wx.Brush(wx.Colour(color.Red(), color.Green(), color.Blue(), 60))
            pen = wx.Pen(wx.Colour(color.Red(), color.Green(), color.Blue(), 160))

            gc.SetBrush(brush)
            gc.SetPen(pen)

            for s in spans:
                a, b = int(s.get("start", 0)), int(s.get("end", 0))
                if b < a:
                    a, b = b, a
                A = max(left_i, min(right_i, a))
                B = max(left_i, min(right_i, b))
                if B <= A:
                    continue
                x0, _ = self.to_px(A, mn, w, h, mn, mx)
                x1, _ = self.to_px(B, mn, w, h, mn, mx)
                gc.DrawRectangle(x0, self.padding,
                                 max(1, x1 - x0), h - 2 * self.padding)
    # ==============================================================
    # Active Learning (UI triggers)
    # ==============================================================
    def _on_al_mode_changed(self, evt=None):
        try:
            new_mode = self.al_mode_choice.GetStringSelection()
            self.al_engine.set_mode(new_mode)
            self.al_mode = new_mode
            if self.debug:
                debug_print(f"[{self.title}] AL mode -> {new_mode}")
        except Exception as e:
            if self.debug:
                debug_print(f"[{self.title}] AL mode error: {e}")

    def _on_al_threshold_changed(self, evt=None):
        try:
            th = float(self.al_thresh_ctrl.GetValue())
            self.al_engine.set_threshold(th)
            self.al_threshold = th
            if self.debug:
                debug_print(f"[{self.title}] AL threshold -> {th}")
        except Exception as e:
            if self.debug:
                debug_print(f"[{self.title}] AL threshold error: {e}")

    def _on_al_suggest_clicked(self, evt=None):
        """
        Run Active Learning suggestion:
        - uses points / spans / hybrid depending on mode
        - highlights suggested indices
        - optionally reports labels to MainFrame
        """
        if self.al_engine is None or self.seq is None:
            return

        try:
            idxs, labels = self.al_engine.suggest()
        except Exception as e:
            if self.debug:
                debug_print(f"[{self.title}] AL suggest error: {e}")
            idxs, labels = np.array([], dtype=int), np.array([], dtype=str)

        self._suggest_indices = idxs.tolist()
        self._suggest_labels = labels.tolist()

        # Notify MainFrame so it can update the legend or table
        try:
            if self._active_labels_callback:
                self._active_labels_callback(self._suggest_indices, self._suggest_labels)
        except Exception:
            pass

        self._request_refresh()

    # ==============================================================
    # Keyboard handling: Undo/Redo, Delete, AL shortcuts
    # ==============================================================
    def on_key_down(self, evt):
        code = evt.GetKeyCode()

        # CTRL+Z → Undo
        if evt.ControlDown() and code == ord('Z'):
            self._handle_undo_hotkey()
            return

        # CTRL+Y → Redo
        if evt.ControlDown() and code == ord('Y'):
            self._handle_redo_hotkey()
            return

        # Delete → remove selected span (annotation only)
        if code == wx.WXK_DELETE:
            self._handle_delete_hotkey()
            return

        # CTRL+S → Suggest AL
        if evt.ControlDown() and code == ord('S'):
            self._on_al_suggest_clicked(None)
            return

        evt.Skip()

    # ----- Undo hotkey -----
    def _handle_undo_hotkey(self):
        if self.role == "annotator":
            self._undo()
            self._propagate_from_annotator()
            return
        if self.role == "annotation":
            self._span_undo()
            return

    # ----- Redo hotkey -----
    def _handle_redo_hotkey(self):
        if self.role == "annotator":
            self._redo()
            self._propagate_from_annotator()
            return
        if self.role == "annotation":
            self._span_redo()
            return

    # ----- Delete hotkey -----
    def _handle_delete_hotkey(self):
        """
        Delete the span under the cursor (annotation only).
        Cursor must lie inside the span or on an edge.
        """
        if self.role != "annotation" or self.frozen:
            return

        x, y = wx.GetMousePosition()
        try:
            x, y = self.ScreenToClient((x, y))
        except Exception:
            return

        w, h = self.GetClientSize()
        idx, which = self._hit_span_edge(x, y, w, h, tol=6)
        # If none found: look for span body
        if idx is None:
            idx = self._hit_span_body(x, y, w, h)
        if idx is None:
            return

        # Delete it with undo support
        self._push_span_undo()
        try:
            self.spans.pop(idx)
        except Exception:
            return

        self._request_refresh()
        top = wx.GetTopLevelParent(self)
        if hasattr(top, "_refresh_legend"):
            try:
                top._refresh_legend()
            except Exception:
                pass

    def _hit_span_body(self, x_px, y_px, w, h):
        """Hit test for clicking inside the body of a span."""
        left_i, right_i = self.get_visible_index_range()
        for i, s in enumerate(self.spans):
            a, b = int(s.get("start", 0)), int(s.get("end", 0))
            if b < a:
                a, b = b, a
            if b <= left_i or a >= right_i:
                continue
            x0, _ = self.to_px(a, self.seq[a], w, h, *self._get_y_display_range())
            x1, _ = self.to_px(b, self.seq[b], w, h, *self._get_y_display_range())
            if x0 <= x_px <= x1:
                return i
        return None

    # ==============================================================
    # Right-click context menu (Delete Span)
    # ==============================================================
    def on_right_down(self, evt):
        super().on_right_down(evt)
        if self.role != "annotation" or self.frozen:
            return

        # Build a context menu
        menu = wx.Menu()
        id_delete = wx.NewIdRef()

        menu.Append(id_delete, "Delete Span")

        def _do_delete(_evt):
            self._handle_delete_hotkey()

        self.Bind(wx.EVT_MENU, _do_delete, id=id_delete)
        self.PopupMenu(menu)
        menu.Destroy()

    # ==============================================================
    # Playhead helpers (optional, minimal)
    # ==============================================================
    def set_play_idx(self, idx: int):
        """Keep API compatible; does not draw any playhead line."""
        idx = max(0, min(idx, self.n - 1))
        self.play_idx = idx
        for sp in self.sync_panels:
            sp.play_idx = idx
        self._request_refresh()

    # ==============================================================
    # Utility: finish any pending edit / cleanup mouse capture
    # ==============================================================
    def _finish_state(self):
        # End any drag/previews
        self.dragging = False
        self._drag_preview_idx = None
        self._drag_preview_value = None

        # End region
        self._region_dragging = False
        self._region_start = None
        self._region_current = None

        # End span edit
        self._span_edit = None

        # Release mouse if needed
        if self.HasCapture():
            try:
                self.ReleaseMouse()
            except Exception:
                pass

        self._request_refresh()

    # ==============================================================
    # FORCE CLEAR OF HOVER / SELECTION
    # ==============================================================
    def clear_hover_selection(self):
        self.hover_idx = None
        self.selected_idx = None
        self._request_refresh()

    # ==============================================================
    # Programmatic clearing of suggestions
    # ==============================================================
    def clear_suggestions(self):
        self._suggest_indices = []
        self._suggest_labels = []
        self._request_refresh()

    # ==============================================================
    # Programmatic adding of spans (from load_project)
    # ==============================================================
    def add_spans(self, span_list: List[Dict[str, Any]]):
        """
        Called by Load/Project Manager to restore spans safely.
        """
        if self.role != "annotation":
            return

        self.spans = [self._normalize_span(dict(s)) for s in span_list]
        self._request_refresh()

        # update AL
        for s in self.spans:
            if self.al_engine.mode in ("span", "hybrid"):
                try:
                    self.al_engine.add_span(s["start"], s["end"], s.get("label", self.title))
                except Exception:
                    pass

    # ==============================================================
    # Clean up AL when sequence changes externally
    # ==============================================================
    def refresh_active_learning(self):
        """
        Recompute features after sequence or original changes.
        """
        base_signal = np.asarray(self.original_seq, dtype=float) \
            if getattr(self, "original_seq", None) is not None \
            else np.asarray(self.seq, dtype=float)

        try:
            self.al_engine.fit(base_signal)
        except Exception as e:
            if self.debug:
                debug_print(f"[{self.title}] AL fit error: {e}")
        self._request_refresh()
