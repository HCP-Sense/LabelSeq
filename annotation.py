"""
annotation.py: Role-specific panel for an Annotation layer (non-draggable).
Thin wrapper around InteractiveSequencePanel that fixes role and defaults.
"""

import wx
import numpy as np
from typing import Optional, Callable, Any, Dict
from InteractiveSequencePanel import InteractiveSequencePanel


class AnnotationPanel(InteractiveSequencePanel):
    """
    AnnotationPanel is a specialized version of InteractiveSequencePanel
    that automatically configures itself for 'annotation' role.

    ✅ Supports spans (highlighted regions)
    ✅ Integrates Active Learning in hybrid mode
    ✅ Works with fast rendering and sync system
    """

    def __init__(
        self,
        parent,
        sequence: np.ndarray,
        title: Optional[str] = None,
        visible_count: Optional[int] = 200,
        color: wx.Colour = wx.Colour(255, 140, 0),
        al_mode: str = "hybrid",
        al_threshold: float = 0.85,
        **kwargs
    ):
        # Ensure proper role enforcement and forward AL arguments
        kwargs = dict(kwargs)
        kwargs["role"] = "annotation"
        kwargs["draggable"] = False  # make sure it is non-draggable
        kwargs["al_mode"] = al_mode
        kwargs["al_threshold"] = al_threshold

        # ✅ Only pass arguments through kwargs to avoid duplicates
        super().__init__(
            parent=parent,
            sequence=sequence,
            title=title or "Annotation",
            visible_count=visible_count,
            color=color,
            **kwargs
        )

    # === Custom Span Create Behavior (Optional) ===
    def on_span_created(self, span: dict):
        """
        Optional override if you want span labels to also write numeric values into
        the annotation array. If AL hybrid mode is active, AL engine will take over.
        """
        a = int(span.get("start", 0))
        b = int(span.get("end", a))
        if b < a:
            a, b = b, a
        b = min(b, self.n - 1)

        if a <= b:
            self.seq[a: b + 1] = 1.0  # default numeric label
            cb = getattr(self, "on_update", None)
            if callable(cb):
                cb(np.asarray(self.seq, dtype=float).copy())
            self._request_refresh()
