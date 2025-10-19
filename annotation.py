"""
annotation.py: Role-specific panel for an Annotation layer (non-draggable).
Thin wrapper around InteractiveSequencePanel that fixes role and defaults.
"""
from InteractiveSequencePanel import InteractiveSequencePanel
import wx
import numpy as np
from typing import Optional

from typing import Optional, Callable, Any, Dict

class AnnotationPanel(InteractiveSequencePanel):
    def on_span_created(self, span: dict):
        # Default behavior: fill annotation over [start:end] with either a constant (1.0)
        # or the current values (unchanged). Here we choose a flag value 1.0.
        a = int(span.get("start", 0)); b = int(span.get("end", a))
        if b < a: a, b = b, a
        b = min(b, self.n-1)
        if a <= b:
            import numpy as np
            self.seq[a:b+1] = 1.0
            if callable(getattr(self, "on_update", None)):
                self.on_update(self.seq.copy())

    def __init__(self, parent, sequence: np.ndarray, title: Optional[str] = None,
                 visible_count: Optional[int] = 200, color: wx.Colour = wx.Colour(255, 140, 0), **kwargs):
        # Force role and draggable regardless of callers
        kwargs = dict(kwargs)
        kwargs['role'] = 'annotation'
        kwargs['draggable'] = False
        super().__init__(parent, sequence, title or "Annotation",
                         draggable=False, visible_count=visible_count, color=color, role="annotation")
