"""
annotator.py: Role-specific panel for the Annotator (draggable edits).
Thin wrapper around InteractiveSequencePanel that fixes role and defaults.
"""
from InteractiveSequencePanel import InteractiveSequencePanel
import wx
import numpy as np
from typing import Optional, Callable

from typing import Optional, Callable, Any, Dict

class AnnotatorPanel(InteractiveSequencePanel):
    def __init__(self, parent, sequence: np.ndarray, title: Optional[str] = None,
                 visible_count: Optional[int] = 200, color: wx.Colour = wx.GREEN,
                 on_change: Optional[Callable[[np.ndarray], None]] = None, **kwargs):
        # Force role and draggable regardless of callers
        kwargs = dict(kwargs)
        kwargs['role'] = 'annotator'
        kwargs['draggable'] = True
        super().__init__(parent, sequence, title or "Annotator",
                         draggable=True, visible_count=visible_count, color=color, role="annotator")
        self._on_change = on_change

    # If/when InteractiveSequencePanel calls self.on_drag_end() or similar, we can hook here.
    # For now, expose a helper to be called by external code if needed.
    def notify_changed(self):
        if callable(self._on_change):
            try:
                self._on_change(self.seq)
            except Exception:
                pass
