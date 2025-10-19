"""
result.py: Role-specific panel for the Result view.
Thin wrapper around InteractiveSequencePanel that fixes role and defaults.
"""
from InteractiveSequencePanel import InteractiveSequencePanel
import wx
import numpy as np
from typing import Optional

from typing import Optional, Callable, Any, Dict

class ResultPanel(InteractiveSequencePanel):
    def __init__(self, parent, sequence: np.ndarray, title: Optional[str] = None,
                 visible_count: Optional[int] = 200, color: wx.Colour = wx.Colour(200, 0, 0), **kwargs):
        # Force role and draggable regardless of callers
        kwargs = dict(kwargs)
        kwargs['role'] = 'result'
        kwargs['draggable'] = False
        super().__init__(parent, sequence, title or "Result",
                         draggable=False, visible_count=visible_count, color=color, role="result")
        # Convenience aliases used by existing code
        self.original_seq = None
        self.annotator_ref = None
