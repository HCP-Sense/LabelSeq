"""
original.py: Role-specific panel for the Original signal.
Thin wrapper around InteractiveSequencePanel that fixes role and defaults.
"""
from InteractiveSequencePanel import InteractiveSequencePanel
import wx
import numpy as np
from typing import Optional

from typing import Optional, Callable, Any, Dict

class OriginalPanel(InteractiveSequencePanel):
    def __init__(self, parent, sequence: np.ndarray, title: Optional[str] = None,
                 visible_count: Optional[int] = 200, color: wx.Colour = wx.BLUE, **kwargs):
        # Force role and draggable regardless of callers
        kwargs = dict(kwargs)
        kwargs['role'] = 'original'
        kwargs['draggable'] = False
        super().__init__(parent, sequence, title or "Original",
                         draggable=False, visible_count=visible_count, color=color, role="original")
        # Keep a direct copy for convenience; InteractiveSequencePanel already uses self.seq
        self.original_seq = sequence
