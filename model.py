
import numpy as np
from event_bus import EventBus

class Model:
    """
    Minimal shared model for Milestone A/B:
      - holds original, annotator, annotation (single layer for now), result
      - supports renaming the annotation layer (legend reflects instantly)
    """
    def __init__(self):
        self.bus = EventBus()
        self.original = np.zeros(2, dtype=float)
        self.annotator = np.zeros(2, dtype=float)
        self.annotation_name = "Annotation"
        self.annotation = np.zeros(2, dtype=float)
        self.result = np.zeros(2, dtype=float)
        # Legend colors per layer name
        self.annotation_colors = {self.annotation_name: (255, 140, 0)}  # orange-ish

    # Compute result = original + annotation (default behavior)
    def recompute(self):
        self.result = (self.original + self.annotation).astype(float)
        self.bus.emit("MODEL_UPDATED")

    def set_original(self, arr):
        self.original = np.asarray(arr, dtype=float).copy()
        self.bus.emit("ORIGINAL_CHANGED", self.original)

    def set_annotator(self, arr):
        self.annotator = np.asarray(arr, dtype=float).copy()
        # default annotation definition
        self.annotation = self.original - self.annotator
        self.bus.emit("ANNOTATOR_CHANGED", self.annotator)
        self.recompute()

    def rename_annotation(self, new_name: str):
        old = self.annotation_name
        if not new_name or new_name == old:
            return
        # move color mapping
        color = self.annotation_colors.pop(old, (255, 140, 0))
        self.annotation_name = new_name
        self.annotation_colors[new_name] = color
        self.bus.emit("ANNOTATION_RENAMED", old, new_name)

    def set_annotation_color(self, name, rgb_tuple):
        self.annotation_colors[name] = tuple(rgb_tuple)
        self.bus.emit("ANNOTATION_COLOR_CHANGED", name, rgb_tuple)
