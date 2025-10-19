
import json
from pathlib import Path

def save_project(path, model, panels):
    data = {
        "annotation_name": model.annotation_name,
        "annotation_colors": model.annotation_colors,
        "spans": {},  # per-panel spans where applicable
    }
    for k, p in panels.items():
        # collect spans as list of dicts
        spans = []
        for s in getattr(p, "spans", []):
            spans.append({
                "start": int(s.get("start", 0)),
                "end": int(s.get("end", 0)),
                "label": s.get("label"),
                "value": s.get("value"),
            })
        data["spans"][k] = spans
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(path)

def load_project(path, model, panels):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # colors and name
    name = data.get("annotation_name") or model.annotation_name
    if name != model.annotation_name:
        model.rename_annotation(name)
    colors = data.get("annotation_colors") or {}
    for k, v in colors.items():
        model.set_annotation_color(k, tuple(v))
    # spans per panel
    spans_map = data.get("spans", {})
    for key, spans in spans_map.items():
        p = panels.get(key)
        if not p:
            continue
        p.spans = []
        for s in spans:
            p.spans.append({
                "start": int(s.get("start", 0)),
                "end": int(s.get("end", 0)),
                "label": s.get("label"),
                "value": s.get("value"),
            })
        p.Refresh()
    return True
