# Sequence Annotation Tool (LabelSeq)

A fast, intuitive, and machine-learning-ready **sequence annotation software** built with **wxPython**.  
Designed for time-series labeling, sensor data annotation, active learning, and interactive ML workflows.

This tool supports:
- Multi-track synchronized visualization
- Annotation using points or regions
- Dynamic result computation
- Active Learning suggestions
- High-performance rendering with decimation
- Project save/load
- Undo/Redo for every operation
- Sync pan/zoom/hover across panels

---

## Features Overview

###  **Panels**
The interface uses four synchronized panels:

1. **Original Panel**  
   - Visualizes raw input signal  
   - Defines the global Y-range for all panels  
   - Read-only  

2. **Annotator Panel**  
   - Editable time-series  
   - Supports **Point Mode** → drag points  
   - Supports **Region Mode** → create spans (highlight regions)  
   - Undo/Redo for every change  
   - Changes propagate automatically to Annotation & Result panels  

3. **Annotation Panel**  
   - Stores spans (regions) and/or annotation values  
   - Each annotation panel has **its own color**  
   - Spans are highlighted in that color  
   - Fully synchronized with annotator changes  
   - Freeze mode prevents editing but allows pan/zoom  

4. **Result Panel**  
   - Shows computed results using:  
     ```
     annotation = original - annotator
     result     = original + annotation
     ```
   - Multi-colored highlight regions based on annotations  
   - Displays AL suggestion markers  

---

## **Annotation Modes**

###  Point Mode
- Drag individual points in the annotator panel  
- Hover and selection markers only (no clutter dots)  
- Undo/Redo supported  
- Changes propagate:  
  - Annotation Panel (difference)  
  - Result Panel (final output)  

###  Region Mode
- Click-and-drag to create spans  
- Each annotation panel has its **own highlight color**  
- Delete spans using:
  - Delete key  
  - Right-click → Delete Span  
- Undo/Redo for spans  
- Freeze mode blocks:  
  - Region creation  
  - Span editing  
  - Point editing  
  but still allows pan/zoom  

---

##  **Active Learning (AL)**

Active Learning modes:
- **Point** — learning from labeled points  
- **Span** — learning from annotated regions  
- **Hybrid** — both modes combined  

### AL Pipeline:
1. AL engine extracts multi-feature vectors per time index  
2. KNN + (optional) Random Forest used for probabilistic scoring  
3. Suggestions returned where prediction confidence ≥ threshold  
4. Suggestions shown as purple markers on panels  
5. User can confirm suggestions:
   - As points  
   - Or grouped spans  
6. Suggestions become new training labels for AL  

AL threshold, window size, and mode are adjustable via UI.

---

##  **Interactions**

### Zoom
- Mouse wheel → zoom horizontally  
- Shift + wheel → vertical pan  

### Panning
- Space + drag → horizontal pan  
- Right-drag → vertical pan  
- Middle-drag → horizontal pan  

### Editing
- Left-drag in point mode → move points  
- Left-drag in region mode → create span  
- Click edge/body of span → drag to resize/move  

### Delete
- Select span under cursor → press Delete  
- Or right-click → Delete Span  

---

## **Undo/Redo**

Undo/Redo includes:
- Point movements  
- Region creation  
- Region deletion  
- Span resizing  
- Span shifting  
- AL suggestions that modify annotation  
Everything is stored in separate undo stacks for:
- Annotator panel  
- Annotation panel  

Shortcuts:
- **Ctrl + Z** → Undo  
- **Ctrl + Y** → Redo  

---

##  **Synchronization**

All panels stay in perfect sync:
- 💡 **X-axis** (pan, zoom, play index)
- 💡 **Y-axis** (based on Original panel)
- 💡 Hover index  
- 💡 Selected point  
- 💡 Active Learning suggestions  

Changing the original panel’s value range propagates to all other panels automatically.

---

## **Saving and Loading Projects**

Each project saves:
- Original signal  
- Annotator data  
- Annotation panels + spans  
- Colors & roles  
- Formula values  
- Sync settings  
- Active Learning labels  
- View (zoom, pans, offsets)  

This ensures full session restoration.

---

## **Machine-Learning Ready Exports**

Project exports include:
- Clean CSV / NumPy arrays  
- Annotation spans  
- Per-sample annotation vectors  
- Synchronized frames  

Good for:
- Deep learning preprocessing  
- Supervised training  
- Anomaly detection pipelines  
- Sensor fusion research  

---

##  **Performance & Rendering**

The renderer uses:
- Decimation (MAX_VISIBLE_POINTS=1000)  
- Auto–refresh throttling  
- Optimized wx.GraphicsContext pipelines  
- No heavy loops inside event handlers  

Able to handle:
- Long sensor sequences (50k–200k samples)  
- Multi-panel live updates  
- Real-time dragging without lag  

---

##  **Main Components**

### `InteractiveSequencePanel.py`
Large, feature-rich interactive panel for:
- Rendering  
- Input handling  
- AL integration  
- Undo/Redo  
- Sync propagation  
- Span editing  

### `SequencePanel.py`
Fast base class with:
- Decimation  
- Coordinate transforms  
- View range calculations  

### `active_learning.py`
ML engine with:
- Feature extraction  
- KNN + RF models  
- Confidence scoring  

### `main.py`
Full application layout & orchestration:
- Panel creation  
- Sync linking  
- Legend management  
- AL table integration  
- Save/load  

---

## 📦 Installation

```bash
pip install wxPython numpy scikit-learn
