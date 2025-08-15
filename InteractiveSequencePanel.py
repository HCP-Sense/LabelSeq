"""InteractiveSequencePanel.py: SequencePanel Subclass adding Mouse Events for Interaction"""
__author___ = "Moye Nyuysoni Glein Perry"
__email__ = "moyegp@gmail.com"
__maintainer__ = "Moye Nyuysoni Glein Perry"
__status__ = "Moye Nyuysoni Glein Perry"
# Project Imports
from SequencePanel import SequencePanel
from FormulaParser import safe_eval

# Third-Party Imports
import wx
 

class InteractiveSequencePanel(SequencePanel):
    """SequencePanel Subclass adding Mouse Events for Interaction"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Setup UI controls (title, buttons)
        self._setup_ui()

        # Bind interactive events
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_mouse_leave)
        self.Bind(wx.EVT_LEFT_DCLICK, self.on_double_click)

        # Bind mouse motion to unified handler (handles drag and pan)
        self.Bind(wx.EVT_MOTION, self.on_mouse_motion)

        if self.draggable:
            self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
            self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
            self.Bind(wx.EVT_MIDDLE_DOWN, self.on_middle_down)
            self.Bind(wx.EVT_MIDDLE_UP, self.on_middle_up)
            self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)

    def _setup_ui(self):
        """Creates the title and buttons UI at the top"""
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.title_label = wx.StaticText(self, label=self.title, style=wx.ALIGN_CENTER)
        title_font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.title_label.SetFont(title_font)
        btn_sizer.Add(self.title_label, 1, wx.ALIGN_CENTER_VERTICAL)

        btn_title = wx.Button(self, label="Edit Title", size=(90, -1))
        btn_title.Bind(wx.EVT_BUTTON, self.edit_title)
        btn_sizer.Add(btn_title, 0, wx.LEFT, 5)

        if self.formula:
            btn_formula = wx.Button(self, label="Edit Formula", size=(100, -1))
            btn_formula.Bind(wx.EVT_BUTTON, self.edit_formula)
            btn_sizer.Add(btn_formula, 0, wx.LEFT, 5)

        btn_delete = wx.Button(self, label="Delete", size=(70, -1))
        btn_delete.Bind(wx.EVT_BUTTON, self.delete_self)
        btn_sizer.Add(btn_delete, 0, wx.LEFT, 5)

        main_sizer = self.GetSizer()
        if main_sizer is None:
            main_sizer = wx.BoxSizer(wx.VERTICAL)
            self.SetSizer(main_sizer)
        main_sizer.Clear()
        main_sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.Layout()


    def edit_title(self, evt):
        dlg = wx.TextEntryDialog(self, "Enter new title:", "Edit Title", self.title)
        if dlg.ShowModal() == wx.ID_OK:
            self.title = dlg.GetValue()
            self.title_label.SetLabel(self.title)
            self.Refresh()
        dlg.Destroy()


    def edit_formula(self, evt):
        dlg = wx.TextEntryDialog(self, "Enter new formula:", "Edit Formula", self.formula or "")
        if dlg.ShowModal() == wx.ID_OK:
            self.formula = dlg.GetValue()
            self.Refresh()
        dlg.Destroy()


    def delete_self(self, evt):
        top = wx.GetTopLevelParent(self)
        if hasattr(top, 'remove_panel'):
            top.remove_panel(self)


    def on_mouse_motion(self, evt):
        # Middle button panning
        if evt.MiddleIsDown() and self._pan_start is not None:
            dx = evt.GetX() - self._pan_start
            new_offset = self._pan_origin + dx

            w, _ = self.GetClientSize()
            gw = w - 2 * self.padding
            min_offset = gw * (1 - self.zoom_factor)

            self.pan_offset = min(max(new_offset, min_offset), 0)

            # Sync pan offset with linked panels
            for sp in self.sync_panels:
                sp.pan_offset = self.pan_offset
                sp.Refresh()

            self.Refresh()

        # Left button dragging
        elif self.draggable and self.dragging and evt.LeftIsDown():
            self._handle_drag(evt)
        else:
            # Hover move
            self._handle_hover(evt)


    def _handle_drag(self, evt):
        x, y = evt.GetPosition()
        w, h = self.GetClientSize()
        mn, mx = self.get_y_range()
        rng = mx - mn or 1

        val = ((h - self.padding - y) / (h - 2 * self.padding)) * rng + mn
        val = max(min(val, mx), mn)

        self.seq[self.selected_idx] = val

        # Propagate changes to synchronized panels with formula
        for sp in self.sync_panels:
            if sp.original_seq is not None and sp.formula:
                try:
                    sp.seq[self.selected_idx] = safe_eval(
                        sp.formula,
                        {'orig': sp.original_seq[self.selected_idx], 'res': val}
                    )
                except Exception:
                    pass
                sp.Refresh()

        self.Refresh()


    def _handle_hover(self, evt):
        x, _ = evt.GetPosition()
        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        step = gw / (self.n - 1) * self.zoom_factor
        idx = int((x - self.padding - self.pan_offset) / step + 0.5)
        self.hover_idx = max(0, min(idx, self.n - 1))
        self.Refresh()
        evt.Skip()


    def on_mouse_leave(self, evt):
        self.hover_idx = None
        self.Refresh()
        evt.Skip()


    def on_double_click(self, evt):
        dlg = wx.TextEntryDialog(self, "Enter new title:", "Rename", self.title)
        if dlg.ShowModal() == wx.ID_OK:
            self.title = dlg.GetValue()
            self.title_label.SetLabel(self.title)
            self.Refresh()
        dlg.Destroy()

        if self.formula:
            dlg2 = wx.TextEntryDialog(self, "Enter new formula:", "Edit Formula", self.formula)
            if dlg2.ShowModal() == wx.ID_OK:
                self.formula = dlg2.GetValue()
                self.Refresh()
            dlg2.Destroy()


    def on_left_down(self, evt):
        if not self.draggable:
            return
        x0, y0 = evt.GetPosition()
        w, h = self.GetClientSize()
        best_dist = float('inf')
        best_idx = None
        for i, v in enumerate(self.seq):
            cx, cy = self.to_px(i, v, w, h)
            dist = (cx - x0)**2 + (cy - y0)**2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None and best_dist < (self.radius * 3)**2:
            self.selected_idx = best_idx
            self.dragging = True
            self.CaptureMouse()


    def on_left_up(self, evt):
        if self.dragging:
            self.dragging = False
            self.ReleaseMouse()
            self.Refresh()


    def on_middle_down(self, evt):
        self._pan_start = evt.GetX()
        self._pan_origin = self.pan_offset


    def on_middle_up(self, evt):
        self._pan_start = None


    def on_mouse_wheel(self, evt):
        if not self.draggable:
            return
        rotation = evt.GetWheelRotation() / evt.GetWheelDelta()
        factor = 1 + rotation * 0.1
        old_zoom = self.zoom_factor
        self.zoom_factor = max(1.0, min(old_zoom * factor, 10))

        w, _ = self.GetClientSize()
        gw = w - 2 * self.padding
        mouse_x = evt.GetX() - self.padding - self.pan_offset

        # Adjust pan to zoom on mouse position
        self.pan_offset -= mouse_x * (self.zoom_factor / old_zoom - 1)

        # Clamp pan offset
        min_offset = gw * (1 - self.zoom_factor)
        self.pan_offset = min(max(self.pan_offset, min_offset), 0)

        # Sync zoom/pan with linked panels
        for sp in self.sync_panels:
            sp.zoom_factor = self.zoom_factor
            sp.pan_offset = self.pan_offset
            sp.Refresh()
        self.Refresh()
