import wx
import numpy as np

class SequencePanel(wx.Panel):
    def __init__(self, parent, seq, padding=80, draggable=False,
                 title="", xlabel="Sample", ylabel="Amplitude", visible_count=None):
        super().__init__(parent, style=wx.NO_FULL_REPAINT_ON_RESIZE)
        self.seq = list(seq)
        self.n = len(self.seq)
        self.padding = padding
        self.circle_radius = 4
        self.draggable = draggable
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.pan_offset = 0
        self._pan_start_x = None
        self._pan_origin = 0
        if visible_count and visible_count < self.n:
            self.zoom_factor = (self.n - 1) / visible_count
        else:
            self.zoom_factor = 1.0

        self.sync_panels = []
        self.spike_panels = []
        self.original_seq = None

        self.dragging = False
        self.selected_idx = None

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        if self.draggable:
            self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
            self.Bind(wx.EVT_MOTION,    self.on_mouse_move)
            self.Bind(wx.EVT_LEFT_UP,   self.on_left_up)
            self.Bind(wx.EVT_MIDDLE_DOWN, self.on_middle_down)
            self.Bind(wx.EVT_MIDDLE_UP,   self.on_middle_up)
            self.Bind(wx.EVT_MOUSEWHEEL,  self.on_mouse_wheel)

        self._bg_bmp = None

    def set_seq(self, new_seq):
        self.seq = list(new_seq)
        self.redraw()

    def redraw(self):
        self._create_background_bitmap()
        self.Refresh()

    def _create_background_bitmap(self):
        w, h = self.GetClientSize()
        if w <= 0 or h <= 0 or self.n < 2:
            return
        bmp = wx.Bitmap(w, h)
        mdc = wx.MemoryDC(bmp)
        mdc.SetBackground(wx.Brush(wx.Colour(255,255,255)))
        mdc.Clear()

        graph_w = w - 2*self.padding
        graph_h = h - 2*self.padding
        scale_x = graph_w / (self.n - 1) * self.zoom_factor
        max_val, min_val = max(self.seq), min(self.seq)
        range_y = max_val - min_val or 1
        max_val += 0.1 * range_y
        min_val -= 0.1 * range_y
        range_y = max_val - min_val
        scale_y = graph_h / range_y

        def map_point(i, v):
            x = int(self.padding + i*scale_x + self.pan_offset)
            y = int(h - self.padding - (v - min_val)*scale_y)
            return x, y

        mdc.SetPen(wx.Pen(wx.LIGHT_GREY))
        mdc.DrawRectangle(self.padding, self.padding, graph_w, graph_h)

        mdc.SetPen(wx.Pen(wx.BLACK,1))
        mdc.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        max_tick_w = 0
        for i in range(6):
            val = min_val + (range_y / 5) * i
            y = int(h - self.padding - (val - min_val)*scale_y)
            mdc.DrawLine(self.padding-5, y, self.padding, y)
            txt = f"{val:.1f}"
            tw, th = mdc.GetTextExtent(txt)
            max_tick_w = max(max_tick_w, tw)
            mdc.DrawText(txt, self.padding - 30 - tw, y - th//2)

        step = max(1, self.n // 10)
        for i in range(0, self.n, step):
            x, _ = map_point(i, min_val)
            if self.padding <= x <= w - self.padding - 10:
                mdc.DrawLine(x, h-self.padding, x, h-self.padding+5)
                mdc.DrawText(str(i), x-10, h-self.padding+8)

        lw, lh = mdc.GetTextExtent(self.xlabel)
        mdc.DrawText(self.xlabel, (w-lw)//2, h-25)
        yw, yh = mdc.GetTextExtent(self.ylabel)
        ylabel_x = max(5, self.padding - max_tick_w - 50)
        mdc.DrawRotatedText(self.ylabel, ylabel_x, (h + yw)//2, 90)

        mdc.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        tw, th = mdc.GetTextExtent(self.title)
        mdc.DrawText(self.title, (w-tw)//2, self.padding//4)

        mdc.SetClippingRegion(self.padding, self.padding, graph_w, graph_h)
        pts = [map_point(i, v) for i, v in enumerate(self.seq)]
        mdc.SetPen(wx.Pen(wx.BLUE,1))
        for (x0,y0),(x1,y1) in zip(pts, pts[1:]):
            mdc.DrawLine(x0,y0,x1,y1)
        if self.draggable:
            mdc.SetBrush(wx.Brush(wx.BLUE))
            for x,y in pts:
                mdc.DrawCircle(x,y,self.circle_radius)
        mdc.DestroyClippingRegion()

        mdc.SelectObject(wx.NullBitmap)
        self._bg_bmp = bmp

    def on_size(self, evt):
        self.redraw(); evt.Skip()
    def on_paint(self, evt):
        dc = wx.BufferedPaintDC(self)
        if self._bg_bmp: dc.DrawBitmap(self._bg_bmp,0,0)

    def on_left_down(self, evt):
        x0,y0 = evt.GetPosition(); w,h = self.GetClientSize()
        scale_x = (w-2*self.padding)/(self.n-1)*self.zoom_factor
        max_val, min_val = max(self.seq), min(self.seq)
        scale_y = (h-2*self.padding)/(max_val-min_val or 1)
        for i,v in enumerate(self.seq):
            cx, cy = self.padding+i*scale_x+self.pan_offset, h-self.padding-(v-min_val)*scale_y
            if (x0-cx)**2+(y0-cy)**2<=self.circle_radius**2:
                self.selected_idx=i; self.dragging=True; self.CaptureMouse(); return

    def on_mouse_move(self, evt):
        if self.draggable and self.dragging and evt.LeftIsDown():
            _,y = evt.GetPosition(); h=self.GetClientSize().height
            graph_h = h-2*self.padding
            max_val, min_val = max(self.seq), min(self.seq)
            val = ((h-self.padding-y)/graph_h)*(max_val-min_val or 1)+min_val
            self.seq[self.selected_idx]=max(min_val, min(max_val, val))
            if self.original_seq is not None:
                spikes = np.array(self.original_seq)-np.array(self.seq)
                for pnl in self.spike_panels:
                    pnl.set_seq(spikes.tolist())
            self.redraw()
        elif self.draggable and self._pan_start_x is not None and evt.MiddleIsDown():
            dx = evt.GetX()-self._pan_start_x; self.pan_offset=self._pan_origin+dx
            for pnl in self.sync_panels:
                pnl.pan_offset=self.pan_offset; pnl.zoom_factor=self.zoom_factor; pnl.redraw()
            self.redraw()

    def on_left_up(self, evt):
        if self.dragging:
            self.dragging=False; self.ReleaseMouse(); self.redraw()

    def on_middle_down(self, evt): self._pan_start_x=evt.GetX(); self._pan_origin=self.pan_offset
    def on_middle_up(self, evt): self._pan_start_x=None

    def on_mouse_wheel(self, evt):
        if not self.draggable: return
        rot = evt.GetWheelRotation()/evt.GetWheelDelta(); factor = 1+rot*0.1
        self.zoom_factor = max(0.1, min(self.zoom_factor*factor, 10))
        mx = evt.GetX()-self.padding-self.pan_offset; self.pan_offset -= mx*(factor-1)
        for pnl in self.sync_panels:
            pnl.zoom_factor=self.zoom_factor; pnl.pan_offset=self.pan_offset; pnl.redraw()
        self.redraw()

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="LabelSeq", size=(1000,800))
        panel = wx.Panel(self)
        panel.SetBackgroundColour(wx.Colour(255,255,255))
        lbl=wx.StaticText(panel,label="LabelSeq"); lbl.SetFont(wx.Font(14,wx.FONTFAMILY_SWISS,wx.FONTSTYLE_NORMAL,wx.FONTWEIGHT_BOLD))
        top=wx.BoxSizer(wx.HORIZONTAL); top.Add(lbl,0,wx.ALL|wx.ALIGN_CENTER_VERTICAL,5)
        data_len, vis_count = 300, 200
        original = np.random.rand(data_len)
        result   = original.copy()
        spikes   = original - result
        p_orig = SequencePanel(panel, original, title="Original Signal", visible_count=vis_count)
        p_spk  = SequencePanel(panel, spikes,    title="Signal Componet 1",   visible_count=vis_count)
        p_res  = SequencePanel(panel, result,    title="Result", draggable=True, visible_count=vis_count)
        p_res.original_seq = original
        p_res.spike_panels  = [p_spk]
        p_res.sync_panels   = [p_spk, p_orig]
        for pnl in p_res.sync_panels:
            pnl.zoom_factor = p_res.zoom_factor; pnl.pan_offset = p_res.pan_offset; pnl.redraw()
        layout=wx.BoxSizer(wx.VERTICAL)
        for p in (p_orig, p_spk, p_res):
            layout.Add(p, 1, wx.EXPAND|wx.ALL, 5)
        main = wx.BoxSizer(wx.VERTICAL)
        main.Add(top, 0, wx.EXPAND|wx.LEFT|wx.TOP|wx.RIGHT, 5)
        main.Add(layout, 1, wx.EXPAND|wx.ALL, 20)
        panel.SetSizer(main)
        self.Centre()
        self.Show()

if __name__=='__main__':
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()
