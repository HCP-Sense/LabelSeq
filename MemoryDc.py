import wx
import numpy as np
import ast, operator

def safe_eval(expr, names):
    """Evaluate arithmetic expressions safely with support for +,-,*,/,**, unary -, returning 0 on divide-by-zero."""
    node = ast.parse(expr, mode='eval')
    OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg
    }
    def _visit(n):
        if isinstance(n, ast.Expression):
            return _visit(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp):
            l = _visit(n.left)
            r = _visit(n.right)
            if isinstance(n.op, ast.Div) and abs(r) < 1e-12:
                return 0.0
            return OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp):
            return OPS[type(n.op)](_visit(n.operand))
        if isinstance(n, ast.Name) and n.id in names:
            v = names[n.id]
            try:
                return float(v)
            except:
                return v
        raise ValueError(f"Unsupported: {expr}")
    return _visit(node)

class SequencePanel(wx.Panel):
    def __init__(self, parent, seq, title, formula, draggable, visible_count):
        super().__init__(parent, style=wx.NO_FULL_REPAINT_ON_RESIZE)
        self.parent_frame = wx.GetTopLevelParent(self)
        self.seq = list(seq)
        self.title = title
        self.formula = formula
        self.draggable = draggable
        self.n = len(self.seq)
        self.visible_count = visible_count

        # view params
        self.padding = 60
        self.radius = 6
        self.zoom = max(1.0, (self.n-1)/self.visible_count)
        self.pan = 0.0

        # interaction state
        self.selected = None
        self.dragging = False
        self._pan_start = None
        self._pan_origin = 0.0

        # bind events
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, lambda e: (self.Refresh(), e.Skip()))
        self.Bind(wx.EVT_LEFT_DCLICK, self.on_double_click)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        if draggable:
            self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
            self.Bind(wx.EVT_MOTION, self.on_mouse_move)
            self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
            self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel)

    def to_pixels(self, i, v, w, h):
        gw = w - 2*self.padding
        gh = h - 2*self.padding
        sx = gw/(self.n-1)*self.zoom
        mx, mn = max(self.seq), min(self.seq)
        rng = mx-mn or 1
        mx += 0.1*rng; mn -= 0.1*rng; rng = mx-mn
        sy = gh / rng
        x = self.padding + i*sx + self.pan
        y = h - self.padding - (v-mn)*sy
        return x, y

    def clamp_pan(self, w):
        gw = w - 2*self.padding
        total = (self.n-1)*(gw/(self.n-1))*self.zoom
        min_pan = min(0, gw - total)
        self.pan = max(min(self.pan, 0), min_pan)

    def on_paint(self, evt):
        w, h = self.GetClientSize()
        if w<2 or h<2 or self.n<2:
            return
        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        gc.SetBrush(wx.Brush(wx.WHITE)); gc.DrawRectangle(0,0,w,h)

        font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        gf = gc.CreateFont(font, wx.BLACK); gc.SetFont(gf)

        px, py = self.padding, self.padding
        pw, ph = w-2*self.padding, h-2*self.padding
        gc.PushState(); gc.Clip(px, py, pw, ph)

        pts = [self.to_pixels(i,v,w,h) for i,v in enumerate(self.seq)]
        if pts:
            path = gc.CreatePath(); path.MoveToPoint(*pts[0])
            for p in pts[1:]: path.AddLineToPoint(*p)
            gc.SetPen(wx.Pen(wx.BLUE,1)); gc.StrokePath(path)

        if self.draggable and self.selected is not None:
            x,y = pts[self.selected]
            gc.SetBrush(wx.Brush(wx.BLUE))
            gc.DrawEllipse(x-self.radius, y-self.radius, 2*self.radius, 2*self.radius)

        gc.PopState()
        gc.SetPen(wx.Pen(wx.LIGHT_GREY))
        gc.StrokeLine(px,py,px,py+ph); gc.StrokeLine(px,py+ph,px+pw,py+ph)

        mx, mn = max(self.seq), min(self.seq); rng = mx-mn or 1
        mx += 0.1*rng; mn -= 0.1*rng; rng = mx-mn
        for j in range(6):
            val = mn + rng*j/5
            y = py+ph - (val-mn)*(ph/rng)
            gc.StrokeLine(px-5,y,px,y)
            s = f"{val:.1f}"; tw,th = gc.GetTextExtent(s)
            gc.DrawText(s, px-10-tw, y-th/2)

        step = max(1, self.n//10)
        for i in range(0,self.n,step):
            x,_ = self.to_pixels(i,mn,w,h)
            if px<=x<=px+pw:
                gc.StrokeLine(x, py+ph, x, py+ph+5)
                s = str(i); tw,th = gc.GetTextExtent(s)
                gc.DrawText(s, x-tw/2, py+ph+8)

        title_font = wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        gtf = gc.CreateFont(title_font, wx.BLACK); gc.SetFont(gtf)
        tw,th = gc.GetTextExtent(self.title)
        gc.DrawText(self.title, (w-tw)/2, 5)

    def on_double_click(self, evt):
        x,y = evt.GetPosition(); w,_ = self.GetClientSize()
        dc = wx.ClientDC(self); tw,th = dc.GetTextExtent(self.title)
        tx = (w-tw)//2
        if tx<=x<=tx+tw and 0<=y<=th:
            dlg = wx.TextEntryDialog(self, "New label:", "Rename", self.title)
            if dlg.ShowModal()==wx.ID_OK:
                self.title = dlg.GetValue(); self.Refresh()
            dlg.Destroy()
            if self.formula:
                dlg2 = wx.TextEntryDialog(self, "Edit formula:", "Formula", self.formula)
                if dlg2.ShowModal()==wx.ID_OK:
                    self.formula = dlg2.GetValue()
                    self.parent_frame.recalc_all(); self.parent_frame.refresh_all()
                dlg2.Destroy()

    def on_right_down(self, evt):
        x,y = evt.GetPosition(); w,_ = self.GetClientSize()
        dc = wx.ClientDC(self); tw,th = dc.GetTextExtent(self.title)
        tx = (w-tw)//2
        if tx<=x<=tx+tw and 0<=y<=th:
            menu = wx.Menu(); menu.Append(wx.ID_DELETE, "Delete")
            self.Bind(wx.EVT_MENU, self.on_delete, id=wx.ID_DELETE)
            self.PopupMenu(menu); menu.Destroy()

    def on_delete(self, evt):
        self.parent_frame.remove_panel(self)

    def on_left_down(self, evt):
        pos = evt.GetPosition(); w,h = self.GetClientSize()
        best,bi = float('inf'),None
        for i,v in enumerate(self.seq):
            x,y = self.to_pixels(i,v,w,h)
            d = (x-pos.x)**2 + (y-pos.y)**2
            if d<best and d<(self.radius*3)**2:
                best,bi = d,i
        if bi is not None:
            self.selected = bi; self.dragging = True; self.CaptureMouse()
        else:
            self._pan_start = pos.x; self._pan_origin = self.pan

    def on_mouse_move(self, evt):
        pos = evt.GetPosition(); w,h = self.GetClientSize()
        if self.dragging and evt.LeftIsDown():
            mx,mn = max(self.seq),min(self.seq); rng=mx-mn or 1
            mx+=0.1*rng; mn-=0.1*rng; rng=mx-mn
            val = ((h-self.padding-pos.y)/(h-2*self.padding))*rng + mn
            val = max(min(val,mx),mn); idx=self.selected
            self.seq[idx] = val
            self.parent_frame.recalc_all(idx); self.parent_frame.refresh_all()
        elif self._pan_start is not None:
            dx = pos.x - self._pan_start; self.pan = self._pan_origin+dx
            self.clamp_pan(w); self.parent_frame.sync_pan_zoom(self.pan,self.zoom); self.parent_frame.refresh_all()

    def on_left_up(self, evt):
        if self.dragging: self.dragging=False; self.ReleaseMouse()
        else: self._pan_start=None

    def on_mouse_wheel(self, evt):
        delta = evt.GetWheelRotation(); factor = 1.1 if delta>0 else 1/1.1
        new_zoom = max(1.0, min(self.zoom*factor,5.0))
        w = self.GetClientSize().width
        old_total = (self.n-1)*((w-2*self.padding)/(self.n-1)*self.zoom)
        new_total = (self.n-1)*((w-2*self.padding)/(self.n-1)*new_zoom)
        mx = evt.GetPosition().x - self.padding - self.pan
        rel = (mx+old_total-(w-2*self.padding))/old_total if old_total else 0
        self.pan -= (new_total-old_total)*rel; self.zoom=new_zoom; self.clamp_pan(w)
        self.parent_frame.sync_pan_zoom(self.pan,self.zoom); self.parent_frame.refresh_all()

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Signal Editor (GC‑only)", size=(1000,800))
        self.panels = []
        pnl = wx.Panel(self)
        vs = wx.BoxSizer(wx.VERTICAL)
        btn_add = wx.Button(pnl, label="Add Component")
        btn_add.Bind(wx.EVT_BUTTON, self.on_add)
        vs.Add(btn_add, 0, wx.ALL|wx.CENTER, 5)
        self.sig_area = wx.ScrolledWindow(pnl, style=wx.VSCROLL)
        self.sig_area.SetScrollRate(0,20)
        self.sig_sizer = wx.BoxSizer(wx.VERTICAL)
        self.sig_area.SetSizer(self.sig_sizer)
        vs.Add(self.sig_area, 1, wx.EXPAND|wx.ALL, 5)
        pnl.SetSizer(vs); self.Layout(); self.Show(); self.Centre()
        self.original = np.array([(-1)**i*(i%10) for i in range(300)], dtype=float)
        self.result = self.original.copy()
        self.formulas = []
        self.add_panel("Original", self.original, None, False)
        self.add_panel("Signal 1", self.original-self.result, "orig-res", False)
        self.add_panel("Result", self.result, None, True)
        self.sync_pan_zoom(0, self.panels[-1].zoom); self.refresh_all()

    def add_panel(self, label, seq, formula, draggable):
        sp = SequencePanel(self.sig_area, seq, label, formula, draggable, visible_count=200)
        sp.parent_frame = self; sp.SetMinSize((-1,250)); self.panels.append(sp)
        dr = [i for i,p in enumerate(self.panels) if p.draggable]
        if draggable or not dr:
            self.sig_sizer.Add(sp, 0, wx.EXPAND|wx.ALL,5)
        else:
            self.sig_sizer.Insert(dr[0], sp, 0, wx.EXPAND|wx.ALL,5)
        self.sig_area.FitInside(); self.sig_area.Layout()

    def recalc_all(self, idx=None):
        res = next(p.seq for p in self.panels if p.draggable)
        for p in self.panels:
            if p.formula:
                if idx is None:
                    p.seq = [safe_eval(p.formula, {'orig':o,'res':r}) for o,r in zip(self.original, res)]
                else:
                    p.seq[idx] = safe_eval(p.formula, {'orig':self.original[idx],'res':res[idx]})

    def sync_pan_zoom(self, pan, zoom):
        for p in self.panels: p.pan=pan; p.zoom=zoom

    def refresh_all(self):
        for p in self.panels: p.clamp_pan(p.GetClientSize().width); p.Refresh()

    def remove_panel(self, panel):
        self.sig_sizer.Detach(panel); panel.Destroy()
        if panel.formula in self.formulas: self.formulas.remove(panel.formula)
        self.sig_area.Layout()

    def on_add(self, evt):
        dlg = wx.TextEntryDialog(self, "Formula (orig,res):", "New", "")
        if dlg.ShowModal() == wx.ID_OK:
            formula = dlg.GetValue().strip()
            dlg.Destroy()
            if formula:
                self.formulas.append(formula)
                zero_seq = [0.0] * len(self.original)
                self.add_panel(f"Component {len(self.formulas)}", zero_seq, formula, False)
                self.refresh_all()
        else:
            dlg.Destroy()

if __name__=='__main__':
    app = wx.App(False)
    MainFrame()
    app.MainLoop()
