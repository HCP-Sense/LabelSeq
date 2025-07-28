import wx
import numpy as np
import ast, operator
from wx.lib import floatcanvas  # ensure this import so WX knows about GC
from SequencePanelCollection import SequencePanel

# Safe eval (unchanged)
def safe_eval(expr, names):
    try:
        node = ast.parse(expr, mode='eval')
    except Exception:
        raise ValueError("Invalid formula syntax")
    OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg
    }
    def _visit(n):
        if isinstance(n, ast.Expression):
            return _visit(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp):
            l = _visit(n.left)
            r = _visit(n.right)
            if type(n.op) in OPS:
                return OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp):
            v = _visit(n.operand)
            if type(n.op) in OPS:
                return OPS[type(n.op)](v)
        if isinstance(n, ast.Name) and n.id in names:
            return names[n.id]
        raise ValueError(f"Unsupported expression: {expr}")
    return _visit(node)



class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="LabelSeq (GC‑only)", size=(1000, 800))
        pnl = wx.Panel(self)
        pnl.SetBackgroundColour(wx.WHITE)

        btn = wx.Button(pnl, label="Add Component")
        btn.Bind(wx.EVT_BUTTON, self.on_add)
        vs = wx.BoxSizer(wx.VERTICAL)
        vs.Add(btn, 0, wx.ALL | wx.CENTER, 5)

        self.sig_area = wx.ScrolledWindow(pnl, style=wx.VSCROLL)
        self.sig_area.SetScrollRate(0, 20)
        self.sig_sizer = wx.BoxSizer(wx.VERTICAL)
        self.sig_area.SetSizer(self.sig_sizer)
        vs.Add(self.sig_area, 1, wx.EXPAND | wx.ALL, 5)

        pnl.SetSizer(vs)
        pnl.Layout()
        self.Layout()
        self.Show()
        self.Centre()

        self.original = np.random.rand(300)
        self.result = self.original.copy()
        self.signals = []
        self.result_panel = None

        # default panels
        self.add_panel("Original", self.original, draggable=False)
        self.add_panel("Signal 1", self.original - self.result,
                       formula="orig-res", draggable=False)
        self.add_panel("Result", self.result, draggable=True)

    def add_panel(self, label, seq, formula=None, draggable=False):
        sp = SequencePanel(self.sig_area, seq, label, formula,
                           draggable=draggable, visible_count=200)
        sp.SetMinSize((-1, 250))

        if formula:
            sp.original_seq = self.original

        if draggable:
            self.result_panel = sp
            sp.sync_panels = [entry['panel'] for entry in self.signals if entry['formula']]
            self.sig_sizer.Add(sp, 0, wx.EXPAND | wx.ALL, 5)
        else:
            if self.result_panel:
                idx = next(
                    i for i in range(self.sig_sizer.GetItemCount())
                    if self.sig_sizer.GetItem(i).GetWindow() is self.result_panel
                )
                self.sig_sizer.Insert(idx, sp, 0, wx.EXPAND | wx.ALL, 5)
                if formula:
                    self.result_panel.sync_panels.append(sp)
            else:
                self.sig_sizer.Add(sp, 0, wx.EXPAND | wx.ALL, 5)

        self.signals.append({'panel': sp, 'formula': formula})
        self.sig_area.FitInside()
        self.sig_area.Layout()
        sp.Refresh()

    def on_add(self, evt):
        dlg = wx.TextEntryDialog(self, "Formula (orig,res):", "New", "")
        if dlg.ShowModal() == wx.ID_OK:
            formula = dlg.GetValue()
            dlg.Destroy()
            seq = [0.0] * len(self.original)
            ld = wx.TextEntryDialog(self, "Label:", "New", "")
            if ld.ShowModal() == wx.ID_OK:
                label = ld.GetValue()
                ld.Destroy()
                self.add_panel(label, seq, formula=formula, draggable=False)
            else:
                ld.Destroy()
        else:
            dlg.Destroy()

if __name__ == '__main__':
    app = wx.App(False)
    MainFrame()
    app.MainLoop()
