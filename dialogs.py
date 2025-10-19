# dialogs.py
import wx
import pandas as pd

class NonNumericColumnDialog(wx.Dialog):
    """
    Ask user how to interpret a non-numeric column:
      - Convert to numeric timestamp (dates)
      - Map unique values to integers (categorical)
      - Skip and choose another column
    Also lets the user choose categorical mapping order.
    """
    def __init__(self, parent, col_name: str, sample_values=None):
        super().__init__(parent, title="Non-numeric column detected", size=(520, 360))
        pnl = wx.Panel(self)
        v = wx.BoxSizer(wx.VERTICAL)

        header = wx.StaticText(pnl, label=f"Column '{col_name}' contains non-numeric values.")
        font = header.GetFont(); font.MakeBold(); header.SetFont(font)
        v.Add(header, 0, wx.ALL, 10)

        if sample_values is not None:
            st = wx.StaticText(pnl, label="Sample values:")
            v.Add(st, 0, wx.LEFT | wx.RIGHT, 10)
            txt = wx.TextCtrl(pnl, value="\n".join(map(str, sample_values[:10])), style=wx.TE_MULTILINE | wx.TE_READONLY)
            txt.SetMinSize((480, 80))
            v.Add(txt, 0, wx.ALL | wx.EXPAND, 10)

        self.rb_ts = wx.RadioButton(pnl, label="Convert to numeric timestamps (treat as dates)", style=wx.RB_GROUP)
        self.rb_cat = wx.RadioButton(pnl, label="Map unique strings to integer categories")
        self.rb_skip = wx.RadioButton(pnl, label="Skip this column (I'll choose another)")

        v.Add(self.rb_ts, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)
        v.Add(self.rb_cat, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)
        v.Add(self.rb_skip, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        # Categorical mapping options
        self.cat_box = wx.StaticBoxSizer(wx.StaticBox(pnl, label="Categorical mapping options"), wx.VERTICAL)
        self.rb_cat_appearance = wx.RadioButton(pnl, label="Order of appearance (recommended)", style=wx.RB_GROUP)
        self.rb_cat_alpha = wx.RadioButton(pnl, label="Alphabetical order")
        self.rb_cat_appearance.SetValue(True)
        self.cat_box.Add(self.rb_cat_appearance, 0, wx.ALL, 4)
        self.cat_box.Add(self.rb_cat_alpha, 0, wx.ALL, 4)
        self.cat_box.ShowItems(False)  # hidden until categorical chosen
        v.Add(self.cat_box, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        def on_rb_cat(evt):
            self.cat_box.ShowItems(self.rb_cat.GetValue())
            pnl.Layout()
        self.rb_cat.Bind(wx.EVT_RADIOBUTTON, on_rb_cat)
        self.rb_ts.Bind(wx.EVT_RADIOBUTTON, on_rb_cat)
        self.rb_skip.Bind(wx.EVT_RADIOBUTTON, on_rb_cat)

        btns = self.CreateSeparatedButtonSizer(wx.OK | wx.CANCEL)
        v.Add(btns, 0, wx.ALL | wx.EXPAND, 10)

        pnl.SetSizer(v)
        self.Fit()
        self.CentreOnParent()

    def get_choice(self):
        """
        Returns:
          ("timestamp", None) or ("categorical", "appearance"/"alphabetical") or ("skip", None)
        """
        if self.rb_ts.GetValue():
            return "timestamp", None
        if self.rb_cat.GetValue():
            order = "appearance" if self.rb_cat_appearance.GetValue() else "alphabetical"
            return "categorical", order
        return "skip", None


def interpret_non_numeric_series(parent, name: str, s: pd.Series):
    """
    Show dialog and return a numeric numpy array or None if skipped/cancelled.
    """
    # Build a small sample to display
    sample_vals = list(s.dropna().astype(str).unique()[:20])
    dlg = NonNumericColumnDialog(parent, name, sample_vals)
    if dlg.ShowModal() != wx.ID_OK:
        dlg.Destroy()
        return None, None  # user cancelled
    choice, order = dlg.get_choice()
    dlg.Destroy()

    if choice == "skip":
        return None, "skip"

    if choice == "timestamp":
        # Try to parse to datetime -> numeric seconds
        try:
            dt = pd.to_datetime(s, errors="coerce", dayfirst=False, infer_datetime_format=True)
            num = (dt.view("int64") // 10**9).astype("float64")  # seconds since epoch
            num = num.fillna(method="ffill").fillna(method="bfill")  # smooth out NaNs if any
            return num.to_numpy(), "timestamp"
        except Exception:
            wx.MessageBox("Failed to parse to datetime. Try categorical mapping instead.", "Error",
                          wx.OK | wx.ICON_ERROR)
            return None, None

    if choice == "categorical":
        # Map strings to integers
        vals = s.fillna("").astype(str)
        if order == "alphabetical":
            uniq = sorted(vals.unique())
        else:
            # order of appearance
            seen = []
            for v in vals:
                if v not in seen:
                    seen.append(v)
            uniq = seen
        mapping = {v: i for i, v in enumerate(uniq)}
        mapped = vals.map(mapping).astype("float64")
        return mapped.to_numpy(), ("categorical", mapping, order)

    return None, None
