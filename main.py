
# Third-Party Import
import numpy as np
import wx
from wx.lib.floatcanvas import FloatCanvas
from wx.lib.floatcanvas.FCObjects import Line


app = wx.App() 
window = wx.Frame(None, title = "LabelSeq", size = (600,600))
panel = wx.Panel(window)
label = wx.StaticText(panel, label = "Hello World", pos = (100,50)) 
window.Show(True) 




canvas = FloatCanvas.FloatCanvas(window, -1,
                             size=(500, 500),
                             ProjectionFun=None,
                             Debug=0,
                             BackgroundColor="White",
                             )



def draw_sequence(seq: np.ndarray, scale_x: float = 10, scale_y: float = 10):

    for step, value in enumerate(seq):
        # add a circle
        cir = canvas.AddCircle((step*scale_x, value*scale_y), 5, LineWidth=0, FillColor="Blue")
        canvas.AddObject(cir)

    for step, (value_a, value_b) in enumerate(zip(seq[:-1], seq[1:])):
        print(f"{step} - {value_a} - {value_b}")
        line = Line(np.array([[step*scale_x, value_a*scale_y], [(step+1)*scale_x, value_b*scale_y]]))
        canvas.AddObject(line)


seq = np.random.rand(20)
print(seq)

draw_sequence(seq)

# add a rectangle
#rect = FloatCanvas.Rectangle((110, 10), (100, 100), FillColor='Red')
#canvas.AddObject(rect)

canvas.Draw()

app.MainLoop()
