
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



def draw_sequence(seq: np.ndarray):

    for step, value in enumerate(seq):
        # add a circle
        cir = canvas.AddCircle((step*10, value*10), 5, LineWidth=0, FillColor="Blue")
        canvas.AddObject(cir)

    for step, (value_a, value_b) in enumerate(zip(seq[:-1], seq[1:])):
        print(f"{step} - {value_a} - {value_b}")
        line = Line(np.array([[step*10,value_a*10],[(step+1)*10,value_b*10]]))
        canvas.AddObject(line)


seq = np.random.rand(20)
print(seq)

draw_sequence(seq)

# add a rectangle
#rect = FloatCanvas.Rectangle((110, 10), (100, 100), FillColor='Red')
#canvas.AddObject(rect)

canvas.Draw()

app.MainLoop()
