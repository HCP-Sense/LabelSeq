
# Third-Party Import
import numpy as np
import wx
from wx.lib.floatcanvas import FloatCanvas


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


    # add a circle
    for step, value in enumerate(seq):

        cir = canvas.AddCircle((step*10, value*10), 5)
        canvas.AddObject(cir)



seq = np.random.rand(20)
print(seq)

draw_sequence(seq)

# add a rectangle
rect = FloatCanvas.Rectangle((110, 10), (100, 100), FillColor='Red')
canvas.AddObject(rect)

canvas.Draw()

app.MainLoop()
