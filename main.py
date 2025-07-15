
# Third-Party Import
import wx
from wx.lib.floatcanvas import FloatCanvas


app = wx.App() 
window = wx.Frame(None, title = "LabelSeq", size = (300,200)) 
panel = wx.Panel(window)
label = wx.StaticText(panel, label = "Hello World", pos = (100,50)) 
window.Show(True) 




canvas = FloatCanvas.FloatCanvas(window, -1,
                             size=(500, 500),
                             ProjectionFun=None,
                             Debug=0,
                             BackgroundColor="White",
                             )


# add a circle
cir = canvas.AddCircle((10, 10), 100)
canvas.AddObject(cir)

# add a rectangle
rect = FloatCanvas.Rectangle((110, 10), (100, 100), FillColor='Red')
canvas.AddObject(rect)

canvas.Draw()

app.MainLoop()
