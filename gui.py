# coding:utf-8

import wx

APP_TITLE = u'深度学习-手写体数字识别'


class mainFrame(wx.Frame):
    '''程序主窗口类，继承自wx.Frame'''

    def __init__(self):
        '''构造函数'''

        wx.Frame.__init__(self, None, -1, APP_TITLE,
                          style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        # 默认style是下列项的组合：wx.MINIMIZE_BOX | wx.MAXIMIZE_BOX | wx.RESIZE_BORDER | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX | wx.CLIP_CHILDREN

        self.SetBackgroundColour(wx.Colour(224, 224, 224))
        self.SetSize((1000, 600))
        self.Center()

        # 以下可以添加各类控件

    def btn_click(self, evt):
        return


class mainApp(wx.App):
    def OnInit(self):
        self.SetAppName(APP_TITLE)
        self.Frame = mainFrame()
        self.Frame.Show()
        return True


if __name__ == "__main__":
    app = mainApp()
    app.MainLoop()
