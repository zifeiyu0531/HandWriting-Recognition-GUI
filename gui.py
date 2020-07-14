# coding:utf-8

import wx
from neural import app as App
from neural import test as Test
from neural import backward as Backward


APP_TITLE = u'深度学习-手写体数字识别'


class mainFrame(wx.Frame):
    '''程序主窗口类，继承自wx.Frame'''

    def __init__(self):
        '''构造函数'''

        wx.Frame.__init__(self, None, -1, APP_TITLE,
                          style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)
        # 默认style是下列项的组合：wx.MINIMIZE_BOX | wx.MAXIMIZE_BOX | wx.RESIZE_BORDER | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX | wx.CLIP_CHILDREN

        self.SetBackgroundColour(wx.Colour(224, 224, 224))
        self.SetSize((900, 600))
        self.Center()

        # 图片显示
        self.staticBmp1 = wx.StaticBitmap(self, -1, size=(200, 300))
        self.staticBmp1.SetBackgroundColour(wx.Colour(0, 0, 0))
        # 结果
        self.out = wx.StaticText(self, -1, 'null')
        self.out.SetFont(wx.Font(28, wx.ROMAN, wx.ITALIC, wx.NORMAL))
        self.out.SetForegroundColour(wx.Colour(254, 67, 101))
        # 选择文件
        self.choose_btn = wx.Button(self, -1, "选择文件", size=(200, -1))
        # 统计信息
        self.state_btn = wx.Button(self, -1, "统计信息", size=(70, -1))
        # 继续训练
        self.train_continue_btn = wx.Button(self, -1, "继续训练", size=(70, -1))
        # 重新训练
        self.train_begin_btn = wx.Button(self, -1, "重新训练", size=(70, -1))
        # 训练状态
        self.train_step_text = wx.StaticText(self, -1, u"就绪", size=(100, -1))

        self.button_sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.button_sizer1.Add(self.staticBmp1, 0, wx.ALL, 10)
        self.button_sizer1.Add(self.choose_btn, 0, wx.ALL, 10)

        self.button_sizer2 = wx.BoxSizer(wx.VERTICAL)
        self.button_sizer2.Add(self.out, 0, wx.ALL, 10)

        self.button_sizer3 = wx.BoxSizer(wx.VERTICAL)
        self.button_sizer3.Add(self.state_btn, 0, wx.ALL, 10)

        self.button_sizer4 = wx.BoxSizer(wx.VERTICAL)
        self.button_sizer4.Add(self.train_continue_btn, 0, wx.ALL, 10)

        self.button_sizer5 = wx.BoxSizer(wx.VERTICAL)
        self.button_sizer5.Add(self.train_begin_btn, 0, wx.ALL, 10)

        self.button_sizer6 = wx.BoxSizer(wx.VERTICAL)
        self.button_sizer6.Add(self.train_step_text, 0, wx.ALL, 10)

        self.topsizer = wx.BoxSizer(wx.HORIZONTAL)
        self.topsizer.Add(self.button_sizer1, 0, wx.ALIGN_CENTER)
        self.topsizer.Add(self.button_sizer2, 0, wx.ALIGN_CENTER)
        self.topsizer.Add(self.button_sizer3, 0, wx.ALIGN_CENTER)
        self.topsizer.Add(self.button_sizer4, 0, wx.ALIGN_CENTER)
        self.topsizer.Add(self.button_sizer5, 0, wx.ALIGN_CENTER)
        self.topsizer.Add(self.button_sizer6, 0, wx.ALIGN_CENTER)

        self.SetSizerAndFit(self.topsizer)

        self.choose_btn.Bind(wx.EVT_BUTTON, self.choose_click)
        self.state_btn.Bind(wx.EVT_BUTTON, self.state_click)
        self.train_continue_btn.Bind(wx.EVT_BUTTON, self.train_continue_click)
        self.train_begin_btn.Bind(wx.EVT_BUTTON, self.train_begin_click)

    # 选择图片
    def choose_click(self, evt):
        # 重置Image对象尺寸的函数
        def resizeBitmap(image, width=200, height=300):
            bmp = image.Scale(width, height).ConvertToBitmap()
            return bmp

        filesFilter = "image (*.png)|*.png|" "image (*.jpg)|*.jpg"
        fileDialog = wx.FileDialog(
            self, message="选择单个文件", wildcard=filesFilter, style=wx.FD_OPEN)
        dialogResult = fileDialog.ShowModal()
        if dialogResult != wx.ID_OK:
            return
        path = fileDialog.GetPath()
        img = wx.Image(path, wx.BITMAP_TYPE_ANY)
        self.staticBmp1.SetBitmap(resizeBitmap(img, 200, 300))
        self.out.SetLabel((str)(App.application(path)))

    # 统计信息
    def state_click(self, evt):
        state = {
            "BATCH_SIZE": 0,
            "STEPS": 0,
            "ACCURACY": 0
        }
        Test.main(state)
        dlg = wx.MessageDialog(None, u"神经网络信息：\n调用mnist数据集\n训练步长："+str(state["BATCH_SIZE"])+"\n已训练轮数："+str(
            state["STEPS"])+"\n准确率："+str(state["ACCURACY"]), u"统计信息", wx.ICON_INFORMATION)
        dlg.ShowModal()

    # 继续训练
    def train_continue_click(self, evt):
        dlg = wx.MessageDialog(None, u"模型训练中...", u"提示", wx.ICON_INFORMATION)
        dlg.ShowModal()
        self.train_step_text.SetLabel("模型训练中...")
        if(Backward.main(False)):
            dlg = wx.MessageDialog(None, u"训练完成！", u"提示", wx.ICON_INFORMATION)
            dlg.ShowModal()
            self.train_step_text.SetLabel("就绪")

    # 重新训练
    def train_begin_click(self, evt):
        dlg = wx.MessageDialog(None, u"模型训练中...", u"提示", wx.ICON_INFORMATION)
        dlg.ShowModal()
        self.train_step_text.SetLabel("模型训练中...")
        if(Backward.main(True)):
            dlg = wx.MessageDialog(None, u"训练完成！", u"提示", wx.ICON_INFORMATION)
            dlg.ShowModal()
            self.train_step_text.SetLabel("就绪")


class mainApp(wx.App):
    def OnInit(self):
        self.SetAppName(APP_TITLE)
        self.Frame = mainFrame()
        self.Frame.Show()
        return True


if __name__ == "__main__":
    app = mainApp()
    app.MainLoop()
