# GUIdemo2.py
# Demo2 of GUI by PqYt5
# Copyright 2021 Youcans, XUPT
# Crated：2021-10-06

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
import sys
import Ui_test

class MyMainWindow(QMainWindow, Ui_test.Ui_MainWindow):  # 继承 QMainWindow类和 Ui_MainWindow界面类
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # 初始化父类
        self.setupUi(self)  # 继承 Ui_MainWindow 界面


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序