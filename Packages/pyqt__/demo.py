import sys
import time
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QTextEdit, QPushButton, QFileDialog
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout
from PyQt5.QtGui import QTextOption, QImage, QPixmap
 
class QtDemo(QWidget):
    def __init__(self):
        super(QtDemo, self).__init__()
        
        self.desktop = QApplication.desktop()
 
        #获取显示器分辨率大小
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        #获取不同分辨率下需要缩放的比例，2560为参考尺寸
        self.scale_ratio = self.width / 2560
       
        # self.resize(900, 400)
        #根据显示器分辨率自动设置窗口大小
        self.setGeometry(50, 50, 50 + int(900 * self.scale_ratio), 50 + int(400 * self.scale_ratio))
        #设置窗口名称与标题
        self.setWindowTitle("QtDemo")
       
        self.label1 = QLabel(self)
        self.label1.setText('标题一')
        self.label1.setWordWrap(True)
        self.label1.setStyleSheet("QLabel{qproperty-alignment: AlignCenter;font-size:30px;font-weight:bold;font-family:宋体;}")
        self.label2 = QLabel(self)
        self.label2.setText('标题2')
        self.label2.setWordWrap(True)
        self.label2.setStyleSheet("QLabel{qproperty-alignment: AlignCenter;font-size:30px;font-weight:bold;font-family:宋体;}")
        self.label3 = QLabel(self)
        self.label3.setText('标题3')
        self.label3.setWordWrap(True)
        self.label3.setStyleSheet("QLabel{qproperty-alignment: AlignCenter;font-size:30px;font-weight:bold;font-family:宋体;}")
        
        self.label11 = QLabel(self)
        self.label11.setText("Label11")
        self.label11.resize(320, 320)
        #设置label样式
        self.label11.setStyleSheet("QLabel{background:#FFFFFF;}"
                                 "QLabel{qproperty-alignment: AlignCenter;color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")
        self.label12 = QLabel(self)
        self.label12.setText("Label12")
        self.label12.resize(320, 320)
        self.label12.setStyleSheet("QLabel{background:#FFFFFF;}"
                                 "QLabel{qproperty-alignment: AlignCenter;color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")
        
        self.text1 = QTextEdit(self)
        self.text1.setWordWrapMode(QTextOption.NoWrap)
        self.text1.setText("Text1")
        self.text1.setFixedSize(200, 320)
        self.text1.setStyleSheet("QTextEdit{background:#FFFFFF;}"
                                 "QTextEdit{qproperty-alignment: AlignCenter;color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")
 
        self.btn1 = QPushButton(self)
        self.btn1.setText("按钮一")
        self.btn1.clicked.connect(self.fun1)
        self.btn1.setStyleSheet("QPushButton{background:(#F000F0)}"
                                 "QPushButton{qproperty-alignment: AlignCenter;color:rgb(300,300,300,120);font-size:25px;font-weight:bold;font-family:宋体;}")
        self.btn2 = QPushButton(self)
        self.btn2.setText("按钮二")
        self.btn2.clicked.connect(self.fun2)
        self.btn2.setStyleSheet("QPushButton{background:(#F000F0)}"
                                 "QPushButton{qproperty-alignment: AlignCenter;color:rgb(300,300,300,120);font-size:25px;font-weight:bold;font-family:宋体;}")
        self.btn3 = QPushButton(self)
        self.btn3.setText("按钮三")
        self.btn3.clicked.connect(self.fun3)
        self.btn3.setStyleSheet("QPushButton{background:(#F000F0)}"
                                 "QPushButton{qproperty-alignment: AlignCenter;color:rgb(300,300,300,120);font-size:25px;font-weight:bold;font-family:宋体;}")
        self.btn4 = QPushButton(self)
        self.btn4.setText("按钮四")
        self.btn4.clicked.connect(self.fun4)
        self.btn4.setStyleSheet("QPushButton{background:(#F000F0)}"
                                 "QPushButton{qproperty-alignment: AlignCenter;color:rgb(300,300,300,120);font-size:25px;font-weight:bold;font-family:宋体;}")  
        self.btn5 = QPushButton(self)
        self.btn5.setText("按钮五")
        self.btn5.clicked.connect(self.fun5)
        self.btn5.setStyleSheet("QPushButton{background:(#F000F0)}"
                                 "QPushButton{qproperty-alignment: AlignCenter;color:rgb(300,300,300,120);font-size:25px;font-weight:bold;font-family:宋体;}")
        
        grid1 = QGridLayout()
        grid1.addWidget(self.label1, 1, 1)
        grid1.addWidget(self.label2, 1, 2)
        grid1.addWidget(self.label3, 1, 3)
        grid1.addWidget(self.label11, 2, 1)
        grid1.addWidget(self.label12, 2, 2)
        grid1.addWidget(self.text1, 2, 3)
        
        grid2 = QGridLayout()
        grid2.addWidget(self.btn1, 1, 1)
        grid2.addWidget(self.btn2, 1, 2)
        
        grid3 = QGridLayout()
        grid3.addWidget(self.btn3, 1, 1)
        grid3.addWidget(self.btn4, 1, 2)
        grid3.addWidget(self.btn5, 1, 3)
 
        hbox = QHBoxLayout()
        hbox.addStretch(2)
        hbox.addLayout(grid2)
        hbox.addStretch(2)
        hbox.addLayout(grid3)
        hbox.addStretch(3)
        
        vbox = QVBoxLayout()
        vbox.addLayout(grid1)
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addStretch(1)
        
        self.setLayout(vbox)    
        
    def closeEvent(self, event):
        import os
        os._exit(1)
        
    def resizeEvent(self, event):
        #900为窗口初始尺寸
        font_size = event.size().width() * 30 // int(900 * self.scale_ratio)
        config = "QLabel{font-size:"+str(font_size)+"px;font-weight:bold;font-family:宋体;}"
        self.label1.setStyleSheet(config)
        self.label2.setStyleSheet(config)
        self.label3.setStyleSheet(config)
        
        w  = event.size().width()  * 320 // 900
        h = event.size().height() * 320 // 400
        self.label11.resize(w, h)
        self.label12.resize(w, h)
        self.text1.setFixedSize(w*200//320, h)
        
        btn_font_size = event.size().width() * 25 // int(900 * self.scale_ratio)
        config = "QPushButton{font-size:"+str(btn_font_size)+"px;font-weight:bold;font-family:宋体;}"
        self.btn1.setStyleSheet(config)
        self.btn2.setStyleSheet(config)
        self.btn3.setStyleSheet(config)
        self.btn4.setStyleSheet(config)
        self.btn5.setStyleSheet(config)
        
    def fun1(self):
        image_path, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png")
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # frame = QImage(image_path)
        frame = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        picture = QPixmap.fromImage(frame).scaled(self.label11.width(), self.label12.height())
        self.label11.setPixmap(picture)
        
    def fun2(self):
        self.btn2.setEnabled(False)
        self.btn2.setText('处理中')
        self.label12.setText("Fun2运行中，请稍后...")
        self.label12.setStyleSheet("color:#FF0000;font-size:25px;}")
        QApplication.processEvents()
        time.sleep(3)
        self.btn2.setEnabled(True)
        self.btn2.setText('按钮二')
        self.label12.setText("Fun2运行结束")
           
    def fun3(self):
        pass
       
    def fun4(self):
        pass
        
    def fun5(self):
        pass
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QtDemo()
    win.show()
    sys.exit(app.exec_())
    sys.exit(0)