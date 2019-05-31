import example
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel)
from PyQt5.QtGui import (QPainter, QPen, QFont)
from PyQt5.QtCore import Qt
import sys
from PyQt5.QtWidgets import QApplication
from PIL import ImageGrab, Image
import time

class MyMnistWindow(QWidget):

    def __init__(self):
        super(MyMnistWindow, self).__init__()
        self.resize(284, 330)
        self.move(100, 100) 
        self.setWindowFlags(Qt.FramelessWindowHint) 
        self.setMouseTracking(False)
        self.pos_xy = [] 
        self.label_draw = QLabel('', self)
        self.label_draw.setGeometry(2, 2, 280, 280)
        self.label_draw.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_draw.setAlignment(Qt.AlignCenter)
        self.label_result_name = QLabel('result: ', self)
        self.label_result_name.setGeometry(2, 290, 60, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)
        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(64, 290, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.btn_recognize = QPushButton("run", self)
        self.btn_recognize.setGeometry(110, 290, 50, 35)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)
        self.btn_clear = QPushButton("clean", self)
        self.btn_clear.setGeometry(170, 290, 50, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)
        self.btn_close = QPushButton("close", self)
        self.btn_close.setGeometry(230, 290, 50, 35)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 30, Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        pos_tmp = (event.pos().x(), event.pos().y())
        self.pos_xy.append(pos_tmp)
        self.update()

    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        self.update()

    def btn_recognize_on_clicked(self):
        bbox = (104, 104, 380, 380)
        im = ImageGrab.grab(bbox) 
        im = im.resize((28, 28), Image.ANTIALIAS)
        recognize_result = self.recognize_img(im)
        self.label_result.setText(str(recognize_result))
        self.update()

    def btn_clear_on_clicked(self):
        self.pos_xy = []
        self.label_result.setText('')
        self.update()

    def btn_close_on_clicked(self):
        self.close()

    def recognize_img(self, img):
        '''
        
        '''
        myimage = img.convert('L')
        tv = list(myimage.getdata())
        tva = [0 if (255 - x) * 1.0 / 255.0<0.5 else 1 for x in tv]
        start=time.clock()
        n=example.Network()
        n.load_network("../data/model-neural-network.dat")
        n.read_from_board(tva)
        result = n.test()
        print(time.clock()-start)
        return result
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mymnist = MyMnistWindow()
    mymnist.show()
    app.exec_()