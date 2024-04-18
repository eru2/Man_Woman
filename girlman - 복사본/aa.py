import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from keras.models import load_model
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Teachable Machine with PyQt5')

        self.imageLabel = QLabel(self)#설정하는코드
        self.imageLabel.resize(640, 480)#이미지 라벨
        
        self.resultLabel = QLabel("분석 결과를 여기에 표시합니다.", self)
        
        self.endBtn = QPushButton('종료', self)
        self.endBtn.clicked.connect(self.endCam)
        
        layout = QVBoxLayout()#위젯수직배열
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.resultLabel)
        layout.addWidget(self.endBtn)
        
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.camera = cv2.VideoCapture(0)

        self.model = load_model("keras_Model.h5", compile=False)
        with open("labels.txt", "r", encoding="utf-8") as file:
            self.class_names = [line.strip() for line in file.readlines()]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateImage)
        self.timer.start(30)
    
    def updateImage(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_array = np.asarray(rgb_frame, dtype=np.float32).reshape(1, 224, 224, 3)
            image_array = (image_array / 127.5) - 1

            prediction = self.model.predict(image_array)
            index = np.argmax(prediction)
            class_name = self.class_names[index]

            self.resultLabel.setText(f"Class: {class_name}, Confidence: {prediction[0][index]*100:.2f}%")

            qImage = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImage)
            self.imageLabel.setPixmap(pixmap)

    def endCam(self):
        self.timer.stop()
        self.camera.release()
        cv2.destroyAllWindows()
        self.close()  

app = QApplication(sys.argv)
mainWindow = MainWindow()
mainWindow.show()
sys.exit(app.exec_())
