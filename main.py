import sys

import cv2
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QShortcut, QKeySequence
from ui.canvas import InteractiveCanvas
from core.engine import FFTEngine
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Canvas v1.0")

        # Layout Principal
        central = QWidget()
        layout = QHBoxLayout(central)

        # 1. Canvas (Esquerda)
        self.canvas = InteractiveCanvas()

        # 2. Gráfico FFT (Direita)
        self.fig = Figure(figsize=(5, 5), facecolor='#121212')
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas_fft = FigureCanvasQTAgg(self.fig)

        layout.addWidget(self.canvas)
        layout.addWidget(self.canvas_fft)
        self.setCentralWidget(central)

        # Timer para o Delay (Debounce de 50ms para fluidez)
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_fft)
        self.canvas.changed.connect(lambda: self.timer.start(50))

        # Atalho Ctrl+Z
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.canvas.undo)

    def update_fft(self):
        mag = FFTEngine.get_fft_magnitude(self.canvas.data)
        self.ax.clear()
        self.ax.imshow(mag, cmap='magma')
        self.ax.axis('off')
        self.canvas_fft.draw()

    def set_image(self, image):
        self.canvas.set_image_new(image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.set_image(cv2.imread("manuel.jpg"))
    sys.exit(app.exec())