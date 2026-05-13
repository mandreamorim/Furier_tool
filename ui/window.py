from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QShortcut, QKeySequence
import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ui.canvas import InteractiveCanvas
from core.engine import FFTEngine


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Editor - Data Lumini Edition")

        # Widget Central e Layouts
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_v_layout = QVBoxLayout(central_widget)

        # Layout Superior: Canvas (E) e FFT (D)
        top_layout = QHBoxLayout()

        self.canvas = InteractiveCanvas()

        # Configuração do Matplotlib para o Gráfico
        self.fig = Figure(figsize=(5, 5), facecolor='#121212')
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas_fft = FigureCanvasQTAgg(self.fig)

        top_layout.addWidget(self.canvas)
        top_layout.addWidget(self.canvas_fft)

        # Layout Inferior: Botões de Operação
        self.button_layout = QHBoxLayout()
        self.setup_buttons()

        # Montagem do Layout Principal
        main_v_layout.addLayout(top_layout)
        main_v_layout.addLayout(self.button_layout)

        # Timer para atualização reativa (Debounce)
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.update_fft)
        self.canvas.changed.connect(lambda: self.timer.start(50))

        # Atalhos
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.canvas.undo)

        # Carregar imagem inicial se existir
        self.load_initial_image()

    def load_initial_image(self):
        img = cv2.imread("manuel.jpg")
        if img is not None:
            self.canvas.set_image(img)

    def setup_buttons(self):
        actions = [
            ("Passa-Baixa", lambda: self.apply_op(FFTEngine.apply_low_pass)),
            ("Passa-Alta", lambda: self.apply_op(FFTEngine.apply_high_pass)),
            ("Erosão", lambda: self.apply_op(lambda img: FFTEngine.apply_morphology(img, "erosion"))),
            ("Dilatação", lambda: self.apply_op(lambda img: FFTEngine.apply_morphology(img, "dilation"))),
            ("Ruído Gauss", lambda: self.apply_op(lambda img: FFTEngine.add_noise(img, "gauss"))),
        ]

        for text, func in actions:
            btn = QPushButton(text)
            btn.clicked.connect(func)
            self.button_layout.addWidget(btn)

    def apply_op(self, func):
        # Salva estado atual na pilha antes de aplicar a operação
        self.canvas.undo_stack.append(self.canvas.data.copy())
        self.canvas.data = func(self.canvas.data)
        self.canvas.update_display()
        self.canvas.changed.emit()

    def update_fft(self):
        mag = FFTEngine.get_fft_magnitude(self.canvas.data)
        self.ax.clear()
        self.ax.imshow(mag, cmap='magma')
        self.ax.axis('off')
        self.canvas_fft.draw()

    def set_image(self, image):
        self.canvas.set_image_new(image)