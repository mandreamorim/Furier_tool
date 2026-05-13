from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, Signal
import numpy as np
import cv2
from collections import deque

class InteractiveCanvas(QLabel):
    changed = Signal()

    def __init__(self, size=512):
        super().__init__()
        self.setFixedSize(size, size)
        self.data = np.zeros((size, size), dtype=np.uint8)
        self.undo_stack = deque(maxlen=30)
        self.brush_size = 15
        self.update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.undo_stack.append(self.data.copy())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            x, y = int(event.position().x()), int(event.position().y())
            cv2.circle(self.data, (x, y), self.brush_size, 255, -1)
            self.update_display()
            self.changed.emit()

    def undo(self):
        if self.undo_stack:
            self.data = self.undo_stack.pop()
            self.update_display()
            self.changed.emit()

    def update_display(self):
        h, w = self.data.shape
        q_img = QImage(self.data.data, w, h, w, QImage.Format_Grayscale8)
        self.setPixmap(QPixmap.fromImage(q_img))

    def set_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            self.data = cv2.resize(image, (self.width(), self.height()))
            self.update_display()
            self.changed.emit()

    def set_image_new(self, image_array):
        """Carrega imagem com padding para não distorcer frequências."""
        h, w = image_array.shape[:2]
        side = self.width()  # 512

        # 1. Calcula escala para caber sem distorcer
        scale = side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        # 2. Redimensiona preservando energia (INTER_AREA)
        resized = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 3. Centraliza no canvas preto
        self.data = np.zeros((side, side), dtype=np.uint8)
        x_off = (side - new_w) // 2
        y_off = (side - new_h) // 2
        self.data[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        self.update_display()
        self.changed.emit()