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
        self.data_rgb = np.zeros((size, size, 3), dtype=np.uint8)
        self.original_data = None
        self.original_data_rgb = None
        self.undo_stack = deque(maxlen=30)
        self.brush_size = 15
        self.brush_shape = "Círculo"
        self.brush_shade = 255
        self.show_rgb = False
        self.update_display()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.undo_stack.append((self.data.copy(), self.data_rgb.copy()))

    def reset(self):
        if self.original_data is not None:
            self.undo_stack.append((self.data.copy(), self.data_rgb.copy()))
            self.data = self.original_data.copy()
            self.data_rgb = self.original_data_rgb.copy()
            self.update_display()
            self.changed.emit()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            x, y = int(event.position().x()), int(event.position().y())
            shade_rgb = (self.brush_shade, self.brush_shade, self.brush_shade)
            if self.brush_shape == "Círculo":
                cv2.circle(self.data, (x, y), self.brush_size, self.brush_shade, -1)
                cv2.circle(self.data_rgb, (x, y), self.brush_size, shade_rgb, -1)
            else:
                p1 = (x - self.brush_size, y - self.brush_size)
                p2 = (x + self.brush_size, y + self.brush_size)
                cv2.rectangle(self.data, p1, p2, self.brush_shade, -1)
                cv2.rectangle(self.data_rgb, p1, p2, shade_rgb, -1)
            self.update_display()
            self.changed.emit()

    def undo(self):
        if self.undo_stack:
            self.data, self.data_rgb = self.undo_stack.pop()
            self.update_display()
            self.changed.emit()

    def update_display(self):
        if self.show_rgb:
            h, w, c = self.data_rgb.shape
            bytes_per_line = c * w
            q_img = QImage(self.data_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            h, w = self.data.shape
            q_img = QImage(self.data.data, w, h, w, QImage.Format_Grayscale8)
        self.setPixmap(QPixmap.fromImage(q_img))

    def set_image(self, image):
        # Esta função parece estar em desuso a favor da set_image_new, 
        # mas vamos mantê-la funcional
        if len(image.shape) == 3:
            self.data_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.data = image
            self.data_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        self.data = cv2.resize(self.data, (self.width(), self.height()))
        self.data_rgb = cv2.resize(self.data_rgb, (self.width(), self.height()))
        self.update_display()
        self.changed.emit()

    def set_image_new(self, image_array):
        """Carrega imagem sem padding para não distorcer frequências."""
        h, w = image_array.shape[:2]
        max_side = 512

        # 1. Calcula escala para caber sem distorcer
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        # 2. Redimensiona preservando energia (INTER_AREA)
        resized = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if len(resized.shape) == 3:
            self.data_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.data = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            self.data = resized
            self.data_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        # 3. Ajusta o tamanho do widget
        self.setFixedSize(new_w, new_h)
        self.original_data = self.data.copy()
        self.original_data_rgb = self.data_rgb.copy()

        self.update_display()
        self.changed.emit()