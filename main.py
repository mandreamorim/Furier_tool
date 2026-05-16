import sys

import cv2
from PySide6.QtWidgets import QApplication

from ui.window import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    window.set_image(cv2.imread("manuel.jpg"))
    sys.exit(app.exec())