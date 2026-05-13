import numpy as np
import cv2
from scipy import fft


class FFTEngine:
    @staticmethod
    def get_fft_magnitude(image_array):
        # Converte para float32 para precisão
        f_transform = fft.fft2(image_array.astype(np.float32))
        f_shift = fft.fftshift(f_transform)

        # Magnitude em log para visualização
        magnitude = 20 * np.log(np.abs(f_shift) + 1)

        # Normaliza para 0-255 para o Matplotlib/OpenCV
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)