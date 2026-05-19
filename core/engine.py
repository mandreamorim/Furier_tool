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
        magnitude = np.log(np.abs(f_shift) + 1) * 20

        # Normaliza para 0-255 para o Matplotlib/OpenCV
        # return magnitude
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def apply_low_pass(image):
        """Suavização (Blur) - Remove altas frequências."""
        return cv2.GaussianBlur(image, (9, 9), 0)

    @staticmethod
    def apply_high_pass(image):
        """Realce de bordas (Laplacian) - Mantém altas frequências."""
        if len(image.shape) == 3:
            # Para RGB, aplicar laplaciano e subtrair de cada canal
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            return cv2.convertScaleAbs(image.astype(float) - laplacian)
        else:
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            return cv2.convertScaleAbs(image.astype(float) - laplacian)

    @staticmethod
    def apply_morphology(image, op_type="erosion"):
        kernel = np.ones((5, 5), np.uint8)
        if op_type == "erosion":
            return cv2.erode(image, kernel, iterations=1)
        return cv2.dilate(image, kernel, iterations=1)

    @staticmethod
    def add_noise(image, noise_type="gauss"):
        if noise_type == "gauss":
            shape = image.shape
            mean = 0
            sigma = 30
            gauss = np.random.normal(mean, sigma, shape)
            noisy = image + gauss
            return np.clip(noisy, 0, 255).astype(np.uint8)
        elif noise_type == "s&p":
            prob = 0.05
            noisy = image.copy()
            # Salt
            thres = 1 - prob
            rdn = np.random.random(image.shape)
            noisy[rdn > thres] = 255
            # Pepper
            noisy[rdn < prob] = 0
            return noisy