import numpy as np
import cv2
from scipy import fft


class FFTEngine:
    @staticmethod
    def get_fft_magnitude(image_array):
        # Converte para float32 para precisão
        f_shift = FFTEngine.get_fft_complex(image_array)

        # Magnitude em log para visualização
        magnitude = np.log(np.abs(f_shift) + 1) * 20

        # Normaliza para 0-255 para o Matplotlib/OpenCV
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def get_fft_complex(image_array):
        f_transform = fft.fft2(image_array.astype(np.float32))
        return fft.fftshift(f_transform)

    @staticmethod
    def apply_ifft(f_shift):
        f_ishift = fft.ifftshift(f_shift)
        img_back = fft.ifft2(f_ishift)
        return np.clip(np.abs(img_back), 0, 255).astype(np.uint8)

    @staticmethod
    def apply_low_pass(image):
        """Suavização (Blur) - Remove altas frequências."""
        return cv2.GaussianBlur(image, (9, 9), 0)

    @staticmethod
    def apply_gaussian(image, kernel_size=9):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def apply_mean(image, kernel_size=9):
        return cv2.blur(image, (kernel_size, kernel_size))

    @staticmethod
    def apply_median(image, kernel_size=9):
        # MedianBlur requer kernel ímpar e imagem 8 bits ou float32
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def apply_knn(image):
        """Filtro de K-vizinhos (usando Bilateral como aproximação comum)."""
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, 9, 75, 75)
        return cv2.bilateralFilter(image, 9, 75, 75)

    @staticmethod
    def apply_canny(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return cv2.Canny(image, 100, 200)

    @staticmethod
    def apply_sobel(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Binarização de Otsu primeiro
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        if len(image.shape) == 3:
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges

    @staticmethod
    def apply_laplacian(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Binarização de Otsu primeiro
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        lap = cv2.Laplacian(binary, cv2.CV_64F)
        edges = cv2.convertScaleAbs(lap)
        
        if len(image.shape) == 3:
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges

    @staticmethod
    def apply_high_pass(image):
        """Realce de bordas (Laplacian) - Mantém altas frequências."""
        # Esta já existia, mas agora temos a Laplacian pura acima
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