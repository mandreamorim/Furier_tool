from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSpinBox, QLabel, QFileDialog, QComboBox, QCheckBox
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QShortcut, QKeySequence, QGuiApplication
import cv2
import numpy as np

class BrushSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.NoFocus)
        self.lineEdit().setReadOnly(True)

    def stepBy(self, steps):
        modifiers = QGuiApplication.keyboardModifiers()
        if modifiers & Qt.ShiftModifier:
            super().stepBy(int(steps * 50))
        else:
            super().stepBy(steps)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ui.canvas import InteractiveCanvas
from core.engine import FFTEngine


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Editor")

        # Widget Central e Layouts
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_v_layout = QVBoxLayout(central_widget)

        # Layout Superior: Canvas (E) e FFT (D) com Labels
        top_layout = QHBoxLayout()

        # Original
        col_original = QVBoxLayout()
        self.label_original = QLabel("Original")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet("font-weight: bold; color: white; margin-bottom: 5px;")
        self.canvas = InteractiveCanvas()
        col_original.addWidget(self.label_original)
        col_original.addWidget(self.canvas)

        # FFT
        col_fft = QVBoxLayout()
        self.label_fft = QLabel("Espectro de Furier")
        self.label_fft.setAlignment(Qt.AlignCenter)
        self.label_fft.setStyleSheet("font-weight: bold; color: white; margin-bottom: 5px;")
        
        self.fig = Figure(facecolor='#121212')
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.canvas_fft = FigureCanvasQTAgg(self.fig)
        
        col_fft.addWidget(self.label_fft)
        col_fft.addWidget(self.canvas_fft)

        # Inverse FFT
        col_ifft = QVBoxLayout()
        self.label_ifft = QLabel("Transformada Inversa")
        self.label_ifft.setAlignment(Qt.AlignCenter)
        self.label_ifft.setStyleSheet("font-weight: bold; color: white; margin-bottom: 5px;")
        
        self.fig2 = Figure(facecolor='#121212')
        self.fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.axis('off')
        self.canvas_fft_2 = FigureCanvasQTAgg(self.fig2)
        
        self.label_ifft.setVisible(False)
        self.canvas_fft_2.setVisible(False)
        
        col_ifft.addWidget(self.label_ifft)
        col_ifft.addWidget(self.canvas_fft_2)

        top_layout.addLayout(col_original)
        top_layout.addLayout(col_fft)
        top_layout.addLayout(col_ifft)
        top_layout.setAlignment(Qt.AlignTop)

        # Layout Inferior: Botões de Operação
        self.button_layout = QHBoxLayout()
        self.reset_button_layout = QHBoxLayout()
        self.third_row_layout = QHBoxLayout()
        self.setup_buttons()

        # Montagem do Layout Principal
        main_v_layout.addLayout(top_layout)
        main_v_layout.addLayout(self.button_layout)
        main_v_layout.addLayout(self.reset_button_layout)
        main_v_layout.addLayout(self.third_row_layout)

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
            self.canvas.set_image_new(img)

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

        btn_load = QPushButton("Carregar Imagem")
        btn_load.clicked.connect(self.load_image_dialog)
        self.reset_button_layout.addWidget(btn_load)

        # Toggle de visualização RGB
        self.toggle_rgb = QCheckBox("Vis. RGB")
        self.toggle_rgb.stateChanged.connect(self.toggle_rgb_view)
        self.reset_button_layout.addWidget(self.toggle_rgb)

        # Botão de reset isolado numa linha abaixo
        btn_reset = QPushButton("Resetar")
        btn_reset.clicked.connect(self.canvas.reset)
        self.reset_button_layout.addWidget(btn_reset)

        # Adicionar controle de tamanho do pincel
        size_label = QLabel("Tamanho:")
        self.size_spin = BrushSpinBox()
        self.size_spin.setRange(1, 200)
        self.size_spin.setValue(15)
        self.size_spin.valueChanged.connect(self.update_brush_size)

        # Adicionar controle de formato do pincel
        shape_label = QLabel("Formato:")
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(["Círculo", "Quadrado"])
        self.shape_combo.setFocusPolicy(Qt.NoFocus)
        self.shape_combo.currentTextChanged.connect(self.update_brush_shape)

        # Adicionar controle de tom do pincel na mesma linha
        shade_label = QLabel("Tom:")
        self.shade_spin = BrushSpinBox()
        self.shade_spin.setRange(0, 255)
        self.shade_spin.setValue(255)
        self.shade_spin.valueChanged.connect(self.update_brush_shade)

        self.reset_button_layout.addStretch()
        self.reset_button_layout.addWidget(size_label)
        self.reset_button_layout.addWidget(self.size_spin)
        self.reset_button_layout.addWidget(shape_label)
        self.reset_button_layout.addWidget(self.shape_combo)
        self.reset_button_layout.addWidget(shade_label)
        self.reset_button_layout.addWidget(self.shade_spin)

        # Terceira linha de botões e selects
        self.check_extra_view = QCheckBox("Trasnformada inversa")
        self.check_extra_view.stateChanged.connect(self.toggle_extra_view)
        
        self.combo_extra_filter = QComboBox()
        # Adiciona itens com categorias
        self.combo_extra_filter.addItem("Nenhum")
        self.combo_extra_filter.insertSeparator(1)
        
        self.combo_extra_filter.addItem("--- PASSA-BAIXA ---")
        self.combo_extra_filter.addItems(["Gaussiano", "Média", "Mediana", "K-Vizinhos"])
        
        self.combo_extra_filter.insertSeparator(self.combo_extra_filter.count())
        self.combo_extra_filter.addItem("--- PASSA-ALTA ---")
        self.combo_extra_filter.addItems(["Canny", "Sobel", "Laplace"])
        
        # Desabilita os headers para não serem selecionáveis
        model = self.combo_extra_filter.model()
        model.item(2).setEnabled(False) # PASSA-BAIXA
        model.item(8).setEnabled(False) # PASSA-ALTA
        
        self.combo_extra_filter.setVisible(False)
        self.combo_extra_filter.currentTextChanged.connect(self.update_fft)
        
        self.third_row_layout.addWidget(self.check_extra_view)
        self.third_row_layout.addWidget(self.combo_extra_filter)
        self.third_row_layout.addStretch()

    def update_brush_size(self, value):
        self.canvas.brush_size = value

    def update_brush_shape(self, text):
        self.canvas.brush_shape = text

    def load_image_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem", "", "Imagens (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if file_name:
            img = cv2.imread(file_name)
            if img is not None:
                self.canvas.set_image_new(img)

    def update_brush_shade(self, value):
        self.canvas.brush_shade = value

    def toggle_rgb_view(self, state):
        self.canvas.show_rgb = (state == Qt.Checked.value)
        self.canvas.update_display()

    def apply_op(self, func):
        # Salva estado atual na pilha antes de aplicar a operação
        self.canvas.undo_stack.append((self.canvas.data.copy(), self.canvas.data_rgb.copy()))
        
        # Aplica a operação na versão RGB e gera a versão P&B a partir dela
        # para garantir consistência (especialmente em operações aleatórias)
        self.canvas.data_rgb = func(self.canvas.data_rgb)
        self.canvas.data = cv2.cvtColor(self.canvas.data_rgb, cv2.COLOR_RGB2GRAY)
        
        self.canvas.update_display()
        self.canvas.changed.emit()

    def toggle_extra_view(self, state):
        is_checked = (state == Qt.Checked.value)
        self.label_ifft.setVisible(is_checked)
        self.combo_extra_filter.setVisible(is_checked)
        self.canvas_fft_2.setVisible(is_checked)
        if is_checked:
            self.update_fft()

    def update_fft(self):
        mag = FFTEngine.get_fft_magnitude(self.canvas.data)
        
        # Atualiza primeiro FFT
        self.canvas_fft.setFixedSize(self.canvas.width(), self.canvas.height())
        self.ax.clear()
        self.ax.imshow(mag, cmap='gray', aspect='auto')
        self.ax.axis('off')
        self.canvas_fft.draw()

        # Atualiza segundo FFT se estiver visível
        if self.canvas_fft_2.isVisible():
            # 1. Obter imagem base
            img_process = self.canvas.data.copy()
            
            # 2. Aplicar filtro espacial se selecionado
            filter_name = self.combo_extra_filter.currentText()
            if filter_name == "Gaussiano":
                img_process = FFTEngine.apply_gaussian(img_process)
            elif filter_name == "Média":
                img_process = FFTEngine.apply_mean(img_process)
            elif filter_name == "Mediana":
                img_process = FFTEngine.apply_median(img_process)
            elif filter_name == "K-Vizinhos":
                img_process = FFTEngine.apply_knn(img_process)
            elif filter_name == "Canny":
                img_process = FFTEngine.apply_canny(img_process)
            elif filter_name == "Sobel":
                img_process = FFTEngine.apply_sobel(img_process)
            elif filter_name == "Laplace":
                img_process = FFTEngine.apply_laplacian(img_process)
            
            # 3. Calcular FFT -> IFFT para demonstrar
            f_complex = FFTEngine.get_fft_complex(img_process)
            result_img = FFTEngine.apply_ifft(f_complex)

            self.canvas_fft_2.setFixedSize(self.canvas.width(), self.canvas.height())
            self.ax2.clear()
            self.ax2.imshow(result_img, cmap='gray', aspect='auto')
            self.ax2.axis('off')
            self.canvas_fft_2.draw()

    def set_image(self, image):
        self.canvas.set_image_new(image)