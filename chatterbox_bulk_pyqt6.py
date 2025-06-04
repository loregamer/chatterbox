import sys
import os
import random
import numpy as np
import torch
import torchaudio as ta
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QSlider, QSpinBox, QTabWidget,
    QFileDialog, QGroupBox, QGridLayout, QComboBox, QProgressBar,
    QMessageBox, QDoubleSpinBox, QCheckBox, QSplitter, QListWidget,
    QListWidgetItem, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl, QThreadPool, QRunnable, pyqtSlot
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon, QDragEnterEvent, QDropEvent
import sounddevice as sd
import soundfile as sf
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC


class AudioRecorder:
    """Simple audio recorder using sounddevice"""
    def __init__(self):
        self.recording = False
        self.frames = []
        self.samplerate = 44100
        self.channels = 1
        self.q = queue.Queue()
        
    def start_recording(self):
        self.recording = True
        self.frames = []
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self.audio_callback
        )
        self.stream.start()
        
    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.q.put(indata.copy())
            
    def stop_recording(self):
        self.recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            
        # Process all queued audio
        while not self.q.empty():
            self.frames.append(self.q.get())
            
        if self.frames:
            return np.vstack(self.frames)
        return None
        
    def save_recording(self, filename):
        audio_data = self.stop_recording()
        if audio_data is not None:
            sf.write(filename, audio_data, self.samplerate)
            return True
        return False


class BulkTTSWorker(QThread):
    """Worker thread for bulk TTS generation"""
    progress = pyqtSignal(int, int, str)  # current, total, message
    file_completed = pyqtSignal(str, str)  # input_text, output_path
    error = pyqtSignal(str, str)  # text, error_message
    finished = pyqtSignal()
    
    def __init__(self, model, texts, output_dir, base_filename, params):
        super().__init__()
        self.model = model
        self.texts = texts
        self.output_dir = output_dir
        self.base_filename = base_filename
        self.params = params
        self._is_running = True
        
    def run(self):
        total = len(self.texts)
        
        for idx, text in enumerate(self.texts):
            if not self._is_running:
                break
                
            try:
                self.progress.emit(idx + 1, total, f"Processing: {text[:50]}...")
                
                # Set seed if specified
                if self.params['seed'] != 0:
                    self.set_seed(self.params['seed'] + idx)  # Different seed per file
                
                # Generate audio
                wav = self.model.generate(
                    text,
                    audio_prompt_path=self.params.get('ref_audio'),
                    exaggeration=self.params['exaggeration'],
                    temperature=self.params['temperature'],
                    cfg_weight=self.params['cfg_weight'],
                )
                
                # Save with numbered filename
                if total == 1:
                    filename = f"{self.base_filename}.wav"
                else:
                    filename = f"{self.base_filename}_{idx+1:03d}.wav"
                output_path = os.path.join(self.output_dir, filename)
                
                ta.save(output_path, wav, self.model.sr)
                self.file_completed.emit(text, output_path)
                
            except Exception as e:
                self.error.emit(text, str(e))
                
        self.finished.emit()
        
    def stop(self):
        self._is_running = False
        
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)


class BulkVCWorker(QThread):
    """Worker thread for bulk voice conversion"""
    progress = pyqtSignal(int, int, str)  # current, total, message
    file_completed = pyqtSignal(str, str)  # input_path, output_path
    error = pyqtSignal(str, str)  # input_path, error_message
    finished = pyqtSignal()
    
    def __init__(self, model, input_files, output_dir, target_voice_path):
        super().__init__()
        self.model = model
        self.input_files = input_files
        self.output_dir = output_dir
        self.target_voice_path = target_voice_path
        self._is_running = True
        
    def run(self):
        total = len(self.input_files)
        
        for idx, input_path in enumerate(self.input_files):
            if not self._is_running:
                break
                
            try:
                self.progress.emit(idx + 1, total, f"Converting: {Path(input_path).name}")
                
                # Generate converted audio
                wav = self.model.generate(
                    audio=input_path,
                    target_voice_path=self.target_voice_path,
                )
                
                # Save with _converted suffix
                input_name = Path(input_path).stem
                output_filename = f"{input_name}_converted.wav"
                output_path = os.path.join(self.output_dir, output_filename)
                
                ta.save(output_path, wav, self.model.sr)
                self.file_completed.emit(input_path, output_path)
                
            except Exception as e:
                self.error.emit(input_path, str(e))
                
        self.finished.emit()
        
    def stop(self):
        self._is_running = False


class DragDropListWidget(QListWidget):
    """List widget that accepts file drops"""
    files_dropped = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                files.append(file_path)
        if files:
            self.files_dropped.emit(files)


class CompactParamWidget(QWidget):
    """Compact parameter control widget"""
    def __init__(self):
        super().__init__()
        layout = QGridLayout()
        layout.setSpacing(5)
        
        # Row 1: Exaggeration and CFG
        layout.addWidget(QLabel("Exag:"), 0, 0)
        self.exaggeration_slider = QSlider(Qt.Orientation.Horizontal)
        self.exaggeration_slider.setRange(25, 200)
        self.exaggeration_slider.setValue(50)
        self.exaggeration_slider.setMaximumWidth(120)
        self.exaggeration_value = QLabel("0.50")
        self.exaggeration_value.setMinimumWidth(40)
        self.exaggeration_slider.valueChanged.connect(
            lambda v: self.exaggeration_value.setText(f"{v/100:.2f}")
        )
        layout.addWidget(self.exaggeration_slider, 0, 1)
        layout.addWidget(self.exaggeration_value, 0, 2)
        
        layout.addWidget(QLabel("CFG:"), 0, 3)
        self.cfg_slider = QSlider(Qt.Orientation.Horizontal)
        self.cfg_slider.setRange(0, 100)
        self.cfg_slider.setValue(50)
        self.cfg_slider.setMaximumWidth(120)
        self.cfg_value = QLabel("0.50")
        self.cfg_value.setMinimumWidth(40)
        self.cfg_slider.valueChanged.connect(
            lambda v: self.cfg_value.setText(f"{v/100:.2f}")
        )
        layout.addWidget(self.cfg_slider, 0, 4)
        layout.addWidget(self.cfg_value, 0, 5)
        
        # Row 2: Temperature and Seed
        layout.addWidget(QLabel("Temp:"), 1, 0)
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(5, 500)
        self.temperature_slider.setValue(80)
        self.temperature_slider.setMaximumWidth(120)
        self.temperature_value = QLabel("0.80")
        self.temperature_value.setMinimumWidth(40)
        self.temperature_slider.valueChanged.connect(
            lambda v: self.temperature_value.setText(f"{v/100:.2f}")
        )
        layout.addWidget(self.temperature_slider, 1, 1)
        layout.addWidget(self.temperature_value, 1, 2)
        
        layout.addWidget(QLabel("Seed:"), 1, 3)
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 999999)
        self.seed_input.setValue(0)
        self.seed_input.setSpecialValueText("Random")
        self.seed_input.setMaximumWidth(100)
        layout.addWidget(self.seed_input, 1, 4, 1, 2)
        
        self.setLayout(layout)
        
    def get_params(self):
        return {
            'exaggeration': self.exaggeration_slider.value() / 100,
            'cfg_weight': self.cfg_slider.value() / 100,
            'temperature': self.temperature_slider.value() / 100,
            'seed': self.seed_input.value()
        }


class BulkChatterBoxGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatterBox Bulk Processor")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set up device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        # Initialize models (will be loaded lazily)
        self.tts_model = None
        self.vc_model = None
        
        # Audio components
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        # Audio recorder
        self.recorder = AudioRecorder()
        
        # Output directory
        self.output_dir = str(Path.home() / "ChatterBox_Output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup UI
        self.init_ui()
        self.apply_theme()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        
        # Compact Header
        header_layout = QHBoxLayout()
        title_label = QLabel("ChatterBox Bulk Processor")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        device_label = QLabel(f"Device: {self.device.upper()}")
        device_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
        header_layout.addStretch()
        
        # Output directory selector
        self.output_dir_label = QLabel(f"Output: {self.output_dir}")
        self.output_dir_label.setMaximumWidth(300)
        self.output_dir_label.setStyleSheet("QLabel { color: #888; }")
        output_btn = QPushButton("Change")
        output_btn.clicked.connect(self.select_output_dir)
        output_btn.setMaximumWidth(80)
        
        header_layout.addWidget(self.output_dir_label)
        header_layout.addWidget(output_btn)
        header_layout.addWidget(device_label)
        
        main_layout.addLayout(header_layout)
        
        # Tab Widget
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.create_bulk_tts_tab(), "Bulk TTS")
        self.tab_widget.addTab(self.create_bulk_vc_tab(), "Bulk Voice Conversion")
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar with progress
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_progress = QProgressBar()
        self.status_progress.setVisible(False)
        self.status_progress.setMaximumHeight(20)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_progress)
        
        main_layout.addLayout(status_layout)
        
    def create_bulk_tts_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Top controls
        top_layout = QHBoxLayout()
        
        # Input type selector
        input_group = QGroupBox("Input Mode")
        input_layout = QHBoxLayout()
        self.tts_multiline_radio = QCheckBox("Multi-line (one file per line)")
        self.tts_multiline_radio.setChecked(True)
        self.tts_single_radio = QCheckBox("Single text")
        self.tts_single_radio.toggled.connect(lambda checked: self.tts_multiline_radio.setChecked(not checked))
        self.tts_multiline_radio.toggled.connect(lambda checked: self.tts_single_radio.setChecked(not checked))
        
        input_layout.addWidget(self.tts_multiline_radio)
        input_layout.addWidget(self.tts_single_radio)
        input_group.setLayout(input_layout)
        top_layout.addWidget(input_group)
        
        # Reference voice
        ref_group = QGroupBox("Reference Voice")
        ref_layout = QHBoxLayout()
        self.tts_ref_label = QLabel("None")
        self.tts_ref_label.setMinimumWidth(150)
        ref_load_btn = QPushButton("Load")
        ref_load_btn.clicked.connect(self.load_tts_reference)
        ref_record_btn = QPushButton("Record")
        ref_record_btn.clicked.connect(self.toggle_tts_recording)
        ref_clear_btn = QPushButton("Clear")
        ref_clear_btn.clicked.connect(self.clear_tts_reference)
        
        ref_layout.addWidget(self.tts_ref_label)
        ref_layout.addWidget(ref_load_btn)
        ref_layout.addWidget(ref_record_btn)
        ref_layout.addWidget(ref_clear_btn)
        ref_group.setLayout(ref_layout)
        top_layout.addWidget(ref_group)
        
        top_layout.addStretch()
        layout.addLayout(top_layout)
        
        # Parameters
        self.tts_params = CompactParamWidget()
        param_group = QGroupBox("Generation Parameters")
        param_layout = QVBoxLayout()
        param_layout.addWidget(self.tts_params)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Text input
        text_group = QGroupBox("Text Input")
        text_layout = QVBoxLayout()
        
        # Add template buttons
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Templates:"))
        
        story_btn = QPushButton("Story")
        story_btn.clicked.connect(lambda: self.load_tts_template("story"))
        game_btn = QPushButton("Game Dialog")
        game_btn.clicked.connect(lambda: self.load_tts_template("game"))
        announcement_btn = QPushButton("Announcements")
        announcement_btn.clicked.connect(lambda: self.load_tts_template("announcement"))
        
        template_layout.addWidget(story_btn)
        template_layout.addWidget(game_btn)
        template_layout.addWidget(announcement_btn)
        template_layout.addStretch()
        
        text_layout.addLayout(template_layout)
        
        self.tts_text_input = QTextEdit()
        self.tts_text_input.setPlainText(
            "Welcome to ChatterBox bulk processing.\n"
            "Each line will be generated as a separate audio file.\n"
            "You can also switch to single text mode for one long text."
        )
        text_layout.addWidget(self.tts_text_input)
        
        # File naming
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Base filename:"))
        self.tts_filename_input = QLineEdit("output")
        self.tts_filename_input.setMaximumWidth(200)
        name_layout.addWidget(self.tts_filename_input)
        name_layout.addWidget(QLabel("(Files will be named: output_001.wav, output_002.wav, etc.)"))
        name_layout.addStretch()
        
        text_layout.addLayout(name_layout)
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)
        
        # Generate button and progress
        generate_layout = QHBoxLayout()
        self.tts_generate_btn = QPushButton("Generate All")
        self.tts_generate_btn.clicked.connect(self.generate_bulk_tts)
        self.tts_generate_btn.setMinimumHeight(40)
        self.tts_generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.tts_stop_btn = QPushButton("Stop")
        self.tts_stop_btn.clicked.connect(self.stop_tts_generation)
        self.tts_stop_btn.setEnabled(False)
        self.tts_stop_btn.setMinimumHeight(40)
        self.tts_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
        generate_layout.addWidget(self.tts_generate_btn)
        generate_layout.addWidget(self.tts_stop_btn)
        generate_layout.addStretch()
        
        layout.addLayout(generate_layout)
        
        # Results table
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.tts_results_table = QTableWidget()
        self.tts_results_table.setColumnCount(4)
        self.tts_results_table.setHorizontalHeaderLabels(["Text", "Output File", "Status", "Actions"])
        self.tts_results_table.horizontalHeader().setStretchLastSection(True)
        self.tts_results_table.setAlternatingRowColors(True)
        
        results_layout.addWidget(self.tts_results_table)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.tts_ref_path = None
        self.tts_worker = None
        
        return widget
        
    def create_bulk_vc_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Input files section
        input_group = QGroupBox("Input Audio Files (Drag & Drop or Click Add)")
        input_layout = QVBoxLayout()
        
        # File list
        self.vc_file_list = DragDropListWidget()
        self.vc_file_list.files_dropped.connect(self.add_vc_files)
        input_layout.addWidget(self.vc_file_list)
        
        # File controls
        file_controls = QHBoxLayout()
        add_files_btn = QPushButton("Add Files")
        add_files_btn.clicked.connect(self.browse_vc_files)
        clear_files_btn = QPushButton("Clear All")
        clear_files_btn.clicked.connect(self.vc_file_list.clear)
        
        file_controls.addWidget(add_files_btn)
        file_controls.addWidget(clear_files_btn)
        file_controls.addWidget(QLabel("Total files:"))
        self.vc_file_count_label = QLabel("0")
        file_controls.addWidget(self.vc_file_count_label)
        file_controls.addStretch()
        
        input_layout.addLayout(file_controls)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Target voice
        target_group = QGroupBox("Target Voice")
        target_layout = QHBoxLayout()
        
        self.vc_target_label = QLabel("Using default voice")
        self.vc_target_label.setMinimumWidth(200)
        target_load_btn = QPushButton("Load")
        target_load_btn.clicked.connect(self.load_vc_target)
        target_record_btn = QPushButton("Record")
        target_record_btn.clicked.connect(self.toggle_vc_recording)
        target_clear_btn = QPushButton("Clear")
        target_clear_btn.clicked.connect(self.clear_vc_target)
        
        target_layout.addWidget(self.vc_target_label)
        target_layout.addWidget(target_load_btn)
        target_layout.addWidget(target_record_btn)
        target_layout.addWidget(target_clear_btn)
        target_layout.addStretch()
        
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # Convert button
        convert_layout = QHBoxLayout()
        self.vc_convert_btn = QPushButton("Convert All")
        self.vc_convert_btn.clicked.connect(self.convert_bulk_vc)
        self.vc_convert_btn.setMinimumHeight(40)
        self.vc_convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        
        self.vc_stop_btn = QPushButton("Stop")
        self.vc_stop_btn.clicked.connect(self.stop_vc_conversion)
        self.vc_stop_btn.setEnabled(False)
        self.vc_stop_btn.setMinimumHeight(40)
        self.vc_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
        convert_layout.addWidget(self.vc_convert_btn)
        convert_layout.addWidget(self.vc_stop_btn)
        convert_layout.addStretch()
        
        layout.addLayout(convert_layout)
        
        # Results table
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.vc_results_table = QTableWidget()
        self.vc_results_table.setColumnCount(4)
        self.vc_results_table.setHorizontalHeaderLabels(["Input File", "Output File", "Status", "Actions"])
        self.vc_results_table.horizontalHeader().setStretchLastSection(True)
        self.vc_results_table.setAlternatingRowColors(True)
        
        results_layout.addWidget(self.vc_results_table)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.vc_target_path = None
        self.vc_worker = None
        self.vc_record_btn = target_record_btn
        
        return widget
        
    def apply_theme(self):
        """Apply a modern dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTextEdit, QLineEdit, QSpinBox, QListWidget, QTableWidget {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 15px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #666;
            }
            QLabel {
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #3d3d3d;
                color: #ffffff;
                padding: 8px 20px;
                margin-right: 5px;
            }
            QTabBar::tab:selected {
                background-color: #4d4d4d;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 6px;
                background: #2d2d2d;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #3d9d3f;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #4CAF50;
                border-radius: 3px;
            }
            QTableWidget {
                gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #3d3d3d;
                padding: 5px;
                border: 1px solid #444;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #2d2d2d;
                border: 2px solid #666;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #4CAF50;
                border-radius: 3px;
            }
        """)
        
    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(f"Output: {dir_path}")
            
    def load_tts_template(self, template_type):
        templates = {
            "story": [
                "Once upon a time, in a land far away, there lived a brave knight.",
                "The knight embarked on a quest to save the kingdom from a terrible dragon.",
                "Along the way, he met a wise wizard who gave him magical advice.",
                "With courage in his heart, the knight faced the dragon in an epic battle.",
                "The kingdom was saved, and the knight became a legend."
            ],
            "game": [
                "Welcome, hero! Your journey begins here.",
                "You've gained a new ability: Fire Strike!",
                "Warning! Enemy approaching from the north.",
                "Quest completed! You've earned 500 gold.",
                "Game over. Would you like to try again?"
            ],
            "announcement": [
                "Attention all passengers: The train will depart in 5 minutes.",
                "This is your captain speaking. We're expecting clear skies ahead.",
                "Welcome to our store! Today's special is 50% off all items.",
                "The meeting will begin in conference room A at 2 PM.",
                "Thank you for visiting. We hope to see you again soon!"
            ]
        }
        
        if template_type in templates:
            self.tts_text_input.setPlainText("\n".join(templates[template_type]))
            
    def load_tts_reference(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Reference Audio", "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if file_path:
            self.tts_ref_path = file_path
            self.tts_ref_label.setText(Path(file_path).name)
            
    def toggle_tts_recording(self):
        btn = self.sender()
        if btn.text() == "Record":
            self.recorder.start_recording()
            btn.setText("Stop")
            btn.setStyleSheet("QPushButton { background-color: #ff4444; }")
        else:
            temp_path = os.path.join(self.output_dir, "temp_tts_ref.wav")
            if self.recorder.save_recording(temp_path):
                self.tts_ref_path = temp_path
                self.tts_ref_label.setText("Recording")
            btn.setText("Record")
            btn.setStyleSheet("")
            
    def clear_tts_reference(self):
        self.tts_ref_path = None
        self.tts_ref_label.setText("None")
        
    def generate_bulk_tts(self):
        # Get texts to generate
        if self.tts_multiline_radio.isChecked():
            text = self.tts_text_input.toPlainText()
            texts = [line.strip() for line in text.split('\n') if line.strip()]
        else:
            texts = [self.tts_text_input.toPlainText().strip()]
            
        if not texts:
            QMessageBox.warning(self, "Warning", "Please enter some text to generate.")
            return
            
        # Load model if needed
        if self.tts_model is None:
            self.status_label.setText("Loading TTS model...")
            QApplication.processEvents()
            try:
                self.tts_model = ChatterboxTTS.from_pretrained(self.device)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                return
                
        # Clear results table
        self.tts_results_table.setRowCount(0)
        
        # Get parameters
        params = self.tts_params.get_params()
        params['ref_audio'] = self.tts_ref_path
        
        # Create worker
        base_filename = self.tts_filename_input.text() or "output"
        self.tts_worker = BulkTTSWorker(
            self.tts_model,
            texts,
            self.output_dir,
            base_filename,
            params
        )
        
        # Connect signals
        self.tts_worker.progress.connect(self.update_tts_progress)
        self.tts_worker.file_completed.connect(self.on_tts_file_completed)
        self.tts_worker.error.connect(self.on_tts_error)
        self.tts_worker.finished.connect(self.on_tts_finished)
        
        # Update UI
        self.tts_generate_btn.setEnabled(False)
        self.tts_stop_btn.setEnabled(True)
        self.status_progress.setVisible(True)
        self.status_progress.setMaximum(len(texts))
        
        # Start generation
        self.tts_worker.start()
        
    def stop_tts_generation(self):
        if self.tts_worker:
            self.tts_worker.stop()
            self.status_label.setText("Stopping...")
            
    def update_tts_progress(self, current, total, message):
        self.status_label.setText(f"Processing {current}/{total}: {message}")
        self.status_progress.setValue(current)
        
    def on_tts_file_completed(self, text, output_path):
        row = self.tts_results_table.rowCount()
        self.tts_results_table.insertRow(row)
        
        # Text
        text_item = QTableWidgetItem(text[:50] + "..." if len(text) > 50 else text)
        self.tts_results_table.setItem(row, 0, text_item)
        
        # Output file
        file_item = QTableWidgetItem(Path(output_path).name)
        self.tts_results_table.setItem(row, 1, file_item)
        
        # Status
        status_item = QTableWidgetItem("✓ Complete")
        status_item.setForeground(QColor("#4CAF50"))
        self.tts_results_table.setItem(row, 2, status_item)
        
        # Actions
        play_btn = QPushButton("Play")
        play_btn.clicked.connect(lambda: self.play_audio(output_path))
        self.tts_results_table.setCellWidget(row, 3, play_btn)
        
    def on_tts_error(self, text, error_msg):
        row = self.tts_results_table.rowCount()
        self.tts_results_table.insertRow(row)
        
        # Text
        text_item = QTableWidgetItem(text[:50] + "..." if len(text) > 50 else text)
        self.tts_results_table.setItem(row, 0, text_item)
        
        # Error in output column
        error_item = QTableWidgetItem("Failed")
        self.tts_results_table.setItem(row, 1, error_item)
        
        # Status
        status_item = QTableWidgetItem(f"✗ {error_msg}")
        status_item.setForeground(QColor("#f44336"))
        self.tts_results_table.setItem(row, 2, status_item)
        
    def on_tts_finished(self):
        self.tts_generate_btn.setEnabled(True)
        self.tts_stop_btn.setEnabled(False)
        self.status_progress.setVisible(False)
        self.status_label.setText("TTS generation complete!")
        
    # Voice Conversion methods
    def browse_vc_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "", "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a)"
        )
        self.add_vc_files(files)
        
    def add_vc_files(self, files):
        for file_path in files:
            if not self.vc_file_list.findItems(file_path, Qt.MatchFlag.MatchExactly):
                self.vc_file_list.addItem(file_path)
        self.update_vc_file_count()
        
    def update_vc_file_count(self):
        count = self.vc_file_list.count()
        self.vc_file_count_label.setText(str(count))
        self.vc_convert_btn.setEnabled(count > 0)
        
    def load_vc_target(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Target Voice", "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if file_path:
            self.vc_target_path = file_path
            self.vc_target_label.setText(Path(file_path).name)
            
    def toggle_vc_recording(self):
        if self.vc_record_btn.text() == "Record":
            self.recorder.start_recording()
            self.vc_record_btn.setText("Stop")
            self.vc_record_btn.setStyleSheet("QPushButton { background-color: #ff4444; }")
        else:
            temp_path = os.path.join(self.output_dir, "temp_vc_target.wav")
            if self.recorder.save_recording(temp_path):
                self.vc_target_path = temp_path
                self.vc_target_label.setText("Recording")
            self.vc_record_btn.setText("Record")
            self.vc_record_btn.setStyleSheet("")
            
    def clear_vc_target(self):
        self.vc_target_path = None
        self.vc_target_label.setText("Using default voice")
        
    def convert_bulk_vc(self):
        # Get files to convert
        input_files = []
        for i in range(self.vc_file_list.count()):
            input_files.append(self.vc_file_list.item(i).text())
            
        if not input_files:
            QMessageBox.warning(self, "Warning", "Please add audio files to convert.")
            return
            
        # Load model if needed
        if self.vc_model is None:
            self.status_label.setText("Loading VC model...")
            QApplication.processEvents()
            try:
                self.vc_model = ChatterboxVC.from_pretrained(self.device)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                return
                
        # Clear results table
        self.vc_results_table.setRowCount(0)
        
        # Create worker
        self.vc_worker = BulkVCWorker(
            self.vc_model,
            input_files,
            self.output_dir,
            self.vc_target_path
        )
        
        # Connect signals
        self.vc_worker.progress.connect(self.update_vc_progress)
        self.vc_worker.file_completed.connect(self.on_vc_file_completed)
        self.vc_worker.error.connect(self.on_vc_error)
        self.vc_worker.finished.connect(self.on_vc_finished)
        
        # Update UI
        self.vc_convert_btn.setEnabled(False)
        self.vc_stop_btn.setEnabled(True)
        self.status_progress.setVisible(True)
        self.status_progress.setMaximum(len(input_files))
        
        # Start conversion
        self.vc_worker.start()
        
    def stop_vc_conversion(self):
        if self.vc_worker:
            self.vc_worker.stop()
            self.status_label.setText("Stopping...")
            
    def update_vc_progress(self, current, total, message):
        self.status_label.setText(f"Converting {current}/{total}: {message}")
        self.status_progress.setValue(current)
        
    def on_vc_file_completed(self, input_path, output_path):
        row = self.vc_results_table.rowCount()
        self.vc_results_table.insertRow(row)
        
        # Input file
        input_item = QTableWidgetItem(Path(input_path).name)
        self.vc_results_table.setItem(row, 0, input_item)
        
        # Output file
        output_item = QTableWidgetItem(Path(output_path).name)
        self.vc_results_table.setItem(row, 1, output_item)
        
        # Status
        status_item = QTableWidgetItem("✓ Complete")
        status_item.setForeground(QColor("#4CAF50"))
        self.vc_results_table.setItem(row, 2, status_item)
        
        # Actions
        actions_widget = QWidget()
        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        
        play_original_btn = QPushButton("Original")
        play_original_btn.clicked.connect(lambda: self.play_audio(input_path))
        play_converted_btn = QPushButton("Converted")
        play_converted_btn.clicked.connect(lambda: self.play_audio(output_path))
        
        actions_layout.addWidget(play_original_btn)
        actions_layout.addWidget(play_converted_btn)
        actions_widget.setLayout(actions_layout)
        
        self.vc_results_table.setCellWidget(row, 3, actions_widget)
        
    def on_vc_error(self, input_path, error_msg):
        row = self.vc_results_table.rowCount()
        self.vc_results_table.insertRow(row)
        
        # Input file
        input_item = QTableWidgetItem(Path(input_path).name)
        self.vc_results_table.setItem(row, 0, input_item)
        
        # Error in output column
        error_item = QTableWidgetItem("Failed")
        self.vc_results_table.setItem(row, 1, error_item)
        
        # Status
        status_item = QTableWidgetItem(f"✗ {error_msg}")
        status_item.setForeground(QColor("#f44336"))
        self.vc_results_table.setItem(row, 2, status_item)
        
    def on_vc_finished(self):
        self.vc_convert_btn.setEnabled(True)
        self.vc_stop_btn.setEnabled(False)
        self.status_progress.setVisible(False)
        self.status_label.setText("Voice conversion complete!")
        self.update_vc_file_count()
        
    def play_audio(self, file_path):
        if os.path.exists(file_path):
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.media_player.play()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = BulkChatterBoxGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
