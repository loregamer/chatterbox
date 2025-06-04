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
    QMessageBox, QDoubleSpinBox, QCheckBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon
import sounddevice as sd
import soundfile as sf
import queue
import threading
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


class TTSWorker(QThread):
    """Worker thread for TTS generation"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, model, text, audio_prompt_path, exaggeration, temperature, seed, cfg_weight):
        super().__init__()
        self.model = model
        self.text = text
        self.audio_prompt_path = audio_prompt_path
        self.exaggeration = exaggeration
        self.temperature = temperature
        self.seed = seed
        self.cfg_weight = cfg_weight
        
    def run(self):
        try:
            self.progress.emit("Setting random seed...")
            if self.seed != 0:
                self.set_seed(int(self.seed))
                
            self.progress.emit("Generating audio...")
            wav = self.model.generate(
                self.text,
                audio_prompt_path=self.audio_prompt_path if self.audio_prompt_path else None,
                exaggeration=self.exaggeration,
                temperature=self.temperature,
                cfg_weight=self.cfg_weight,
            )
            
            # Save to temporary file
            output_path = "temp_tts_output.wav"
            ta.save(output_path, wav, self.model.sr)
            
            self.finished.emit(output_path)
            
        except Exception as e:
            self.error.emit(str(e))
            
    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)


class VCWorker(QThread):
    """Worker thread for Voice Conversion"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, model, audio_path, target_voice_path):
        super().__init__()
        self.model = model
        self.audio_path = audio_path
        self.target_voice_path = target_voice_path
        
    def run(self):
        try:
            self.progress.emit("Processing voice conversion...")
            wav = self.model.generate(
                audio=self.audio_path,
                target_voice_path=self.target_voice_path if self.target_voice_path else None,
            )
            
            # Save to temporary file
            output_path = "temp_vc_output.wav"
            ta.save(output_path, wav, self.model.sr)
            
            self.finished.emit(output_path)
            
        except Exception as e:
            self.error.emit(str(e))


class ChatterBoxGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatterBox - TTS & Voice Conversion")
        self.setGeometry(100, 100, 1200, 800)
        
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
        
        # Setup UI
        self.init_ui()
        self.apply_theme()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("ChatterBox")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        device_label = QLabel(f"Device: {self.device.upper()}")
        device_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
        header_layout.addStretch()
        header_layout.addWidget(device_label)
        
        main_layout.addLayout(header_layout)
        
        # Tab Widget
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.create_tts_tab(), "Text-to-Speech")
        self.tab_widget.addTab(self.create_vc_tab(), "Voice Conversion")
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { padding: 5px; }")
        main_layout.addWidget(self.status_label)
        
    def create_tts_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Input controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Text input
        text_group = QGroupBox("Text Input")
        text_layout = QVBoxLayout()
        self.tts_text_input = QTextEdit()
        self.tts_text_input.setPlainText(
            "Now let's make my mum's favourite. So three mars bars into the pan. "
            "Then we add the tuna and just stir for a bit, just let the chocolate "
            "and fish infuse. A sprinkle of olive oil and some tomato ketchup. "
            "Now smell that. Oh boy this is going to be incredible."
        )
        self.tts_text_input.setMaximumHeight(150)
        
        char_count_label = QLabel("0/300 characters")
        self.tts_char_count = char_count_label
        self.tts_text_input.textChanged.connect(self.update_char_count)
        
        text_layout.addWidget(self.tts_text_input)
        text_layout.addWidget(char_count_label)
        text_group.setLayout(text_layout)
        left_layout.addWidget(text_group)
        
        # Reference audio
        ref_group = QGroupBox("Reference Voice (Optional)")
        ref_layout = QVBoxLayout()
        
        ref_buttons = QHBoxLayout()
        self.tts_load_ref_btn = QPushButton("Load Audio")
        self.tts_load_ref_btn.clicked.connect(self.load_tts_reference)
        self.tts_record_ref_btn = QPushButton("Record")
        self.tts_record_ref_btn.clicked.connect(self.toggle_tts_recording)
        self.tts_clear_ref_btn = QPushButton("Clear")
        self.tts_clear_ref_btn.clicked.connect(self.clear_tts_reference)
        
        ref_buttons.addWidget(self.tts_load_ref_btn)
        ref_buttons.addWidget(self.tts_record_ref_btn)
        ref_buttons.addWidget(self.tts_clear_ref_btn)
        
        self.tts_ref_label = QLabel("No reference audio loaded")
        ref_layout.addLayout(ref_buttons)
        ref_layout.addWidget(self.tts_ref_label)
        ref_group.setLayout(ref_layout)
        left_layout.addWidget(ref_group)
        
        # Parameters
        params_group = QGroupBox("Generation Parameters")
        params_layout = QVBoxLayout()
        
        # Exaggeration Slider
        exag_layout = QHBoxLayout()
        exag_label = QLabel("Exaggeration:")
        exag_label.setMinimumWidth(100)
        exag_layout.addWidget(exag_label)
        
        self.tts_exaggeration_slider = QSlider(Qt.Orientation.Horizontal)
        self.tts_exaggeration_slider.setRange(25, 200)  # 0.25 to 2.0, scaled by 100
        self.tts_exaggeration_slider.setValue(50)  # 0.5
        self.tts_exaggeration_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tts_exaggeration_slider.setTickInterval(25)
        self.tts_exaggeration_slider.setToolTip("Controls the emotional intensity of the voice. Higher values make the voice more expressive.")
        
        self.tts_exaggeration_value = QLabel("0.50")
        self.tts_exaggeration_value.setMinimumWidth(50)
        self.tts_exaggeration_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.tts_exaggeration_slider.valueChanged.connect(
            lambda v: self.tts_exaggeration_value.setText(f"{v/100:.2f}")
        )
        
        exag_layout.addWidget(self.tts_exaggeration_slider)
        exag_layout.addWidget(self.tts_exaggeration_value)
        
        exag_hint = QLabel("(Neutral = 0.5, extreme values can be unstable)")
        exag_hint.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        exag_range = QLabel("Range: 0.25 - 2.0")
        exag_range.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        
        params_layout.addLayout(exag_layout)
        params_layout.addWidget(exag_hint)
        params_layout.addWidget(exag_range)
        params_layout.addSpacing(10)
        
        # CFG Weight Slider
        cfg_layout = QHBoxLayout()
        cfg_label = QLabel("CFG/Pace:")
        cfg_label.setMinimumWidth(100)
        cfg_layout.addWidget(cfg_label)
        
        self.tts_cfg_slider = QSlider(Qt.Orientation.Horizontal)
        self.tts_cfg_slider.setRange(0, 100)  # 0.0 to 1.0, scaled by 100
        self.tts_cfg_slider.setValue(50)  # 0.5
        self.tts_cfg_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tts_cfg_slider.setTickInterval(25)
        self.tts_cfg_slider.setToolTip("Controls the pacing and rhythm of speech. Higher values may produce more measured speech.")
        
        self.tts_cfg_value = QLabel("0.50")
        self.tts_cfg_value.setMinimumWidth(50)
        self.tts_cfg_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.tts_cfg_slider.valueChanged.connect(
            lambda v: self.tts_cfg_value.setText(f"{v/100:.2f}")
        )
        
        cfg_layout.addWidget(self.tts_cfg_slider)
        cfg_layout.addWidget(self.tts_cfg_value)
        
        cfg_range = QLabel("Range: 0.0 - 1.0")
        cfg_range.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        
        params_layout.addLayout(cfg_layout)
        params_layout.addWidget(cfg_range)
        params_layout.addSpacing(10)
        
        # Temperature Slider
        temp_layout = QHBoxLayout()
        temp_label = QLabel("Temperature:")
        temp_label.setMinimumWidth(100)
        temp_layout.addWidget(temp_label)
        
        self.tts_temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.tts_temperature_slider.setRange(5, 500)  # 0.05 to 5.0, scaled by 100
        self.tts_temperature_slider.setValue(80)  # 0.8
        self.tts_temperature_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tts_temperature_slider.setTickInterval(100)
        self.tts_temperature_slider.setToolTip("Controls randomness in generation. Lower values are more conservative, higher values more varied.")
        
        self.tts_temperature_value = QLabel("0.80")
        self.tts_temperature_value.setMinimumWidth(50)
        self.tts_temperature_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.tts_temperature_slider.valueChanged.connect(
            lambda v: self.tts_temperature_value.setText(f"{v/100:.2f}")
        )
        
        temp_layout.addWidget(self.tts_temperature_slider)
        temp_layout.addWidget(self.tts_temperature_value)
        
        temp_range = QLabel("Range: 0.05 - 5.0")
        temp_range.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        
        params_layout.addLayout(temp_layout)
        params_layout.addWidget(temp_range)
        params_layout.addSpacing(10)
        
        # Seed (keep as spinbox)
        seed_layout = QHBoxLayout()
        seed_label = QLabel("Seed:")
        seed_label.setMinimumWidth(100)
        seed_layout.addWidget(seed_label)
        
        self.tts_seed = QSpinBox()
        self.tts_seed.setRange(0, 999999)
        self.tts_seed.setValue(0)
        self.tts_seed.setSpecialValueText("Random")
        self.tts_seed.setMinimumWidth(100)
        self.tts_seed.setToolTip("Set a specific seed for reproducible results, or leave at 0 for random generation.")
        
        seed_hint = QLabel("(0 = random seed)")
        seed_hint.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        
        seed_layout.addWidget(self.tts_seed)
        seed_layout.addWidget(seed_hint)
        seed_layout.addStretch()
        
        params_layout.addLayout(seed_layout)
        
        # Reset to defaults button
        params_layout.addSpacing(15)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_tts_params)
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        params_layout.addWidget(reset_btn)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Generate button
        self.tts_generate_btn = QPushButton("Generate Speech")
        self.tts_generate_btn.clicked.connect(self.generate_tts)
        self.tts_generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        left_layout.addWidget(self.tts_generate_btn)
        
        left_layout.addStretch()
        
        # Right panel - Output
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        # Progress
        self.tts_progress = QProgressBar()
        self.tts_progress.setVisible(False)
        output_layout.addWidget(self.tts_progress)
        
        # Audio player controls
        player_controls = QHBoxLayout()
        self.tts_play_btn = QPushButton("Play")
        self.tts_play_btn.clicked.connect(self.play_tts_audio)
        self.tts_play_btn.setEnabled(False)
        
        self.tts_stop_btn = QPushButton("Stop")
        self.tts_stop_btn.clicked.connect(self.stop_audio)
        self.tts_stop_btn.setEnabled(False)
        
        self.tts_save_btn = QPushButton("Save As...")
        self.tts_save_btn.clicked.connect(self.save_tts_audio)
        self.tts_save_btn.setEnabled(False)
        
        player_controls.addWidget(self.tts_play_btn)
        player_controls.addWidget(self.tts_stop_btn)
        player_controls.addWidget(self.tts_save_btn)
        player_controls.addStretch()
        
        output_layout.addLayout(player_controls)
        
        # Output info
        self.tts_output_info = QLabel("No audio generated yet")
        output_layout.addWidget(self.tts_output_info)
        
        output_group.setLayout(output_layout)
        right_layout.addWidget(output_group)
        right_layout.addStretch()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        self.tts_ref_path = None
        self.tts_output_path = None
        
        return widget
        
    def create_vc_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Input controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Source audio
        source_group = QGroupBox("Source Audio")
        source_layout = QVBoxLayout()
        
        source_buttons = QHBoxLayout()
        self.vc_load_source_btn = QPushButton("Load Audio")
        self.vc_load_source_btn.clicked.connect(self.load_vc_source)
        self.vc_record_source_btn = QPushButton("Record")
        self.vc_record_source_btn.clicked.connect(self.toggle_vc_source_recording)
        
        source_buttons.addWidget(self.vc_load_source_btn)
        source_buttons.addWidget(self.vc_record_source_btn)
        
        self.vc_source_label = QLabel("No source audio loaded")
        source_layout.addLayout(source_buttons)
        source_layout.addWidget(self.vc_source_label)
        source_group.setLayout(source_layout)
        left_layout.addWidget(source_group)
        
        # Target voice
        target_group = QGroupBox("Target Voice")
        target_layout = QVBoxLayout()
        
        target_buttons = QHBoxLayout()
        self.vc_load_target_btn = QPushButton("Load Audio")
        self.vc_load_target_btn.clicked.connect(self.load_vc_target)
        self.vc_record_target_btn = QPushButton("Record")
        self.vc_record_target_btn.clicked.connect(self.toggle_vc_target_recording)
        self.vc_clear_target_btn = QPushButton("Clear")
        self.vc_clear_target_btn.clicked.connect(self.clear_vc_target)
        
        target_buttons.addWidget(self.vc_load_target_btn)
        target_buttons.addWidget(self.vc_record_target_btn)
        target_buttons.addWidget(self.vc_clear_target_btn)
        
        self.vc_target_label = QLabel("No target voice loaded (will use default)")
        target_layout.addLayout(target_buttons)
        target_layout.addWidget(self.vc_target_label)
        target_group.setLayout(target_layout)
        left_layout.addWidget(target_group)
        
        # Convert button
        self.vc_convert_btn = QPushButton("Convert Voice")
        self.vc_convert_btn.clicked.connect(self.convert_voice)
        self.vc_convert_btn.setEnabled(False)
        self.vc_convert_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        left_layout.addWidget(self.vc_convert_btn)
        
        left_layout.addStretch()
        
        # Right panel - Output
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        # Progress
        self.vc_progress = QProgressBar()
        self.vc_progress.setVisible(False)
        output_layout.addWidget(self.vc_progress)
        
        # Audio player controls
        player_controls = QHBoxLayout()
        self.vc_play_btn = QPushButton("Play")
        self.vc_play_btn.clicked.connect(self.play_vc_audio)
        self.vc_play_btn.setEnabled(False)
        
        self.vc_stop_btn = QPushButton("Stop")
        self.vc_stop_btn.clicked.connect(self.stop_audio)
        self.vc_stop_btn.setEnabled(False)
        
        self.vc_save_btn = QPushButton("Save As...")
        self.vc_save_btn.clicked.connect(self.save_vc_audio)
        self.vc_save_btn.setEnabled(False)
        
        player_controls.addWidget(self.vc_play_btn)
        player_controls.addWidget(self.vc_stop_btn)
        player_controls.addWidget(self.vc_save_btn)
        player_controls.addStretch()
        
        output_layout.addLayout(player_controls)
        
        # Output info
        self.vc_output_info = QLabel("No audio converted yet")
        output_layout.addWidget(self.vc_output_info)
        
        output_group.setLayout(output_layout)
        right_layout.addWidget(output_group)
        right_layout.addStretch()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        self.vc_source_path = None
        self.vc_target_path = None
        self.vc_output_path = None
        
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
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTextEdit, QSpinBox, QDoubleSpinBox {
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
            QSlider {
                margin: 5px 0;
            }
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 8px;
                background: #2d2d2d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #3d9d3f;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #5cbf60;
            }
            QSlider::sub-page:horizontal {
                background: #4CAF50;
                border: 1px solid #3d9d3f;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::tick-mark {
                background: #666;
                width: 1px;
                height: 5px;
            }
        """)
        
    def update_char_count(self):
        text = self.tts_text_input.toPlainText()
        count = len(text)
        self.tts_char_count.setText(f"{count}/300 characters")
        if count > 300:
            self.tts_char_count.setStyleSheet("QLabel { color: #ff4444; }")
        else:
            self.tts_char_count.setStyleSheet("QLabel { color: #ffffff; }")
            
    def reset_tts_params(self):
        """Reset all TTS parameters to default values"""
        self.tts_exaggeration_slider.setValue(50)  # 0.5
        self.tts_cfg_slider.setValue(50)  # 0.5
        self.tts_temperature_slider.setValue(80)  # 0.8
        self.tts_seed.setValue(0)  # Random
            
    def load_tts_reference(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Reference Audio", "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if file_path:
            self.tts_ref_path = file_path
            self.tts_ref_label.setText(f"Loaded: {Path(file_path).name}")
            
    def clear_tts_reference(self):
        self.tts_ref_path = None
        self.tts_ref_label.setText("No reference audio loaded")
        
    def toggle_tts_recording(self):
        if self.tts_record_ref_btn.text() == "Record":
            self.recorder.start_recording()
            self.tts_record_ref_btn.setText("Stop Recording")
            self.tts_record_ref_btn.setStyleSheet("QPushButton { background-color: #ff4444; }")
        else:
            temp_path = "temp_tts_ref_recording.wav"
            if self.recorder.save_recording(temp_path):
                self.tts_ref_path = temp_path
                self.tts_ref_label.setText("Loaded: Recording")
            self.tts_record_ref_btn.setText("Record")
            self.tts_record_ref_btn.setStyleSheet("")
            
    def generate_tts(self):
        text = self.tts_text_input.toPlainText()
        if not text:
            QMessageBox.warning(self, "Warning", "Please enter some text to synthesize.")
            return
            
        if len(text) > 300:
            QMessageBox.warning(self, "Warning", "Text is too long. Maximum 300 characters.")
            return
            
        # Load model if not already loaded
        if self.tts_model is None:
            self.status_label.setText("Loading TTS model...")
            QApplication.processEvents()
            try:
                self.tts_model = ChatterboxTTS.from_pretrained(self.device)
                self.status_label.setText("Model loaded successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load TTS model: {str(e)}")
                self.status_label.setText("Failed to load model")
                return
                
        # Create worker thread
        self.tts_worker = TTSWorker(
            self.tts_model,
            text,
            self.tts_ref_path,
            self.tts_exaggeration_slider.value() / 100,  # Convert from slider scale
            self.tts_temperature_slider.value() / 100,   # Convert from slider scale
            self.tts_seed.value(),
            self.tts_cfg_slider.value() / 100            # Convert from slider scale
        )
        
        self.tts_worker.progress.connect(self.update_tts_progress)
        self.tts_worker.finished.connect(self.on_tts_finished)
        self.tts_worker.error.connect(self.on_tts_error)
        
        # Update UI
        self.tts_generate_btn.setEnabled(False)
        self.tts_progress.setVisible(True)
        self.tts_progress.setRange(0, 0)  # Indeterminate progress
        
        self.tts_worker.start()
        
    def update_tts_progress(self, message):
        self.status_label.setText(message)
        
    def on_tts_finished(self, output_path):
        self.tts_output_path = output_path
        self.tts_output_info.setText(f"Generated: {Path(output_path).name}")
        self.tts_play_btn.setEnabled(True)
        self.tts_save_btn.setEnabled(True)
        self.tts_generate_btn.setEnabled(True)
        self.tts_progress.setVisible(False)
        self.status_label.setText("TTS generation completed")
        
    def on_tts_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"TTS generation failed: {error_msg}")
        self.tts_generate_btn.setEnabled(True)
        self.tts_progress.setVisible(False)
        self.status_label.setText("TTS generation failed")
        
    def play_tts_audio(self):
        if self.tts_output_path and os.path.exists(self.tts_output_path):
            self.media_player.setSource(QUrl.fromLocalFile(self.tts_output_path))
            self.media_player.play()
            self.tts_play_btn.setEnabled(False)
            self.tts_stop_btn.setEnabled(True)
            self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)
            
    def save_tts_audio(self):
        if self.tts_output_path and os.path.exists(self.tts_output_path):
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Audio", "output.wav", "WAV Files (*.wav)"
            )
            if save_path:
                import shutil
                shutil.copy(self.tts_output_path, save_path)
                QMessageBox.information(self, "Success", f"Audio saved to {save_path}")
                
    # Voice Conversion methods
    def load_vc_source(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Source Audio", "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if file_path:
            self.vc_source_path = file_path
            self.vc_source_label.setText(f"Loaded: {Path(file_path).name}")
            self.vc_convert_btn.setEnabled(True)
            
    def toggle_vc_source_recording(self):
        if self.vc_record_source_btn.text() == "Record":
            self.recorder.start_recording()
            self.vc_record_source_btn.setText("Stop Recording")
            self.vc_record_source_btn.setStyleSheet("QPushButton { background-color: #ff4444; }")
        else:
            temp_path = "temp_vc_source_recording.wav"
            if self.recorder.save_recording(temp_path):
                self.vc_source_path = temp_path
                self.vc_source_label.setText("Loaded: Recording")
                self.vc_convert_btn.setEnabled(True)
            self.vc_record_source_btn.setText("Record")
            self.vc_record_source_btn.setStyleSheet("")
            
    def load_vc_target(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Target Voice", "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if file_path:
            self.vc_target_path = file_path
            self.vc_target_label.setText(f"Loaded: {Path(file_path).name}")
            
    def toggle_vc_target_recording(self):
        if self.vc_record_target_btn.text() == "Record":
            self.recorder.start_recording()
            self.vc_record_target_btn.setText("Stop Recording")
            self.vc_record_target_btn.setStyleSheet("QPushButton { background-color: #ff4444; }")
        else:
            temp_path = "temp_vc_target_recording.wav"
            if self.recorder.save_recording(temp_path):
                self.vc_target_path = temp_path
                self.vc_target_label.setText("Loaded: Recording")
            self.vc_record_target_btn.setText("Record")
            self.vc_record_target_btn.setStyleSheet("")
            
    def clear_vc_target(self):
        self.vc_target_path = None
        self.vc_target_label.setText("No target voice loaded (will use default)")
        
    def convert_voice(self):
        if not self.vc_source_path:
            QMessageBox.warning(self, "Warning", "Please load source audio first.")
            return
            
        # Load model if not already loaded
        if self.vc_model is None:
            self.status_label.setText("Loading VC model...")
            QApplication.processEvents()
            try:
                self.vc_model = ChatterboxVC.from_pretrained(self.device)
                self.status_label.setText("Model loaded successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load VC model: {str(e)}")
                self.status_label.setText("Failed to load model")
                return
                
        # Create worker thread
        self.vc_worker = VCWorker(
            self.vc_model,
            self.vc_source_path,
            self.vc_target_path
        )
        
        self.vc_worker.progress.connect(self.update_vc_progress)
        self.vc_worker.finished.connect(self.on_vc_finished)
        self.vc_worker.error.connect(self.on_vc_error)
        
        # Update UI
        self.vc_convert_btn.setEnabled(False)
        self.vc_progress.setVisible(True)
        self.vc_progress.setRange(0, 0)
        
        self.vc_worker.start()
        
    def update_vc_progress(self, message):
        self.status_label.setText(message)
        
    def on_vc_finished(self, output_path):
        self.vc_output_path = output_path
        self.vc_output_info.setText(f"Converted: {Path(output_path).name}")
        self.vc_play_btn.setEnabled(True)
        self.vc_save_btn.setEnabled(True)
        self.vc_convert_btn.setEnabled(True)
        self.vc_progress.setVisible(False)
        self.status_label.setText("Voice conversion completed")
        
    def on_vc_error(self, error_msg):
        QMessageBox.critical(self, "Error", f"Voice conversion failed: {error_msg}")
        self.vc_convert_btn.setEnabled(True)
        self.vc_progress.setVisible(False)
        self.status_label.setText("Voice conversion failed")
        
    def play_vc_audio(self):
        if self.vc_output_path and os.path.exists(self.vc_output_path):
            self.media_player.setSource(QUrl.fromLocalFile(self.vc_output_path))
            self.media_player.play()
            self.vc_play_btn.setEnabled(False)
            self.vc_stop_btn.setEnabled(True)
            self.media_player.mediaStatusChanged.connect(self.on_media_status_changed)
            
    def save_vc_audio(self):
        if self.vc_output_path and os.path.exists(self.vc_output_path):
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Audio", "converted_output.wav", "WAV Files (*.wav)"
            )
            if save_path:
                import shutil
                shutil.copy(self.vc_output_path, save_path)
                QMessageBox.information(self, "Success", f"Audio saved to {save_path}")
                
    def stop_audio(self):
        self.media_player.stop()
        self.tts_play_btn.setEnabled(True)
        self.tts_stop_btn.setEnabled(False)
        self.vc_play_btn.setEnabled(True)
        self.vc_stop_btn.setEnabled(False)
        
    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.tts_play_btn.setEnabled(True)
            self.tts_stop_btn.setEnabled(False)
            self.vc_play_btn.setEnabled(True)
            self.vc_stop_btn.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = ChatterBoxGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()