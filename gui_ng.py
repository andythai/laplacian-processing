"""
The framework for visualizing Neuroglancer URLs for the U01 workflow GUI.
Andy Thai
andy.thai@uci.edu
"""

# Import standard libraries
import os
import sys
import glob
from natsort import natsorted
from threading import Thread
import copy
import webbrowser

# Import third-party libraries
import numpy as np
import nibabel as nib
import neuroglancer

# Import PyQt5 libraries
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QFrame, \
                            QLabel, QFileDialog, QSlider, QTabWidget, QDialog, \
                            QCheckBox, QPushButton, QRadioButton, QButtonGroup, QComboBox, QLineEdit, \
                            QProgressBar, QSpinBox, QDoubleSpinBox, QSizePolicy, QMessageBox, QGridLayout
from PyQt5.QtGui import QPixmap, QIcon, QDoubleValidator
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# Import our libraries
import run_ng

# Global formatting variables
BOLD_STYLE = "font-weight: bold; color: black"
TITLE_SPACING = " " * 12
MAX_INT = 2147483647


# Custom QDoubleSpinBox class
class CustomDoubleSpinBox(QDoubleSpinBox):
    def textFromValue(self, value):
        text = super().textFromValue(value)
        if '.' in text:
            text = text.rstrip('0').rstrip('.')
        return text


class NeuroglancerTab(QWidget):
    def __init__(self, app):
        super().__init__()
        
        # Set up tab settings and layout.
        self.app = app
        self.layout = QVBoxLayout()
        
        # Declare variables to keep track of file paths and settings
        self.input0_path = None
        self.input1_path = None
        self.input2_path = None
        self.input3_path = None
        self.input_points = None
        self.template_path = None
        self.matrix = [[None, None, None, None], 
                       [None, None, None, None], 
                       [None, None, None, None]]
        
        ###############################################################################
        ##                                 FILE IO                                   ##
        ###############################################################################
        
        # Title
        file_io_title = "## Visualization File I/O"
        self.file_io_title = QLabel(file_io_title, alignment=Qt.AlignCenter)
        self.file_io_title.setTextFormat(Qt.MarkdownText)
        
        # Button to select folder containing data.
        self.input0_folder_button = QPushButton("1: Select input section directory\n⚠️ NO SECTION DATA LOADED")
        self.input0_folder_button.clicked.connect(self._select_input0_path)
        self.input0_folder_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.input0_folder_button.setStyleSheet(BOLD_STYLE)
        
        self.input1_folder_button = QPushButton("2: Select input section directory\n⚠️ NO SECTION DATA LOADED")
        self.input1_folder_button.clicked.connect(self._select_input1_path)
        self.input1_folder_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.input1_folder_button.setEnabled(False)
        
        self.input2_folder_button = QPushButton("3: Select input section directory\n⚠️ NO SECTION DATA LOADED")
        self.input2_folder_button.clicked.connect(self._select_input2_path)
        self.input2_folder_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.input2_folder_button.setEnabled(False)
        
        self.input3_folder_button = QPushButton("4: Select input section directory\n⚠️ NO SECTION DATA LOADED")
        self.input3_folder_button.clicked.connect(self._select_input3_path)
        self.input3_folder_button.setMinimumSize(400, 50)  # Adjust the size as needed
        self.input3_folder_button.setEnabled(False)
        
        input_folder_desc = "Select the folder directory containing stitched section image data. The folder should contain TIFF images. " + \
                            "Alternatively, a .nii.gz file can be selected for registration."
        self.input_folder_desc = QLabel(input_folder_desc, alignment=Qt.AlignCenter)
        self.input_folder_desc.setWordWrap(True)
        self.input_folder_desc.setTextFormat(Qt.MarkdownText)

        # File type parameter
        filetypes = ["tif", "nii", "OME-Zarr (URL)", "disabled"]
        self.input0_type = QComboBox()
        self.input0_type.addItems(filetypes)
        self.input1_type = QComboBox()
        self.input1_type.addItems(filetypes)
        self.input1_type.setCurrentIndex(3)
        self.input2_type = QComboBox()
        self.input2_type.addItems(filetypes)
        self.input2_type.setCurrentIndex(3)
        self.input3_type = QComboBox()
        self.input3_type.addItems(filetypes)
        self.input3_type.setCurrentIndex(3)
        self.input0_type.currentIndexChanged.connect(self._update_input0_type)
        self.input1_type.currentIndexChanged.connect(self._update_input1_type)
        self.input2_type.currentIndexChanged.connect(self._update_input2_type)
        self.input3_type.currentIndexChanged.connect(self._update_input3_type)
        
        # Layer name parameter
        self.input0_name = QLineEdit()
        self.input0_name.setPlaceholderText("Layer name")
        self.input0_name.setText("Layer 1")
        self.input1_name = QLineEdit()
        self.input1_name.setPlaceholderText("Layer name")
        self.input1_name.setText("Layer 2")
        self.input1_name.setEnabled(False)
        self.input2_name = QLineEdit()
        self.input2_name.setPlaceholderText("Layer name")
        self.input2_name.setText("Layer 3")
        self.input2_name.setEnabled(False)
        self.input3_name = QLineEdit()
        self.input3_name.setPlaceholderText("Layer name")
        self.input3_name.setText("Layer 4")
        self.input3_name.setEnabled(False)

        # Input points button
        self.input_points_button = QPushButton("(Optional) Select input points file\n⚠️ NO CELL POINTS LOADED")
        self.input_points_button.clicked.connect(self._select_input_points)
        self.input_points_button.setMinimumSize(400, 50)  # Adjust the size as needed
        input_points_desc = "Select the input counted cell points file. This should be a .txt file outputted from the cell counting module."
        self.input_points_desc = QLabel(input_points_desc, alignment=Qt.AlignCenter)
        self.input_points_desc.setWordWrap(True)
        self.input_points_desc.setTextFormat(Qt.MarkdownText)
        
        # Button to select atlas to register to.
        self.template_button = QPushButton("(Optional) Select the template file\n⚠️ NO TEMPLATE DATA LOADED")
        self.template_button.clicked.connect(self._select_template_path)
        self.template_button.setMinimumSize(400, 50)  # Adjust the size as needed
        template_desc = "Select the template file containing an atlas. " + \
                        "This normally is a nii.gz file of the 25-micron Allen CCF 2017 annotation template."
        self.template_desc = QLabel(template_desc, alignment=Qt.AlignCenter)
        self.template_desc.setWordWrap(True)
        self.template_desc.setTextFormat(Qt.MarkdownText)
        
        # Divider line
        h_line = QFrame()
        h_line.setFrameShape(QFrame.HLine)
        h_line.setFrameShadow(QFrame.Sunken)
        
        # Client address and port
        self.ip_address_input = QLineEdit()
        self.ip_address_input.setPlaceholderText("0.0.0.0")
        self.ip_address_input.setText("0.0.0.0")
        self.ip_address_input.setMaximumWidth(200)  # Adjust the width as needed
        self.port_input = QSpinBox()
        self.port_input.setRange(0, 65535)  # Set the range as needed
        self.port_input.setValue(8080)  # Set default value
        self.port_input.setMaximumWidth(100)  # Adjust the width as needed
        
        # Create a combo box (dropdown list)
        self.input_dim_str = QLabel("Set the resolution and units of measurements for the **source / input dimensions**.") 
        #self.input_dim_str.setWordWrap(True)
        self.input_dim_str.setTextFormat(Qt.MarkdownText)
        self.output_dim_str = QLabel("Set the resolution and units of measurements for the **output dimensions**.")
        #self.output_dim_str.setWordWrap(True)
        self.output_dim_str.setTextFormat(Qt.MarkdownText)
        measurement_items = ("m", "dm", "cm", "mm", "μm", "nm")
        
        self.x_measure_in = CustomDoubleSpinBox()
        self.y_measure_in = CustomDoubleSpinBox()
        self.z_measure_in = CustomDoubleSpinBox()
        
        self.x_measure_in.setRange(0.0, MAX_INT)  # Set the range as needed
        self.y_measure_in.setRange(0.0, MAX_INT)  # Set the range as needed
        self.z_measure_in.setRange(0.0, MAX_INT)  # Set the range as needed
        self.x_measure_in.setDecimals(8)  # Set the precision as needed
        self.y_measure_in.setDecimals(8)  # Set the precision as needed
        self.z_measure_in.setDecimals(8)  # Set the precision as needed
        self.x_measure_in.setMaximumWidth(90)  # Set the width as needed
        self.y_measure_in.setMaximumWidth(90)  # Set the width as needed
        self.z_measure_in.setMaximumWidth(90)  # Set the width as needed
        self.x_measure_in.setValue(0.00125)  # Set default value
        self.y_measure_in.setValue(0.00125)  # Set default value
        self.z_measure_in.setValue(0.05)  # Set default value
        
        self.x_unit_in = QComboBox()
        self.x_unit_in.addItems(measurement_items)
        self.y_unit_in = QComboBox()
        self.y_unit_in.addItems(measurement_items)
        self.z_unit_in = QComboBox()
        self.z_unit_in.addItems(measurement_items)
        
        self.x_unit_in.setMaximumWidth(40)  # Set the width as needed
        self.y_unit_in.setMaximumWidth(40)  # Set the width as needed
        self.z_unit_in.setMaximumWidth(40)  # Set the width as needed
        
        ### Output dimensions ###
        self.x_measure_out = CustomDoubleSpinBox()
        self.y_measure_out = CustomDoubleSpinBox()
        self.z_measure_out = CustomDoubleSpinBox()
        
        self.x_measure_out.setRange(0.0, MAX_INT)  # Set the range as needed
        self.y_measure_out.setRange(0.0, MAX_INT)  # Set the range as needed
        self.z_measure_out.setRange(0.0, MAX_INT)  # Set the range as needed
        self.x_measure_out.setDecimals(8)  # Set the precision as needed
        self.y_measure_out.setDecimals(8)  # Set the precision as needed
        self.z_measure_out.setDecimals(8)  # Set the precision as needed
        self.x_measure_out.setMaximumWidth(90)  # Set the width as needed
        self.y_measure_out.setMaximumWidth(90)  # Set the width as needed
        self.z_measure_out.setMaximumWidth(90)  # Set the width as needed
        self.x_measure_out.setValue(0.00125)  # Set default value
        self.y_measure_out.setValue(0.00125)  # Set default value
        self.z_measure_out.setValue(0.05)  # Set default value
        
        self.x_unit_out = QComboBox()
        self.x_unit_out.addItems(measurement_items)
        self.y_unit_out = QComboBox()
        self.y_unit_out.addItems(measurement_items)
        self.z_unit_out = QComboBox()
        self.z_unit_out.addItems(measurement_items)
        
        self.x_unit_out.setMaximumWidth(40)  # Set the width as needed
        self.y_unit_out.setMaximumWidth(40)  # Set the width as needed
        self.z_unit_out.setMaximumWidth(40)  # Set the width as needed

        ### 3x4 Grid of matrix value inputs ###
        self.transformation_matrix_str = QLabel("Set the **transformation matrix** to apply to the input data." + \
                                                "\n\nRecommended to set yy to 27 and xx to 24 for 25 micron data.")
        self.transformation_matrix_str.setTextFormat(Qt.MarkdownText)
        self.grid_layout = QGridLayout()
        self.spin_boxes = []
        
        # Titles for columns and rows
        column_titles = ["z", "y", "x", "Translation"]
        row_titles = ["z", "y", "x"]

        # Add column titles
        for j in range(4):
            label = QLabel(column_titles[j])
            label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(label, 0, j + 1)  # Offset by 1 to leave space for row titles

        # 24.6403508772, 27.2375
        default_values = [
            [1, 0, 0, 0],
            [0, 27.2375, 0, 0],
            [0, 0, 24.6403508772, 0],
        ]
        # 3x4 matrix
        for i in range(3):
            # Add row title
            label = QLabel(row_titles[i])
            label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(label, i + 1, 0)  # Offset by 1 to leave space for column titles

            row = []
            for j in range(4):
                spin_box = CustomDoubleSpinBox()
                spin_box.setRange(0.0, MAX_INT)  # Set the range as needed
                spin_box.setDecimals(10)  # Set the precision as needed
                spin_box.setMaximumWidth(90)  # Set the width as needed
                spin_box.setValue(default_values[i][j])  # Set default value
                self.grid_layout.addWidget(spin_box, i + 1, j + 1)  # Offset by 1 to leave space for titles
                row.append(spin_box)
                self.matrix[i][j] = spin_box
            self.spin_boxes.append(row)
        
        ###############################################################################
        ##                          METADATA AND RUN APP                             ##
        ###############################################################################
        
        # Register button
        self.visualize_button = QPushButton("Visualize")
        self.visualize_button.setMinimumSize(100, 50)  # Adjust the size as needed
        self.visualize_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.visualize_button.clicked.connect(self._thread_visualize)
        self.visualize_button.setEnabled(False)  # Initially disabled
        
        ################### SETUP UI LAYOUT ###################
        
        # Input folder
        self.io_layout = QVBoxLayout()
        self.io_layout.addWidget(self.file_io_title, alignment=Qt.AlignCenter)
        
        self.io_profile0_layout = QHBoxLayout()
        self.io_profile0_layout.addWidget(self.input0_folder_button, stretch=1)
        self.io_profile0_layout.addWidget(self.input0_type)
        self.io_profile1_layout = QHBoxLayout()
        self.io_profile1_layout.addWidget(self.input1_folder_button, stretch=1)
        self.io_profile1_layout.addWidget(self.input1_type)
        self.io_profile2_layout = QHBoxLayout()
        self.io_profile2_layout.addWidget(self.input2_folder_button, stretch=1)
        self.io_profile2_layout.addWidget(self.input2_type)
        self.io_profile3_layout = QHBoxLayout()
        self.io_profile3_layout.addWidget(self.input3_folder_button, stretch=1)
        self.io_profile3_layout.addWidget(self.input3_type)
        self.io_layout.addLayout(self.io_profile0_layout)
        self.io_layout.addWidget(self.input0_name)
        self.io_layout.addLayout(self.io_profile1_layout)
        self.io_layout.addWidget(self.input1_name)
        self.io_layout.addLayout(self.io_profile2_layout)
        self.io_layout.addWidget(self.input2_name)
        self.io_layout.addLayout(self.io_profile3_layout)
        self.io_layout.addWidget(self.input3_name)
        self.io_layout.addWidget(self.input_folder_desc, alignment=Qt.AlignTop)
        self.io_layout.addWidget(self.input_points_button, stretch=1)
        self.io_layout.addWidget(self.input_points_desc, alignment=Qt.AlignTop)
        self.io_layout.addWidget(self.template_button, stretch=1)
        self.io_layout.addWidget(self.template_desc, alignment=Qt.AlignTop)
        
        ### Parameters ###
        self.io_layout.addWidget(h_line)
        self.io_layout.addWidget(QLabel("Specifications for running Neuroglancer locally."), alignment=Qt.AlignCenter)
        
        # IP address and port
        self.host_layout = QHBoxLayout()
        self.host_layout.addWidget(QLabel("IP address:"))
        self.host_layout.addWidget(self.ip_address_input)
        self.host_layout.addWidget(QLabel("Port:"))
        self.host_layout.addWidget(self.port_input)
        self.io_layout.addLayout(self.host_layout)
        
        # Measurement layout for units and measurement of input data INPUT
        self.measurement_layout_in = QHBoxLayout()
        # x
        x_layout_in = QVBoxLayout()
        x_layout_inner_in = QHBoxLayout()
        x_layout_in.addWidget(QLabel("x"), alignment=Qt.AlignCenter)
        x_layout_inner_in.addWidget(self.x_measure_in)
        x_layout_inner_in.addWidget(self.x_unit_in)
        x_layout_in.addLayout(x_layout_inner_in)
        # y
        y_layout_in = QVBoxLayout()
        y_layout_inner_in = QHBoxLayout()
        y_layout_in.addWidget(QLabel("y"), alignment=Qt.AlignCenter)
        y_layout_inner_in.addWidget(self.y_measure_in)
        y_layout_inner_in.addWidget(self.y_unit_in)
        y_layout_in.addLayout(y_layout_inner_in)
        # z
        z_layout_in = QVBoxLayout()
        z_layout_inner_in = QHBoxLayout()
        z_layout_in.addWidget(QLabel("z"), alignment=Qt.AlignCenter)
        z_layout_inner_in.addWidget(self.z_measure_in)
        z_layout_inner_in.addWidget(self.z_unit_in)
        z_layout_in.addLayout(z_layout_inner_in)

        # Add the vertical layouts to the horizontal layout
        self.measurement_layout_in.addLayout(x_layout_in)
        self.measurement_layout_in.addLayout(y_layout_in)
        self.measurement_layout_in.addLayout(z_layout_in)
        
        # Measurement layout for units and measurement of input data INPUT
        self.measurement_layout_out = QHBoxLayout()
        # x
        x_layout_out = QVBoxLayout()
        x_layout_inner_out = QHBoxLayout()
        x_layout_out.addWidget(QLabel("x"), alignment=Qt.AlignCenter)
        x_layout_inner_out.addWidget(self.x_measure_out)
        x_layout_inner_out.addWidget(self.x_unit_out)
        x_layout_out.addLayout(x_layout_inner_out)
        # y
        y_layout_out = QVBoxLayout()
        y_layout_inner_out = QHBoxLayout()
        y_layout_out.addWidget(QLabel("y"), alignment=Qt.AlignCenter)
        y_layout_inner_out.addWidget(self.y_measure_out)
        y_layout_inner_out.addWidget(self.y_unit_out)
        y_layout_out.addLayout(y_layout_inner_out)
        # z
        z_layout_out = QVBoxLayout()
        z_layout_inner_out = QHBoxLayout()
        z_layout_out.addWidget(QLabel("z"), alignment=Qt.AlignCenter)
        z_layout_inner_out.addWidget(self.z_measure_out)
        z_layout_inner_out.addWidget(self.z_unit_out)
        z_layout_out.addLayout(z_layout_inner_out)

        # Add the vertical layouts to the horizontal layout
        self.measurement_layout_out.addLayout(x_layout_out)
        self.measurement_layout_out.addLayout(y_layout_out)
        self.measurement_layout_out.addLayout(z_layout_out)
        
        self.io_layout.addWidget(self.input_dim_str, alignment=Qt.AlignCenter)
        self.io_layout.addLayout(self.measurement_layout_in)
        self.io_layout.addWidget(self.output_dim_str, alignment=Qt.AlignCenter)
        self.io_layout.addLayout(self.measurement_layout_out)
        
        self.io_layout.addWidget(self.transformation_matrix_str, alignment=Qt.AlignCenter)
        self.io_layout.addLayout(self.grid_layout)

        ### Register button ###
        self.run_buttons_layout = QHBoxLayout()
        self.run_buttons_layout.addWidget(self.visualize_button, stretch=1)
        
        # Layout setup
        self.layout.addLayout(self.io_layout)
        self.layout.addLayout(self.run_buttons_layout)
        
        # End
        self.setLayout(self.layout)
        
        
    def _update_input0_type(self):
        """
        Select the filetype of the input data.
        """
        self.input0_path = None
        self.input0_folder_button.setEnabled(True)
        self.input0_folder_button.setStyleSheet(BOLD_STYLE)
        self.input0_name.setEnabled(True)
        if self.input0_type.currentText() == "tif":
            self.input0_folder_button.setText("1: Select input section directory\n⚠️ NO SECTION DATA LOADED")
        elif self.input0_type.currentText() == "nii":
            self.input0_folder_button.setText("1: Select input nii file\n⚠️ NO NII FILE LOADED")
        elif self.input0_type.currentText() == "OME-Zarr (URL)":
            self.input0_folder_button.setText("1: Not implemented")
        elif self.input0_type.currentText() == "disabled":
            self.input0_folder_button.setEnabled(False)
            self.input0_folder_button.setStyleSheet("")
            self.input0_name.setEnabled(False)
        if not self.input0_path and not self.input1_path and not self.input2_path and not self.input3_path:
            self.visualize_button.setEnabled(False)
            
            
    def _update_input1_type(self):
        """
        Select the filetype of the input data.
        """
        self.input1_path = None
        self.input1_folder_button.setEnabled(True)
        self.input1_folder_button.setStyleSheet(BOLD_STYLE)
        self.input0_name.setEnabled(True)
        if self.input1_type.currentText() == "tif":
            self.input1_folder_button.setText("2: Select input section directory\n⚠️ NO SECTION DATA LOADED")
        elif self.input1_type.currentText() == "nii":
            self.input1_folder_button.setText("2: Select input nii file\n⚠️ NO NII FILE LOADED")
        elif self.input1_type.currentText() == "OME-Zarr (URL)":
            self.input1_folder_button.setText("2: Not implemented")
        elif self.input1_type.currentText() == "disabled":
            self.input1_folder_button.setEnabled(False)
            self.input1_folder_button.setStyleSheet("")
            self.input1_name.setEnabled(False)
        if not self.input0_path and not self.input1_path and not self.input2_path and not self.input3_path:
            self.visualize_button.setEnabled(False)
            
            
    def _update_input2_type(self):
        """
        Select the filetype of the input data.
        """
        self.input2_path = None
        self.input2_folder_button.setEnabled(True)
        self.input2_folder_button.setStyleSheet(BOLD_STYLE)
        self.input2_name.setEnabled(True)
        if self.input2_type.currentText() == "tif":
            self.input2_folder_button.setText("3: Select input section directory\n⚠️ NO SECTION DATA LOADED")
        elif self.input2_type.currentText() == "nii":
            self.input2_folder_button.setText("3: Select input nii file\n⚠️ NO NII FILE LOADED")
        elif self.input2_type.currentText() == "OME-Zarr (URL)":
            self.input2_folder_button.setText("3: Not implemented")
        elif self.input2_type.currentText() == "disabled":
            self.input2_folder_button.setEnabled(False)
            self.input2_folder_button.setStyleSheet("")
            self.input2_name.setEnabled(False)
        if not self.input0_path and not self.input1_path and not self.input2_path and not self.input3_path:
            self.visualize_button.setEnabled(False)
            
            
    def _update_input3_type(self):
        """
        Select the filetype of the input data.
        """
        self.input3_path = None
        self.input3_folder_button.setEnabled(True)
        self.input3_folder_button.setStyleSheet(BOLD_STYLE)
        self.input3_name.setEnabled(True)
        if self.input3_type.currentText() == "tif":
            self.input3_folder_button.setText("4: Select input section directory\n⚠️ NO SECTION DATA LOADED")
        elif self.input3_type.currentText() == "nii":
            self.input3_folder_button.setText("4: Select input nii file\n⚠️ NO NII FILE LOADED")
        elif self.input3_type.currentText() == "OME-Zarr (URL)":
            self.input3_folder_button.setText("4: Not implemented")
        elif self.input3_type.currentText() == "disabled":
            self.input3_folder_button.setEnabled(False)
            self.input3_folder_button.setStyleSheet("")
            self.input3_name.setEnabled(False)
        if not self.input0_path and not self.input1_path and not self.input2_path and not self.input3_path:
            self.visualize_button.setEnabled(False)

        
    def _select_input0_path(self):
        """
        Select the path to load the input tiles from and saves the selected folder path internally.
        """
        if self.input0_type.currentText() == "nii":
            file_path = QFileDialog.getOpenFileName(None, "Select input nii file", filter="NIfTI files (*.nii *.nii.gz)")[0]
        elif self.input0_type.currentText() == "tif": 
            file_path = QFileDialog.getExistingDirectory(None, "Select folder directory with section data")
        if file_path == '':  # If user cancels out of the dialog, exit.
            return
        
        if self.input0_type.currentText() == "nii":
            self.input0_folder_button.setText(f"1: Select input nii file\n✅ {os.path.normpath(file_path)}")
        else:
            # Check if the selected folder contains images.
            section_files = glob.glob(os.path.join(file_path, "*.tif"))
            
            if not section_files:  # If user selects a folder without any images, exit and output an error message.
                error_dialog = QMessageBox()
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setWindowTitle("Error")
                err_msg = "Selected folder does not contain any TIF files. Please check your folder layout and select a folder containing TIF sections."
                error_dialog.setText(err_msg)
                error_dialog.exec_()
                return
            self.input0_folder_button.setText(f"1: Select input section directory\n✅ {os.path.normpath(file_path)}")

        self.input0_path = file_path
        self.input0_folder_button.setStyleSheet("")
        
        # If a valid path is given, save the filepath internally and enable button operations if possible.
        self.visualize_button.setEnabled(True)
        
        # Update button information
        print(f'Selected input0 path: {file_path}')
        
        
    def _select_input1_path(self):
        """
        Select the path to load the input tiles from and saves the selected folder path internally.
        """
        if self.input1_type.currentText() == "nii":
            file_path = QFileDialog.getOpenFileName(None, "Select input nii file", filter="NIfTI files (*.nii *.nii.gz)")[0]
        elif self.input1_type.currentText() == "tif": 
            file_path = QFileDialog.getExistingDirectory(None, "Select folder directory with section data")
        if file_path == '':  # If user cancels out of the dialog, exit.
            return
        
        if self.input1_type.currentText() == "nii":
            self.input1_folder_button.setText(f"2: Select input nii file\n✅ {os.path.normpath(file_path)}")
        else:
            # Check if the selected folder contains images.
            section_files = glob.glob(os.path.join(file_path, "*.tif"))
            
            if not section_files:  # If user selects a folder without any images, exit and output an error message.
                error_dialog = QMessageBox()
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setWindowTitle("Error")
                err_msg = "Selected folder does not contain any TIF files. Please check your folder layout and select a folder containing TIF sections."
                error_dialog.setText(err_msg)
                error_dialog.exec_()
                return
            self.input1_folder_button.setText(f"2: Select input section directory\n✅ {os.path.normpath(file_path)}")

        self.input1_path = file_path
        self.input1_folder_button.setStyleSheet("")
        
        # If a valid path is given, save the filepath internally and enable button operations if possible.
        self.visualize_button.setEnabled(True)
        
        # Update button information
        print(f'Selected input1 path: {file_path}')
        
        
    def _select_input2_path(self):
        """
        Select the path to load the input tiles from and saves the selected folder path internally.
        """
        if self.input2_type.currentText() == "nii":
            file_path = QFileDialog.getOpenFileName(None, "Select input nii file", filter="NIfTI files (*.nii *.nii.gz)")[0]
        elif self.input2_type.currentText() == "tif": 
            file_path = QFileDialog.getExistingDirectory(None, "Select folder directory with section data")
        if file_path == '':  # If user cancels out of the dialog, exit.
            return
        
        if self.input2_type.currentText() == "nii":
            self.input2_folder_button.setText(f"3: Select input nii file\n✅ {os.path.normpath(file_path)}")
        else:
            # Check if the selected folder contains images.
            section_files = glob.glob(os.path.join(file_path, "*.tif"))
            
            if not section_files:  # If user selects a folder without any images, exit and output an error message.
                error_dialog = QMessageBox()
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setWindowTitle("Error")
                err_msg = "Selected folder does not contain any TIF files. Please check your folder layout and select a folder containing TIF sections."
                error_dialog.setText(err_msg)
                error_dialog.exec_()
                return
            self.input2_folder_button.setText(f"3: Select input section directory\n✅ {os.path.normpath(file_path)}")

        self.input2_path = file_path
        self.input2_folder_button.setStyleSheet("")
        
        # If a valid path is given, save the filepath internally and enable button operations if possible.
        self.visualize_button.setEnabled(True)
        
        # Update button information
        print(f'Selected input2 path: {file_path}')
        
        
    def _select_input3_path(self):
        """
        Select the path to load the input tiles from and saves the selected folder path internally.
        """
        if self.input3_type.currentText() == "nii":
            file_path = QFileDialog.getOpenFileName(None, "Select input nii file", filter="NIfTI files (*.nii *.nii.gz)")[0]
        elif self.input3_type.currentText() == "tif": 
            file_path = QFileDialog.getExistingDirectory(None, "Select folder directory with section data")
        if file_path == '':  # If user cancels out of the dialog, exit.
            return
        
        if self.input3_type.currentText() == "nii":
            self.input3_folder_button.setText(f"4: Select input nii file\n✅ {os.path.normpath(file_path)}")
        else:
            # Check if the selected folder contains images.
            section_files = glob.glob(os.path.join(file_path, "*.tif"))
            
            if not section_files:  # If user selects a folder without any images, exit and output an error message.
                error_dialog = QMessageBox()
                error_dialog.setIcon(QMessageBox.Critical)
                error_dialog.setWindowTitle("Error")
                err_msg = "Selected folder does not contain any TIF files. Please check your folder layout and select a folder containing TIF sections."
                error_dialog.setText(err_msg)
                error_dialog.exec_()
                return
            self.input3_folder_button.setText(f"4: Select input section directory\n✅ {os.path.normpath(file_path)}")

        self.input3_path = file_path
        self.input3_folder_button.setStyleSheet("")
        
        # If a valid path is given, save the filepath internally and enable button operations if possible.
        self.visualize_button.setEnabled(True)
        
        # Update button information
        print(f'Selected input3 path: {file_path}')
        
        
    def _update_channel(self):
        """
        Update function that runs when the channel spinbox is updated.
        """
        self.channel = self.channel_spinbox.value()
    
        
    def _select_input_points(self):
        """
        Select the path to load the input points from and saves the selected folder path internally.
        """
        file_path = QFileDialog.getOpenFileName(None, "Select input points file", filter="Text files (*.txt)")[0]
        if file_path == '':
            return

        self.input_points = file_path

        # Update button information
        print(f'Selected input points path: {file_path}')
        
        self.input_points_button.setText(f"Select input points file\n✅ {os.path.normpath(file_path)}")
        self.input_points_button.setStyleSheet("")
        
        
    def _select_template_path(self):
        """
        Select the path to load the input points from and saves the selected folder path internally.
        """
        file_path = QFileDialog.getOpenFileName(None, "Select template file", filter="NIfTI files (*.nii *.nii.gz)")[0]
        if file_path == '':
            return
        
        try:
            print(file_path)
            self.template_path = file_path
        except Exception as e:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Error")
            err_msg = "Error " + str(e) + " when reading " + str(file_path) + ". Please check the file and try again."
            error_dialog.setText(err_msg)
            error_dialog.exec_()
            return
        
        # Update button information
        print(f'Selected template path: {file_path}')
        
        self.template_button.setText(f"Select the template file\n✅ {os.path.normpath(file_path)}")
        self.template_button.setStyleSheet("")
        
        
    def _toggle_buttons(self, toggle: bool):
        """
        Disable all buttons to prevent user input during processing.
        """
        # File IO
        self.input0_folder_button.setEnabled(toggle)
        self.input1_folder_button.setEnabled(toggle)
        self.input2_folder_button.setEnabled(toggle)
        self.input3_folder_button.setEnabled(toggle)
        self.input0_type.setEnabled(toggle)
        self.input1_type.setEnabled(toggle)
        self.input2_type.setEnabled(toggle)
        self.input3_type.setEnabled(toggle)
        self.input_points_button.setEnabled(toggle)
        self.template_button.setEnabled(toggle)
        self.ip_address_input.setEnabled(toggle)
        self.port_input.setEnabled(toggle)
        self.x_measure_in.setEnabled(toggle)
        self.y_measure_in.setEnabled(toggle)
        self.z_measure_in.setEnabled(toggle)
        self.x_unit_in.setEnabled(toggle)
        self.y_unit_in.setEnabled(toggle)
        self.z_unit_in.setEnabled(toggle)
        self.x_measure_out.setEnabled(toggle)
        self.y_measure_out.setEnabled(toggle)
        self.z_measure_out.setEnabled(toggle)
        self.x_unit_out.setEnabled(toggle)
        self.y_unit_out.setEnabled(toggle)
        self.z_unit_out.setEnabled(toggle)
        for row in self.spin_boxes:
            for spin_box in row:
                spin_box.setEnabled(toggle)
        self.visualize_button.setEnabled(toggle)
        
        
    def _thread_visualize(self):
        """
        Thread the registration function.
        """
        t1 = Thread(target=self._visualize) 
        t1.start()
        
        
    def _visualize(self):
        """
        Generate a neuroglancer link locally and show it.
        """
        self._toggle_buttons(False)
        
        # Input localhost address and port
        ADDRESS = "0.0.0.0"
        PORT = 8080
        
        REGISTERED = True # If true, tries to load the registered data
        
        # Input paths or parameters
        input_paths = [self.input0_path, self.input1_path, self.input2_path, self.input3_path]
        image_layers = [None, None, None, None]
        
        # Input point annotations and misc. data to visualize
        points_path = self.input_points
        segmentation_path = self.template_path

        # Scaling parameters
        x_dim_in, y_dim_in, z_dim_in = [self.x_measure_in.value(), self.y_measure_in.value(), self.z_measure_in.value()]  # 1250, 1250, 50000 micrometers
        x_dim_out, y_dim_out, z_dim_out = [self.x_measure_out.value(), self.y_measure_out.value(), self.z_measure_out.value()]  # 1250, 1250, 50000 micrometers
        x_unit_in, y_unit_in, z_unit_in = [self.x_unit_in.currentText(), self.y_unit_in.currentText(), self.z_unit_in.currentText()]
        x_unit_out, y_unit_out, z_unit_out = [self.x_unit_out.currentText(), self.y_unit_out.currentText(), self.z_unit_out.currentText()]
        
        # For the affine matrix
        if REGISTERED:
            x_mat, y_mat, z_mat = [24.6403508772, 27.2375, 1]  # For registered data
        else:
            x_mat, y_mat, z_mat = [1, 1, 1]
        curr_matrix = [[spin_box.value() for spin_box in row] for row in self.matrix]
        curr_matrix = np.array(curr_matrix)
        curr_matrix[:3, :3] = curr_matrix[:3, :3][:, ::-1][::-1, :]
        #curr_matrix[:3, :3] = np.transpose(curr_matrix[:3, :3])  # Transpose the rotation matrix
        print(curr_matrix)
        
        # Default width and height since no information is available from just URLs
        z, h, w = 280, 8716, 11236
        
        # Names and information for the layers
        IMAGE_LAYER_NAME = 'image layer'
        image_layer_names = [self.input0_name.text(), self.input1_name.text(), self.input2_name.text(), self.input3_name.text()]
        VOLUME_LAYER_NAME = 'volume visualization'
        ANNOTATIONS_LAYER_NAME = 'point annotations'
        ANNOTATION_COLOR = '#ff0000'  # Red color for annotations
        SEGMENTATION_LAYER_NAME = 'allen mouse ccf'
        
        ################################################################
        # Setup the data sources
        ################################################################
        
        # Initialize Neuroglancer
        neuroglancer.set_server_bind_address(ADDRESS, PORT)
        
        dimensions = neuroglancer.CoordinateSpace(names=['z', 'y', 'x'], 
                                                  units=[z_unit_in, y_unit_in, x_unit_in], 
                                                  scales=[z_dim_in, y_dim_in, x_dim_in])
        
        for i in range(len(input_paths)):
            if input_paths[i] is not None:
                image_layer, url, image_dim = run_ng.create_image_layer(input_paths[i], dimensions)
                image_layers[i] = (image_layer, url)
                z, h, w = image_dim
        
        # Make a volume visualization layer, which is a copy of the original layer
        volume_layer = copy.deepcopy(image_layer)
        
        # Load input points
        if points_path:
            annotation_layer = run_ng.create_points_layer(points_path, ANNOTATION_COLOR, REGISTERED)
        
        # Load Allen CCF 2017 segmentation annotation
        if segmentation_path:
            segmentation_layer = run_ng.create_annotation_template_layer(segmentation_path, dimensions)
            
        ################################################################
        #  Setup the Neuroglancer viewer
        ################################################################
        
        viewer = neuroglancer.Viewer()
        
        # Add layers to the viewer
        with viewer.txn() as s:
            # Add image layer
            for i in range(len(image_layers)):
                if image_layers[i] is not None:
                    image_layer = image_layers[i][0]
                    url = image_layers[i][1]
                    s.layers.append(
                        name=image_layer_names[i],
                        source=run_ng.create_source(url, 
                                                    [x_dim_in, y_dim_in, z_dim_in], [x_unit_in, y_unit_in, z_unit_in], 
                                                    [x_dim_out, y_dim_out, z_dim_out], [x_unit_out, y_unit_out, z_unit_out], 
                                                    curr_matrix),
                        layer=image_layer,
                        opacity=1.0,
                        shader=run_ng.image_shader(),
                    )

            # Add volume visualization layer
            s.layers.append(
                name=VOLUME_LAYER_NAME,
                source=run_ng.create_source(url, 
                                            [x_dim_in, y_dim_in, z_dim_in], [x_unit_in, y_unit_in, z_unit_in], 
                                            [x_dim_out, y_dim_out, z_dim_out], [x_unit_out, y_unit_out, z_unit_out], 
                                            curr_matrix),
                layer=volume_layer,
                opacity=0,
                shader=run_ng.volume_shader(registered=REGISTERED),
                shaderControls={
                    "brightness": -1.5,
                    "tissueMinValue": 40,
                },
                blend="additive",
                volume_rendering=True,
                volumeRenderingGain=-4.6 if REGISTERED else -7.1,
            )
            
            # Add annotations layer if points are provided
            if points_path:
                s.layers.append(
                    name=ANNOTATIONS_LAYER_NAME,
                    source=run_ng.create_source("local://annotations", 
                                                [x_dim_in, y_dim_in, z_dim_in], [x_unit_in, y_unit_in, z_unit_in], 
                                                [x_dim_out, y_dim_out, z_dim_out], [x_unit_out, y_unit_out, z_unit_out], 
                                                curr_matrix),
                    layer=annotation_layer,
                )
                
            # Add segmentation layer
            if segmentation_path:
                s.layers.append(
                    name=SEGMENTATION_LAYER_NAME,
                    source=run_ng.create_source(segmentation_layer, 
                                                [x_dim_in, y_dim_in, z_dim_in], [x_unit_in, y_unit_in, z_unit_in], 
                                                [x_dim_out, y_dim_out, z_dim_out], [x_unit_out, y_unit_out, z_unit_out], 
                                                curr_matrix),
                    layer=segmentation_layer,
                    shader=run_ng.segmentation_shader(),
                )
                    
            # Setup viewer settings
            s.voxel_coordinates = [int(w / 2 - 1) * x_mat, int(h / 2 - 1) * y_mat, int(z / 2 - 1) * z_mat]  # Starting location of camera
            s.dimensions = neuroglancer.CoordinateSpace(names=["x", "y", "z"],
                                                        units="m",
                                                        scales=[x_dim_in, y_dim_in, z_dim_in])
            s.projection_scale = 13107.2
            s.cross_section_scale = 16.068429538550138  # How zoomed in the preview is
            s.projection_orientation = [-0.2444217950105667, 
                                        -0.01222456619143486, 
                                        -0.011153397150337696, 
                                        0.9695277810096741]
            s.selectedLayer.layer = IMAGE_LAYER_NAME
            s.selectedLayer.visible = True
            s.selectedLayer.size = 1162
        
        print("\nGenerated local Neuroglancer URL:", viewer)
        webbrowser.open(viewer.get_viewer_url())
        input("Press ENTER to end the visualization...")
            
        self._toggle_buttons(True)


if __name__ == "__main__":
    class U01App(QMainWindow):
        def __init__(self):
            super().__init__()

            # Setup initial window settings
            WINDOW_HEIGHT = 950
            WINDOW_WIDTH = 200
            
            # Setup window information
            self.setWindowTitle("U01 Workflow GUI")
            if os.path.exists('icon.png'):
                window_icon = QIcon('icon.png')
            else:
                window_icon = QIcon('../icon.png')
            self.setWindowIcon(window_icon)
            self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
            
            # Setup layout and tabs
            self.tabs = QTabWidget()
            self.tab_registration = NeuroglancerTab(self)
            self.tabs.addTab(self.tab_registration, "Visualization")
            self.setCentralWidget(self.tabs)
        
    app = QApplication(sys.argv)
    window = U01App()
    window.show()
    sys.exit(app.exec())