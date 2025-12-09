# RheoJAX GUI Stylesheets

Professional light and dark themes for the RheoJAX GUI application.

## Overview

The stylesheets provide comprehensive styling for all Qt widgets with:
- **Light Theme**: Clean, professional design with JAX-inspired royal blue
- **Dark Theme**: Modern dark interface with brighter accent colors
- **Consistent Design Language**: Proper spacing, borders, hover states, and transitions

## Color Schemes

### Light Theme
- **Background**: #FFFFFF (white)
- **Surface**: #F8FAFC (light gray)
- **Primary**: #2563EB (royal blue - JAX-inspired)
- **Accent**: #7C3AED (purple - Bayesian features)
- **Success**: #10B981 (green)
- **Warning**: #F59E0B (amber)
- **Error**: #EF4444 (red)
- **Text**: #1E293B (slate)
- **Border**: #E2E8F0

### Dark Theme
- **Background**: #0F172A (slate-900)
- **Surface**: #1E293B (slate-800)
- **Primary**: #3B82F6 (brighter blue)
- **Accent**: #8B5CF6 (brighter purple)
- **Success**: #34D399 (brighter green)
- **Warning**: #FBBF24 (brighter amber)
- **Error**: #F87171 (brighter red)
- **Text**: #F1F5F9 (light)
- **Border**: #334155 (slate-700)

## Usage

### Basic Usage

```python
from PyQt6.QtWidgets import QApplication
from rheojax.gui.resources.styles import get_stylesheet

app = QApplication([])

# Apply light theme (default)
app.setStyleSheet(get_stylesheet("light"))

# Or apply dark theme
app.setStyleSheet(get_stylesheet("dark"))
```

### Direct Functions

```python
from rheojax.gui.resources.styles import (
    get_light_stylesheet,
    get_dark_stylesheet,
)

# Get specific theme
light_style = get_light_stylesheet()
dark_style = get_dark_stylesheet()

app.setStyleSheet(light_style)
```

### Theme Switching

```python
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_theme = "light"
        self._apply_theme()
    
    def _apply_theme(self):
        """Apply current theme."""
        stylesheet = get_stylesheet(self.current_theme)
        self.setStyleSheet(stylesheet)
    
    def toggle_theme(self):
        """Toggle between light and dark themes."""
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self._apply_theme()
```

## Styled Widgets

All Qt widgets are comprehensively styled:

### Input Widgets
- QPushButton (with variants: accent, success, warning, error, secondary)
- QLineEdit, QSpinBox, QDoubleSpinBox
- QComboBox
- QCheckBox, QRadioButton
- QSlider
- QTextEdit, QPlainTextEdit

### Container Widgets
- QMainWindow, QWidget
- QGroupBox
- QTabWidget, QTabBar
- QDockWidget
- QSplitter

### View Widgets
- QTreeWidget, QTreeView
- QTableWidget, QTableView
- QListWidget
- QHeaderView

### UI Elements
- QMenuBar, QMenu
- QToolBar, QToolButton
- QStatusBar
- QProgressBar
- QScrollBar (horizontal and vertical)
- QToolTip

### Dialogs
- QDialog
- QMessageBox
- QFileDialog

## Custom Button Variants

The stylesheets support custom button variants via objectName:

```python
# Primary button (default)
btn_primary = QPushButton("Primary")

# Accent button (purple - for Bayesian features)
btn_accent = QPushButton("Bayesian Analysis")
btn_accent.setObjectName("accentButton")

# Success button (green)
btn_success = QPushButton("Success")
btn_success.setObjectName("successButton")

# Warning button (amber)
btn_warning = QPushButton("Warning")
btn_warning.setObjectName("warningButton")

# Error button (red)
btn_error = QPushButton("Delete")
btn_error.setObjectName("errorButton")

# Secondary button (outline style)
btn_secondary = QPushButton("Cancel")
btn_secondary.setObjectName("secondaryButton")
```

## Custom Progress Bar Variants

```python
# Primary progress bar (default blue)
progress = QProgressBar()

# Success progress bar (green)
progress_success = QProgressBar()
progress_success.setObjectName("successProgress")

# Warning progress bar (amber)
progress_warning = QProgressBar()
progress_warning.setObjectName("warningProgress")

# Error progress bar (red)
progress_error = QProgressBar()
progress_error.setObjectName("errorProgress")
```

## Features

### Hover States
All interactive widgets have smooth hover transitions for better UX feedback.

### Focus Indicators
Input widgets show clear focus states with blue borders (primary color).

### Disabled States
Disabled widgets are styled with muted colors and reduced opacity.

### Border Radius
Consistent 6px border radius on containers and 4px on smaller elements for a modern look.

### Scrollbars
Custom-styled scrollbars with minimal design and smooth hover effects.

### Spacing
Proper padding and margins for comfortable spacing throughout the UI.

## File Structure

```
styles/
├── __init__.py          # Public API with helper functions
├── light.qss            # Light theme stylesheet (~17KB)
├── dark.qss             # Dark theme stylesheet (~17KB)
└── README.md            # This file
```

## Notes

- Stylesheets reference icon files via `url(:/icons/...)` - ensure icons are available in resources
- Font family uses system fonts: "Segoe UI" (Windows), "SF Pro Text"/"SF Pro Display" (macOS), fallback to sans-serif
- All colors use hex values for consistency and performance
- Border-radius and padding values are carefully tuned for visual harmony
