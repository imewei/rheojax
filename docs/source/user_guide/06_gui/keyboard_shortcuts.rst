.. _gui-keyboard-shortcuts:

==================
Keyboard Shortcuts
==================

Reference for all keyboard shortcuts in RheoJAX GUI.

Global Shortcuts
================

File Operations
---------------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Shortcut
   * - New Project
     - Create new project
     - Ctrl+N / Cmd+N
   * - Open Project
     - Open existing project
     - Ctrl+O / Cmd+O
   * - Save Project
     - Save current project
     - Ctrl+S / Cmd+S
   * - Save As
     - Save project with new name
     - Ctrl+Shift+S / Cmd+Shift+S
   * - Import Data
     - Import data file
     - Ctrl+I / Cmd+I
   * - Export
     - Export current results
     - Ctrl+E / Cmd+E
   * - Quit
     - Exit application
     - Ctrl+Q / Cmd+Q

Edit Operations
---------------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Shortcut
   * - Undo
     - Undo last action
     - Ctrl+Z / Cmd+Z
   * - Redo
     - Redo undone action
     - Ctrl+Shift+Z / Cmd+Shift+Z
   * - Copy
     - Copy selection
     - Ctrl+C / Cmd+C
   * - Paste
     - Paste from clipboard
     - Ctrl+V / Cmd+V
   * - Preferences
     - Open settings
     - Ctrl+, / Cmd+,

View Operations
---------------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Shortcut
   * - Toggle Sidebar
     - Show/hide navigation
     - Ctrl+B / Cmd+B
   * - Toggle Residuals
     - Show/hide residuals panel
     - Ctrl+R / Cmd+R
   * - Full Screen
     - Toggle full screen mode
     - F11 / Ctrl+Cmd+F
   * - Zoom In
     - Increase plot zoom
     - Ctrl++ / Cmd++
   * - Zoom Out
     - Decrease plot zoom
     - Ctrl+- / Cmd+-
   * - Reset Zoom
     - Reset to default zoom
     - Ctrl+0 / Cmd+0

Navigation
----------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Shortcut
   * - Go to Data
     - Navigate to Data page
     - Ctrl+1 / Cmd+1
   * - Go to Fit
     - Navigate to Fit page
     - Ctrl+2 / Cmd+2
   * - Go to Bayesian
     - Navigate to Bayesian page
     - Ctrl+3 / Cmd+3
   * - Go to Transform
     - Navigate to Transform page
     - Ctrl+4 / Cmd+4
   * - Go to Export
     - Navigate to Export page
     - Ctrl+5 / Cmd+5
   * - Next Tab
     - Switch to next tab
     - Ctrl+Tab
   * - Previous Tab
     - Switch to previous tab
     - Ctrl+Shift+Tab

Page-Specific Shortcuts
=======================

Data Page
---------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Shortcut
   * - Browse Files
     - Open file browser
     - Ctrl+O / Cmd+O
   * - Clear Data
     - Clear current dataset
     - Delete / Backspace
   * - Auto Detect Mode
     - Auto-detect test mode
     - Ctrl+D / Cmd+D

Fit Page
--------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Shortcut
   * - Run Fit
     - Start NLSQ fitting
     - F5 / Ctrl+Return
   * - Stop Fit
     - Cancel running fit
     - Escape
   * - Reset Parameters
     - Reset to defaults
     - Ctrl+R / Cmd+R
   * - Auto Initialize
     - Smart parameter init
     - Ctrl+A / Cmd+A
   * - Toggle Parameter Lock
     - Fix/unfix selected param
     - Space

Bayesian Page
-------------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Shortcut
   * - Run Inference
     - Start MCMC sampling
     - F5 / Ctrl+Return
   * - Stop Inference
     - Cancel MCMC
     - Escape
   * - Next Plot Type
     - Cycle ArviZ plot types
     - Right Arrow
   * - Previous Plot Type
     - Cycle ArviZ plot types
     - Left Arrow
   * - Refresh Plot
     - Regenerate current plot
     - F5

Transform Page
--------------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Shortcut
   * - Apply Transform
     - Apply selected transform
     - F5 / Ctrl+Return
   * - Accept Result
     - Accept transformation
     - Enter
   * - Cancel
     - Discard transformation
     - Escape

Plot Canvas Shortcuts
=====================

Mouse Controls
--------------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Control
   * - Zoom
     - Zoom in/out
     - Mouse Wheel
   * - Pan
     - Move view
     - Click + Drag
   * - Reset View
     - Reset to original
     - Double Click
   * - Data Tooltip
     - Show data values
     - Hover
   * - Context Menu
     - Plot options
     - Right Click

Keyboard Controls
-----------------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Shortcut
   * - Toggle Log X
     - Log scale X axis
     - X
   * - Toggle Log Y
     - Log scale Y axis
     - Y
   * - Toggle Legend
     - Show/hide legend
     - L
   * - Toggle Grid
     - Show/hide grid
     - G
   * - Export Plot
     - Export current plot
     - Ctrl+E / Cmd+E

Table Shortcuts
===============

Parameter Table
---------------

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Action
     - Description
     - Shortcut
   * - Edit Cell
     - Enter edit mode
     - Enter / F2
   * - Navigate
     - Move between cells
     - Arrow Keys
   * - Toggle Fixed
     - Lock/unlock parameter
     - Space
   * - Copy Value
     - Copy cell value
     - Ctrl+C / Cmd+C
   * - Paste Value
     - Paste to cell
     - Ctrl+V / Cmd+V

Customizing Shortcuts
=====================

Shortcuts can be customized:

1. Go to **Edit > Preferences > Keyboard**
2. Search for action
3. Click shortcut field
4. Press new key combination
5. Click **Apply**

Conflicts
---------

If a shortcut conflicts with another:

- Warning icon appears
- Both actions listed
- Choose which to keep

Reset Defaults
--------------

To restore default shortcuts:

1. Go to **Edit > Preferences > Keyboard**
2. Click **Reset to Defaults**
3. Confirm action

Platform Notes
==============

macOS
-----

- Use **Cmd** instead of **Ctrl**
- **Option** = **Alt**
- System shortcuts take precedence

Windows/Linux
-------------

- Use **Ctrl** for most shortcuts
- **Alt** for menu access
- Function keys (F1-F12) work directly

Accessibility
-------------

- All shortcuts have menu equivalents
- Screen reader compatible
- High contrast mode available
