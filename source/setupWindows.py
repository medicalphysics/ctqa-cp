
"""
Created on Thu Jan 16 10:41:23 2014

@author: erlean

"""
###############cx_freeze#####################################################
""" Hacked cx_freeze\windist.py to allow no administrator rights install
    In def add_properties(self):
        props = [...
        - ('ALLUSERS', '1')
        + ('ALLUSERS', '2'),
        + ('MSIINSTALLPERUSER','1')
        ]
"""

import sys
from cx_Freeze import setup, Executable
from ctqa_cp import Version as app_version
import numpy as np





base = None
if sys.platform == 'win32':
    base = 'Win32GUI'
    if len(sys.argv) == 1:
        sys.argv.append("bdist_msi")
else:
    if len(sys.argv) == 1:
        sys.argv.append("build")

exe = Executable(
    script="ctqa_cp.pyw",
    base=base,
    icon='Icons\\ic.ico'
    )

# http://msdn.microsoft.com/en-us/library/windows/desktop/aa371847(v=vs.85).aspx
shortcut_table = [
    ("DesktopShortcut",        # Shortcut
     "DesktopFolder",          # Directory_
     "CTQA_cp",                # Name
     "TARGETDIR",              # Component_
     "[TARGETDIR]CTQA_cp.exe",  # Target
     None,                     # Arguments
     None,                     # Description
     None,                     # Hotkey
     None,                     # Icon
     None,                     # IconIndex
     None,                     # ShowCmd
     'TARGETDIR'               # WkDir
     )
    ]

# Now create the table dictionary
msi_data = {"Shortcut": shortcut_table}



includefiles = []#('path2python\\Lib\\site-packages\\scipy\\special\\_ufuncs.pyd','_ufuncs.pyd')]

# modules not included automatical by cx_freeze
#includes = ['skimage.draw', 'skimage.draw._draw','skimage._shared.geometry','scipy.sparse','skimage.filter','skimage.feature',
#            'scipy.ndimage','scipy.special', 'scipy.linalg', 'scipy.integrate']#,'scipy.special._ufuncs_cxx', 'scipy.linalg']
includes = ['scipy.special', 'scipy.ndimage','scipy.linalg', 'scipy.integrate']
excludes = ['curses', 'email', 'ttk', 'PIL', 'matplotlib',]
#            'tzdata']

tk_excludes = ["pywin", "pywin.debugger", "pywin.debugger.dbgcon",
               "pywin.dialogs", "pywin.dialogs.list",
               "Tkconstants", "Tkinter", "tcl"]
excludes += tk_excludes

build_exe_options = {'packages': includes,
                     'excludes': excludes,
                     'includes': includes,
                     'include_files': includefiles}
bdist_msi_options = {'upgrade_code': "{601f8668-1a53-478f-8499-d8735b2eef5b}",
                     'data': msi_data}
setup(
    name="CTQA_cp",
    version=app_version,
    description="CatPhan analysis tool",
    executables=[exe],
    include_dirs=np.get_include(),
    options={
        "build_exe": build_exe_options,
        "bdist_msi": bdist_msi_options})
