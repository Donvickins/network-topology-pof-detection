import sys
from pathlib import Path


def get_base_path() -> Path:
    """
    Determines the base path of the application, whether running as a script or a frozen executable.
    This is crucial for locating bundled resources like Tesseract.
    """

    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # PyInstaller one-file mode
        return Path(sys._MEIPASS)
    elif Path(sys.argv[0]).resolve().suffix.lower() == '.exe':
        # Nuitka or other .exe: sys.argv[0] is the path to the executable.
        return Path(sys.argv[0]).resolve().parent
    else:
        # Running as a normal Python script
        return Path(__file__).resolve().parents[2]
