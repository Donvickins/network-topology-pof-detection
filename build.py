import  logging
import subprocess as command
import importlib.util
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path.cwd() / 'build'

try:
    spec = importlib.util.find_spec("ultralytics")
    nuitka_exec = Path(sys.executable).parent / 'Scripts/nuitka.cmd'

    if not nuitka_exec.exists():
        logger.error('nuitka not found')
        logger.info("Ensure 'nuitka' is installed in current environment")
        exit(1)

    if not spec or not spec.origin:
        raise ImportError("Could not find the 'ultralytics' package. Please ensure it is installed.")

    ultralytics_path = Path(spec.origin).parent
    cfg_path = (ultralytics_path / "cfg").as_posix()
    include_cfg_arg = f"--include-data-files={cfg_path}/*.yaml=ultralytics/cfg/"

except (ImportError, AttributeError) as e:
    logger.error(f"{e}")
    exit(1)

try:
    command.run(
        [nuitka_exec, 'app.py', '--mode=standalone', '--lto=no', '--mingw64', '--assume-yes-for-downloads',
         f'--output-dir={str(OUTPUT_DIR)}', '--enable-plugins=tk-inter', '--python-flag=no_asserts', '--python-flag=no_docstrings', '--include-module=pytesseract',
         '--include-module=core.api', '--noinclude-data-files=ultralytics/assets/*', '--noinclude-data-files=ultralytics/data/*',
         '--noinclude-data-files=ultralytics/hub/*', '--noinclude-data-files=ultralytics/models/hub/*', '--noinclude-data-files=ultralytics/utils/callbacks/*',
         include_cfg_arg, '--output-filename=POF_Detection.exe', '--windows-icon-from-ico=icon/icon.ico'
         ],
        check=True,
        text=True
    )
except command.CalledProcessError as e:
    logger.error(f'Failed to build program: Reason: {e}')