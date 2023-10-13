# hack to make sure protobuf files are found
import pathlib
import sys

_dir_path = str(pathlib.Path(__file__).resolve().parent / "prediction_service")
if _dir_path not in sys.path:
    sys.path.append(_dir_path)
