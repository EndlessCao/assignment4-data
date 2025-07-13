import fasttext
import pathlib
DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data"
CLASSIFY_MODEL_PATH = DATA_PATH / "classify_model.bin"
def prepare_train_data():
    