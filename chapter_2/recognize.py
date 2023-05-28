import os

from fastbook import *
from fastai.vision.widgets import *
from fastai.vision.all import *
import argparse

parser = argparse.ArgumentParser(description="Process arguments.")
parser.add_argument("--model", type=str, help="Name of trained model to use for recognition inside 'models' folder")
parser.add_argument("--image", type=str, help="Path to a file to recognize")
args = parser.parse_args()
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
models_dir = script_dir.joinpath("models")
model_file_path = Path(models_dir).joinpath(args.model)
learn_inf = load_learner(model_file_path)

print(learn_inf.predict(args.image))
