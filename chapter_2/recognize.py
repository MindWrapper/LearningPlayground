import os

from fastbook import *
from fastai.vision.widgets import *
from fastai.vision.all import *
import argparse

# go get azure tokens use # to setup asuze web search, see https://stackoverflow.com/questions/65706220/fast-ai-course-2020-httperror-401-client-error-permissiondenied-for-url
# python3.11 recognize.py --model=guitar.pkl --file=../my_prs.jpg
parser = argparse.ArgumentParser(description="Process arguments.")
parser.add_argument("--model", type=str, help="Name of trained model to use for recognition inside .models folder")
parser.add_argument("--image", type=str, help="Path to a file to recognize")
args = parser.parse_args()
models_dir = os.path.dirname(os.path.abspath(__file__)) + "/.models"
model_file_path = Path(models_dir).joinpath(args.model)
learn_inf = load_learner(model_file_path)

print(learn_inf.predict(args.image))
