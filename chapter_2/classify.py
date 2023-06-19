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
model_file_path = args.model
learn_inf = load_learner(model_file_path)

print(f"Classyfying image {args.image} using model: {model_file_path}...")
pred_class, pred_idx, probs = learn_inf.predict(args.image)

print(f"Predicted class: {pred_class}")
print(f"Confidence: {probs[pred_idx]:.4f}")