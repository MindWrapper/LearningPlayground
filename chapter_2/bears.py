import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

load_dotenv('/Users/yan/.secrets/fast.ai/.env')

# this way we can train models withut GPU support
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# to setup asuze web search, see https://stackoverflow.com/questions/65706220/fast-ai-course-2020-httperror-401-client-error-permissiondenied-for-url
AZURE_KEY = os.environ.get('AZURE_SEARCH_KEY')

import fastbook
from fastbook import *
from fastai.vision.widgets import *
from download import download_images_for_types

# this will let us to see windows with images. almost like in jypiter notebook :) 
plt.ion()

fastbook.setup_book()
data_dir = os.path.dirname(os.path.abspath(__file__)) + "/.data"
download_images_for_types(data_dir, AZURE_KEY, ['grizzly','black','teddy'])

# cleanup up images
fns = get_image_files(data_dir)
failed = verify_images(fns)
failed.map(Path.unlink)

bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # independent variable(images), dependent variable(category - types of bears)
    get_items=get_image_files, # get_image_files is a function that returns a list of all the images in the path (recursively)
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label, # get_y dependant variable if often called y. parent_label is a function that returns the parent directory of the image
    item_tfms=RandomResizedCrop(224, min_scale=0.5), 
    #batch_tfms=aug_transforms(mult=2)
    # we don't feed the model images one-by-one, but instead we feed them in mini-batches. Inside each mini-batch, all images must be the same size.
    # Resize(128) is a transform that resizes all images to 128x128
    # There are several tranform strategies:
    # ResizeMethod.Crop - crops the image to the desired size
    # ResizeMethod.Pad - adds padding to the image to make it the desired size
    # ResizeMethod.Squish - squishes the image to the desired size
    # Each has it's owne problematic
    # crop - we might lose important information
    # squish - unrealistic shapes
    # recommend approch in is to randomly select part of the image, and crop to just that part
    # this is what RandomResizedCrop does
    )

# # dls includes validation set and training set
dls = bears.dataloaders(data_dir)

# resize the images to 128x128
# apply data augmentation to an entier batch_tfms
# We don't use RandomResizedCrop, casue we have aug_transforms which will apply set 
# of an augmentations to the images that fast.ai authors found to be useful
#bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
#dls = bears.dataloaders(path)
#dls.train.show_batch(max_n=8, nrows=2, unique=True)
#input("Viewing results of aug_transforms. Press Enter to continue...")

# bears = bears.new(
#     item_tfms=RandomResizedCrop(224, min_scale=0.5),
#     batch_tfms=aug_transforms())
# dls = bears.dataloaders(path)

found_trained_model = Path('export.pkl').exists()
if not found_trained_model:
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)
    
    # clean up the model
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    input("Viewing confusion matrix. Press Enter to continue...")

    interp.plot_top_losses(6, nrows=2)
    input("Viewing Images that were incorectly classified...")

    # unfortunately, we can't use this, cause it's not working with 
    # https://github.com/jupyter-widgets/ipywidgets/issues/3731
    #cleaner = ImageClassifierCleaner(learn)
    #cleaner
    # for idx in cleaner.delete(): 
    #     cleaner.fns[idx].unlink()

    dls = bears.dataloaders(data_dir)
    print("retraining the model on cleaned data")
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    
    # re-learn on the cleaned data
    learn.fine_tune(4)

    learn.export("export.pkl")

learn_inf = load_learner('export.pkl')

print()
print(learn_inf.predict( os.path.dirname(os.path.abspath(__file__))  + '/teddy.png'))
print(learn_inf.predict( os.path.dirname(os.path.abspath(__file__))  + '/grizly.jpeg'))
print(learn_inf.predict( os.path.dirname(os.path.abspath(__file__))  + '/teddy2.png'))