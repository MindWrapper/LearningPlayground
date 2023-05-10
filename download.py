from fastbook import search_images_bing, download_images
from dotenv import load_dotenv
from fastbook import get_image_files, verify_images, Path
import os

def _cleanupImages(data_dir):
    # cleanup up images
    fns = get_image_files(data_dir)
    failed = verify_images(fns)
    failed.map(Path.unlink) 

def download_images_for_types(baseDir, mainCategory, subcategories):
    path = Path(baseDir).joinpath(mainCategory)
    if path.exists():
        print (path.name + " already exists, skip downloading images")
        return # assume images already downloaded
    
    load_dotenv('/Users/yan/.secrets/fast.ai/.env')
    azureKey = os.environ.get('AZURE_SEARCH_KEY')

    path.mkdir(exist_ok=False, parents=True)

    for o in subcategories:
        print("Downloading images for " + o)
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(azureKey, f'{o} {mainCategory}')
        download_images(dest, urls=results.attrgot('contentUrl'))

    _cleanupImages(path)
