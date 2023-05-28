from fastbook import search_images_bing, download_images
from dotenv import load_dotenv
from fastbook import get_image_files, verify_images, Path
import os
import argparse

def _cleanupInvalidImages(data_dir):
    fns = get_image_files(data_dir)
    failed = verify_images(fns)
    failed.map(Path.unlink) 

def download_images_for_types(baseDir, mainCategory, subcategories):
    path = Path(baseDir).joinpath(mainCategory)
    if path.exists():
        print("Skipping image download because the directory already exists:\n" + str(path.absolute()) + "\nAssuming that the images have already been downloaded.")
        return 

    home_dir = os.path.expanduser("~")
    dotenv_path = os.path.join(home_dir, "secrets", "fast.ai", ".env")
    load_dotenv(dotenv_path)
    print("dotev path:" + dotenv_path)
    azureKey = os.environ.get('AZURE_SEARCH_KEY')

    path.mkdir(exist_ok=False, parents=True)

    for o in subcategories:
        print("Downloading images for " + o)
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(azureKey, f'{o} {mainCategory}')
        download_images(dest, urls=results.attrgot('contentUrl'))

    _cleanupInvalidImages(path)

def main():
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument("--main-category", type=str, help="Main category. `guitars`")
    parser.add_argument("--sub-categories", type=str, help="Sub categories. Comma separated, no spaces. Example:'ESP,PRS,Gibson,PRS,Fender'")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baseDir = Path(script_dir).joinpath('data')
    download_images_for_types(baseDir, args.main_category, args.sub_categories.split(','))

if __name__ == "__main__":
    main()
