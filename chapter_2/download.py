from fastbook import search_images_bing, download_images
from dotenv import load_dotenv
from fastbook import get_image_files, verify_images, Path
import os
import argparse

# separate data directory and category
def _cleanupInvalidImages(data_dir):
    fns = get_image_files(data_dir)
    failed = verify_images(fns)
    failed.map(Path.unlink) 

def download_images_for_types(saveToDir, mainCategory, subcategories):
    saveToDir = Path(saveToDir)
    if saveToDir.exists():
        print("Skipping image download because the directory already exists:\n" + str(saveToDir.absolute()) + "\nAssuming that the images have already been downloaded.")
        return 

    home_dir = os.path.expanduser("~")
    dotenv_path = os.path.join(home_dir, ".secrets", "fast.ai", ".env")
    load_dotenv(dotenv_path)
    azureKey = os.environ.get('AZURE_SEARCH_KEY')

    saveToDir = Path(saveToDir)
    Path(saveToDir).mkdir(exist_ok=False, parents=True)

    for o in subcategories:
        print("Downloading images for " + o)
        dest = (saveToDir/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(azureKey, f'{o} {mainCategory}', max_images=150)
        download_images(dest, urls=results.attrgot('contentUrl'))

    _cleanupInvalidImages(saveToDir)

def main():
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument("--main-category", type=str, help="Main category. `fruit`")
    parser.add_argument("--sub-categories", type=str, help="Sub categories. Comma separated, no spaces. Example:'Banana,Apple,Pear'")
    parser.add_argument("--save-to", type=str, help="Directory to save the images.")
    args = parser.parse_args()
    download_images_for_types(args.save_to, args.main_category, args.sub_categories.split(','))

if __name__ == "__main__":
    main()
