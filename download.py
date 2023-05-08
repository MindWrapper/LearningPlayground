from fastbook import search_images_bing, download_images
import pathlib


def download_images_for_types(toDir, azureKey, types):
    path = pathlib.Path(toDir)
    if path.exists():
        print (path.name + " already exists, skip downloading images")
        return # assume images already downloaded
    
    path.mkdir(exist_ok=False)

    for o in types:
        print("downloading images for " + o)
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(azureKey, f'{o} bear')
        download_images(dest, urls=results.attrgot('contentUrl'))