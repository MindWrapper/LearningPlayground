The way I decided to adopt example from chapter 1 & 2 of the course is making it possible to experment with various categegories of images. My own interes was to see if I can train a model to recognize my newly bought `PRS SE Custom 24-08 Vintage Sunburst`

So here is how it can be used:

## Prepare environment

1. Setup azure account.

I found that Bing image search work the best, therefore you are going to need and AZURE_KEY to perform image search. There are hundreds of Azure services, the one you need is called `Bing Search v7`. [Here](https://stackoverflow.com/questions/65706220/fast-ai-course-2020-httperror-401-client-error-permissiondenied-for-url) I found on the internet on how to set it up.

2. Setup secrets.

Once you have an Azure  key place it under `~/.secrets/fast.ai/.env`. This is where `download.py` will expect it to be

`cat  ~/.secrets/fast.ai/.env`

`AZURE_SEARCH_KEY=your_key_goes_here`

Alternatively you can set it up as an environment variable:

`export AZURE_SEARCH_KEY=your_key_goes_here`

3. Install python 3.11.

## Prepare data

Think of your data. What kind of images you want to train your model with? I went with guitars.

1. `cd chapter_2` 
2. Activate python the environment: `sh ./env/bin/activate`
3. Run `python3.11 ./download.py --main-category='guitars' --sub-categorie='PRS,Ibanez,ESP,Fender,Gibson'`
   It might take a while and will cost you a few cents.

This will download data to `./data` (not stored in git) For the example below the layout will look like this

- data
    - guitars
        - PRS
        - Ibanez
        - ESP
        - Fender

Leaf folders such AS PRS, Ibanexz, ESP, Finder will contain images.

## Train the model 

`PYTORCH_ENABLE_MPS_FALLBACK=1 python3.11 train.py --data-dir='data/Electric Guitars'`

PYTORCH_ENABLE_MPS_FALLBACK is needed to train on Apple M1.

## Classify an image

In this example I'll use photo of my guitar:
![My guitar (./my_prs,jpg)](./my_prs.jpg)

`python3.11 ./recognize.py --image='./my_prs.jpg' --model='./models/Electric Guitars.pkl'`

This will print out the top 5 predictions:


