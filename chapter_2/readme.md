The way I decided to adopt example from chapter 1 & 2 of the course is making it possible to experment with various categegories of images. My own interest was to see if I can train a model to recognize my newly bought `PRS SE Custom 24-08 Vintage Sunburst`.

# Prepare environment

This app is done using python 3.11.

## Setup azure account.

I found that Bing image search work the best, therefore you are going to need an azure key to perform image search. There are hundreds of Azure services, the one you need is called `Bing Search v7`. [Here](https://stackoverflow.com/questions/65706220/fast-ai-course-2020-httperror-401-client-error-permissiondenied-for-url) I found on the internet on how to set it up.

## Setup azure search

Once you have an Azure  key place it under `~/.secrets/fast.ai/.env`. This is where `download.py` will expect it to be. If you type `cat  ~/.secrets/fast.ai/.env` is should show:
`AZURE_SEARCH_KEY=your_key_goes_here`

Alternatively you can set it up as an environment variable:`export AZURE_SEARCH_KEY=your_key_goes_here`

## Setup virtual environment

I found that it is easier to work with virtual environment.

1. `cd chapter_2`
2. `python3.11 -m venv env`
3. `source env/bin/activate`  # On Windows, use `env\Scripts\activate`
4. `pip install -r requirements.txt`

# Prepare data

Think of your data. What kind of images you want to train your model with? First I went with an idea of recognizing electric guitars brands. This didn't work well, probably because electric guitars a to much alike. I hope I'll find out later int the course. But Fruit's classification worked really well. 

Run `python3.11 ./download.py --main-category="fruit" --sub-categorie="Banana,Apply,Pear"`
It might take a while and will cost you a few cents (if you are not no free trial credit)

This will download data to `./data` (not stored in git) For the example below the layout will look like this

```
data
    fruits
        Apple
        Banan
        Pear
```

Each leaf folder will contain corresponing images.

# Fine-tune the model 

`PYTORCH_ENABLE_MPS_FALLBACK=1 python3.11 train.py --data-dir="data/fruits"`

PYTORCH_ENABLE_MPS_FALLBACK is needed to train on Apple M1.

# Classify an image

There are a couple of test images inside `./test_images` folder.

`python3.11 ./recognize.py --image=./test_images/apple.jpg --model=fruits.pkl`

This will produce following output

TODO: