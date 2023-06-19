The method I adopted from Chapters 1 & 2 of the course enables experimentation with various categories of images.

# Prepare Environment

This app is built using Python 3.11.

## Setup Azure Account

I've found that Bing Image Search works the best, hence, you'll need an Azure key to perform image searches. Out of the many Azure services available, the one you need is `Bing Search v7`. You can follow [this guide](https://stackoverflow.com/questions/65706220/fast-ai-course-2020-httperror-401-client-error-permissiondenied-for-url) to set it up.

## Setup Azure Search

Once you have the Azure key, store it under `~/.secrets/fast.ai/.env`. This is where `download.py` expects it to be. If you run `cat ~/.secrets/fast.ai/.env`, it should display: `AZURE_SEARCH_KEY=your_key_goes_here`.

Alternatively, you can set it up as an environment variable: `export AZURE_SEARCH_KEY=your_key_goes_here`.

## Setup Virtual Environment

Working with a virtual environment can make the process simpler.

1. Navigate to the desired directory:

    ```commandline 
    cd chapter_2
    ```

2. Create a virtual environment:

    ```commandline  
    python3.11 -m venv env
    ```

3. Activate the virtual environment:

    For Unix or MacOS, use:

    ```commandline   
    source env/bin/activate
    ``` 

    For Windows, use:

    ```commandline 
    env\Scripts\activate 
    ```

4. Install the required packages:

    ```commandline 
    pip install -r requirements.txt
    ```

# Prepare Data

Think about the kind of images you want your model to train with. Initially, I tried to recognize different brands of electric guitars, but it didn't work out well, possibly due to the similar appearances of electric guitars. On the other hand, fruit classification worked very well. 

Download images 
```commandline
python3.11 ./download.py --main-category="fruit single" --sub-categories="Banana,Apple,Pear" --save-to=data/fruits
``` 

It might take a while and will cost you a few cents (if you're not on free trialAzure credit).

The downloaded data will be stored in `./data/fruits` (not included in git). The directory structure will look like this:

```
data
    fruits
        Apple
        Banana
        Pear
```

Each leaf folder will contain corresponing images. Might be a good idea to clean it up a bit.

# Fine-tune the Model 

To train use: 
```commandline 
PYTORCH_ENABLE_MPS_FALLBACK=1 python3.11 fine_tune.py --data-dir=./data/fruits --save-model-as=models/fruits.pkl
```

# Classify an Image

A couple of test images can be found in the `./test_images` folder.

Run 
```commandline 
python3.11 ./classify.py --image=./test_images/banana.jpg --model=models/fruits.pkl
```
to get an output similar to:

```
Predicted class: Banana                                                                                                  
Confidence: 0.9993
```

Run 
```commandline 
python3.11 ./classify.py --image=./test_images/banana.jpg --model=models/fruits.pkl
```

```commandline
Predicted class: Apple                                                                                                   
Confidence: 0.5690
```