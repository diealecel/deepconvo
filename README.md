# deepConvo

Created as a final project for CS 229 at Stanford University during fall 2018.

## How to use deepConvo

Note that deepConvo runs on Python 2.7.x. Please ensure your environment is set up to allow for this.

### Setup

To run code in this repository, first install the dependencies by running the following code snippet.

``` bash
# Make sure you're in the repository's home folder.
cd deepconvo

# Create and activate a new virtual environment.
virtualenv .env
source .env/bin/activate

# Install the dependencies in your virtual environment.
pip install -r dependencies.txt
```

Make sure to have the environment activated whenever you are dealing with deepConvo.

### Training the model

To train the model, follow the code snippet below and choose one training method.

``` bash
# Assuming you're in the repository's home folder, go to the source folder of a dataset.
cd [insert database here]_src/

# Train the model using stacked RGB tensors.
python run_baseline.py

# Train the model using stacked RGB optical flow tensors.
python run_rgb_optical_flow.py
```

If you would like to train the model using multiprocessing, follow the code below. Please note that this feature is currently only available on the Oxford-BBC LRW Dataset.

``` bash
# Assuming you're in the source folder of the Oxford-BBC LRW Dataset, run either of the following.

# Train the model using stacked RGB tensors with multiprocessing.
python run_mp_baseline.py

# Train the model using stacked RGB optical flow tensors with multiprocessing.
python run_mp_rgb_optical_flow.py
```

## Acknowledgments

This repository uses the [VidTIMIT Audio-Video Dataset](http://conradsanderson.id.au/vidtimit/). Additionally, this repository is made to use the [Oxford-BBC LRW Dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html).
