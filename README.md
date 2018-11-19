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

To train the model, follow the code snippet below.

``` bash
# Assuming you're in the repository's home folder, go to src/.
cd src/

# Train the model.
python run_baseline.py
```

## Acknowledgments

This repository uses the [VidTIMIT Audio-Video Dataset](http://conradsanderson.id.au/vidtimit/).
