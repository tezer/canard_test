# Canard_test

This is a simple command-line utility for anaphora resolution in short dialogs. There is a trainer (`train_t5.py`)
and a model runner (`run_model.py`). The file `constants.py`contains the most important settings for training and
for the task.

I used T5 `t5-small` for this task as the most appropriate type of transformer.

The main libraries used in the project are: `transformers`, `pandas`, `torch` and `pytorch_lightning`. The latter makes
the PyTorch-based code more readable.

Here is what the code does for training:

1. loads the data to DataFrame
1. defines a callback that tracks validation loss, and best checkpoint saver to make sure I
   have the best trained model
1. finally, packs everything into the Trainer and fires it off

run_model.py is much simpler:

1. loads the saved model (a checkpoint file)
1. gets rewritten question with resolved anaphora from the model for a user input
1. and writes them to the screen

## Installation

clone the project with `git clone https://github.com/tezer/canard_test

## Usage

Requires python 3.8
`cd canard_test`

For model training run:

`poetry run python canard/train_t5.py` for interactive mode

or

```
poetry run python canard/train_t5.py --input_file=/path/to/your/data/folder
```

where

```
Options:
  --input_file    Specify the input folder with dev.jsom, train.json and test.json files.
  --help          Show this message and exit.
```

To run the model:

`poetry run python canard/run_model.py` for interactive mode

or

```
poetry run python run_model.py --model_file=/path/to/your/best-checkpoint.ckpt
```

where

```
Options:
  --model_file FILENAME   Enter the path to the file with the trained model.
  --help                  Show this message and exit.