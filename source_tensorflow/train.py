import argparse
import json
import os
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam

import keras
from keras.models import model_from_json

# imports the model in model.py by name
from model import BinaryClassifier


def model_fn(model_dir):
    """Load the Tensorflow model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')

    with open(model_info_path, 'rb') as f:
        loaded_model_json = json_file.read()
        json_file.close()
        model_info = model_from_json(loaded_model_json)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    # device = tf.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(model_info['input_features'], model_info['hidden_dim'], model_info['output_dim'],
                             model_info['dropout_factor'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    # with open(model_path, 'rb') as f:
    #    model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    # model.to(device).eval()

    print("Done loading model.")
    return model


# Gets training data in batches from the train.csv file
def get_train_data(training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    np_train = train_data.to_numpy(dtype='float64')
    x_np_train = np_train[1:]
    y_np_train = np_train[:1]

    return x_np_train, y_np_train


# Provided training function
def train(model, train_data, epochs,  optimizer):
    """
    This is the training method that is called by the Tensorflow training script. The parameters
    passed are as follows:
    model        - The Tensorflow model that we wish to train.
    train_loader - The Tensorflow DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training.
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """


    train_data = get_train_data(args.data_dir)
    model.compile(optimizer=optimizer, loss="binarycrossentropy", metrics=["mae"])
    model.fit(train_data[0], train_data[1])



if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job

    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--input_features', type=int, default=2, metavar='N',
                        help='input dimension (default: 10)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='H',
                        help='hidden dimension (default: 100)')
    parser.add_argument('--output_dim', type=int, default=1, metavar='O',
                        help='output dimension (default: 1)')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dropout_factor', type=float, default=0.1, metavar='DF',
                        help='dropout factor (default: 0.1)')

    # args holds all passed-in arguments
    args = parser.parse_args()

    # Load the training data.
    train_data = get_train_data(args.data_dir)

    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = BinaryClassifier(args.input_features, args.hidden_dim, args.output_dim, args.dropout_factor)

    ## TODO: Define an optimizer and loss function for training


    adam_optimizer = Adam(learning_rate=args.learning_rate)
    # Trains the model (given line of code, which calls the above training function)
    train(model, train_data, args.epochs, adam_optimizer)

    ## TODO: complete in the model_info by adding three argument names, the first is given
    # Keep the keys of this dictionary as they are
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim,
            'learning_rate': args.learning_rate,
            'dropout_factor': args.dropout_factor
        }
        model.save(model_info, f)

    ## --- End of your code  --- ##

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        model.save(model.cpu().state_dict(), f)