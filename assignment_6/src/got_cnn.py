#!/usr/bin/env python

"""
Predicting season of GOT from sentences from the manuscripts using tensorflow.keras.
Parameters:
    infile: str <path-to-images>
    test_size: float <test-size-split>
    chunk_size: int <input-chunk-size>
    epochs: int <num-epochs-training>
    batchsize: int <batchsize-training>
    embed_dim: int <glove-embed-dim>
    path_glove_embedding: str <path-glove-embedding>
    
Usage:
    got_cnn.py --epochs <num-epochs-training>
Example:
    $ python got_cnn.py --epochs 15
"""

# load dependencies
import os, sys
sys.path.append(os.path.join(".."))
import pandas as pd
import argparse
from tensorflow.keras.utils import plot_model



# utility functions 
from utils.got_utils import split_data, one_hot, base_model, create_embedding_matrix, prepro_cnn, train_network, plot_train_hist, evaluate_model, balance_data


def main():
    print("\n[INFO] Initialising analysis...")
    
    # initialise argumentparser
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-i", 
                    "--infile", 
                    required=False, 
                    help="Input path, training data", 
                    type = str, 
                    default="Game_of_Thrones_Script.csv")
    ap.add_argument("-e", 
                    "--epochs", 
                    required=False, 
                    help="Train/test split input data", 
                    type=int, 
                    default=10)
    ap.add_argument("-b", 
                    "--batchsize", 
                    required=False, 
                    help="Batchsize in model training", 
                    type=int, 
                    default = 32)
    ap.add_argument("-ed", 
                    "--embed_dim", 
                    required=False, 
                    help="Dimensions for storing pretrained word embeddings from glove", 
                    type=int, 
                    default = 50)
    ap.add_argument("-ge", 
                    "--path_glove_embedding", 
                    required=False, 
                    help="Path to the pretrained word embedding to use from glove", 
                    type=str, 
                    default = os.path.join("..","glove","glove.6B.50d.txt"))


    # parse arguments to args
    args = vars(ap.parse_args())
    
    # read in data
    df = pd.read_csv(os.path.join("..", "data", args["infile"]))
    
    # df = balance_data(dataframe = df, n=1466, label='Season')

    # split data into train and test set
    X_train, X_test, y_train, y_test, label_names = split_data(df)

    print("\n[INFO] Train baseline model...")
    # train baseline model 
    y_pred_basemodel = base_model(X_train, X_test, y_train)
    
    
    # make labels onehot encoding for the cnn
    trainY = one_hot(y_train)
    testY = one_hot(y_test)
    
    # make data ready for the cnn  
    X_train_pad, X_test_pad, vocab_size, embedding_matrix, maxlen = prepro_cnn(X_train, X_test, args["embed_dim"], args["path_glove_embedding"])
    
    print("\n[INFO] Train CNN model...")
    # train cnn network and get training history
    cnn_model, y_pred_cnn, cnn_history = train_network(X_train_pad, trainY, X_test_pad, testY, args["batchsize"], args["epochs"], vocab_size, embedding_matrix, maxlen, args["embed_dim"])
    
    # save visualization of the model architecture
    plot_model(cnn_model, to_file = os.path.join("..", "output", "model_architecture.png"), show_shapes = True)

    
    # create visualization of training history
    plot_train_hist(cnn_history, args["epochs"])
    
    
    #### EVALUATE MODELS ####
    print("\n[INFO] Evaluate models...")
    # create classification report basemodel 
    evaluate_model(y_test, y_pred_basemodel, label_names, "clf_basemodel")
    
    # create classification report CNN
    evaluate_model(testY.argmax(axis = 1), y_pred_cnn.argmax(axis = 1), label_names, "clf_CNN")
    
    print("\n[INFO] ALL DONE!")

# define behaviour from command line 
if __name__=="__main__":
    main()