#!/usr/bin/env python

"""
Utility functions for convolutional neural network analysis of classical paintings. 
"""

# load dependencies

# data tools
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sklearn tools
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# parsing arguments
import argparse

# tf tools
# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# matplotlib
import matplotlib.pyplot as plt
import pydot
import graphviz



#### Define functions ####
def balance_data(dataframe, n=500, label='Season'):
    """
    Create a balanced sample from imbalanced datasets.
    
    dataframe: 
        Pandas dataframe with a column called 'text' and one called 'label'
    n:         
        Number of samples from each label, defaults to 500
    
    Credits: Ross
    """
    # use pandas to select a random bunch of examples from each label
    out = (dataframe.groupby(label, as_index=False)
            .apply(lambda x: x.sample(n=n))
            .reset_index(drop=True))
    
    return out


def split_data(data):
    """
    Load input filepath, extract sentences, make train/test sets and save label names. 
    
    """ 
    # read data and sort by season
    data = data.sort_values(by="Season", ascending=True)

    # get label names
    label_names = list(data["Season"].unique())

    # get the values in each cell; returns a list
    sentences = data["Sentence"].values
    labels = data["Season"].values   
    
    # train and test split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(sentences, 
                                                        labels, 
                                                        test_size=0.20, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test, label_names



def evaluate_model(testY, predictions, label_names, filename):
    """
    Create and save classification report.
    """
    # create clf report 
    clf_report = classification_report(testY, 
                                       predictions, 
                                       target_names = label_names)
    
    # create df for storing metrics
    df = pd.DataFrame(classification_report(testY,predictions,target_names = label_names,output_dict = True)).transpose().round(decimals=2)
        
    # save classification report    
    df.to_csv(os.path.join("..", "output", f"{filename}.csv"), index = True)
    
    return clf_report



### model training functions 
def base_model(X_train, X_test, y_train):
    """
    Create baseline model and return model predictions. 
    """
    # initialize vectorizer 
    vectorizer = CountVectorizer()
    # vectorize training data 
    X_train_feats = vectorizer.fit_transform(X_train)
    # vectorize test data
    X_test_feats = vectorizer.transform(X_test)

    # create classifier 
    model = LogisticRegression(random_state=42, max_iter=1000)
    # fit model 
    model.fit(X_train_feats, y_train)
    # fit model
    y_pred = model.predict(X_test_feats)
    return y_pred
    
    
    
    
def one_hot(ylabs):
    """
    Take a list of labels (str) and make it into onehot encoding.
    """
    # get values as array
    values = np.array(ylabs)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    ylabs_onehot = onehot_encoder.fit_transform(integer_encoded)
    return ylabs_onehot


def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix
    (ie just a useful way of getting text data into keras)
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    """
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix    



def prepro_cnn(X_train, X_test, embedding_dim, path_to_glove_embedding):
    # tokenize with keras tokenizer 
    tokenizer_k = Tokenizer(num_words=5000)
    # fit to training data
    tokenizer_k.fit_on_texts(X_train)

    # tokenized training and test data
    X_train_toks = tokenizer_k.texts_to_sequences(X_train)
    X_test_toks = tokenizer_k.texts_to_sequences(X_test)

    # overall vocabulary size
    vocab_size = len(tokenizer_k.word_index) + 1  # Adding 1 because of reserved 0 index (just a weird quirk of keras)

    
    # padding the tokens to give them all equal dims 
    # max length for a doc
    maxlen = 200

    # pad training data to maxlen
    X_train_pad = pad_sequences(X_train_toks, 
                                padding='post', # sequences can be padded "pre" or "post"
                                maxlen=maxlen)
    # pad testing data to maxlen
    X_test_pad = pad_sequences(X_test_toks, 
                           padding='post', 
                           maxlen=maxlen)
    
    # create embedding matrix
    embedding_matrix = create_embedding_matrix(f"{path_to_glove_embedding}",
                                           tokenizer_k.word_index, 
                                           embedding_dim)
    
    return X_train_pad, X_test_pad, vocab_size, embedding_matrix, maxlen
    

def train_network(X_train_pad, trainY, X_test_pad, testY, batch_size, epochs, vocab_size, embedding_matrix, maxlen, embedding_dim): 
    """
    Train convolutional neural network and save training history and predictions on the test set. 
    """
    # new model
    model = Sequential()
    # Embedding layer
    model.add(Embedding(vocab_size, 
                        embedding_dim, # the dims of the matrix with the embeddings (ie the word vectors of all words in a doc)
                        weights=[embedding_matrix],  # we've added our pretrained GloVe weights
                        input_length=maxlen, 
                        trainable=False))            # embeddings = pretrained, so trainable=False

    # CONV+ReLU -> MaxPool -> FC+ReLU -> Out
    model.add(Conv1D(128, 5, 
                    activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(10, 
                    activation='relu'))
    model.add(Dense(8, 
                    activation='softmax'))


    # Compile
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # fit model and save training history
    history = model.fit(X_train_pad, trainY,
                    epochs=epochs,
                    verbose=False,
                    validation_data=(X_test_pad, testY),
                    batch_size=batch_size)
    
    y_pred_cnn = model.predict(X_test_pad)
    
    return model, y_pred_cnn, history




def model_structure_viz(model):
    """
    Save visualization of model architecture.
    """
    plot_model(model, to_file = os.path.join("..", "output", "model_architecture.png"), show_shapes = True)
    

def plot_train_hist(H, epochs):
    """
    Plot network training history. 
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("..", "output", "cnn_training_history.png"))