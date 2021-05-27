#!/usr/bin/env python

"""
Utility functions used for automatic poem-generation.
"""

# load dependencies
import os,sys
sys.path.append(os.path.join(".."))
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random

# tf tools
import tensorflow 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam




# define functions 

def load_poems(theme1, theme2):
    """
    Load all poems from the two user defined themes. 
    """
    # make list of themes
    themes_list = [theme1, theme2]
    
    # list for storing theme poems
    alltexts = []
    
    # load poems 
    for theme in themes_list:
        inpath = os.path.join("..", "data", "poems", "topics", f"{theme}")
        for textfile in Path(inpath).glob("*txt"):
            with open(textfile, "r", encoding = "utf-8") as file:
                alltexts.append(file.read())
    return alltexts
    

    
    
def tokenize_poems(text_list):
    """
    For all poems: split text, make lower case and combine words in one big corpus.
    Tokenize combined corpus and return corpus and no. of words in corpus (maximum 5000 words).  
    """
    # for combining poems
    corpus = []
    
    # define tokenizer
    tokenizer = Tokenizer(num_words = 5000)
    
    # split text, make lower case
    for text in text_list:
        corpus.extend(text.lower().split("\n"))
        
    # tokenize corpus
    tokenizer.fit_on_texts(corpus)

    # count words in corpus (+1 because of reserved 0index)
    total_words = len(tokenizer.word_index)+1
    return corpus, total_words, tokenizer


def create_sequences(tokenizer, corpus):
    """
    For all individual lines of text in corpus: 
    * make that line into a sequence.
    * make that sequence into n-gram-sequence.
    Return n-gram sequences. 
    """
    # for storing sequences
    input_sequences = []

    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences


def pad_seqs(input_sequences):
    """
    Padding all sequences to a max_length of {max_seq_length} tokens.
    """
    
    # get maximum sequence length
    max_seq_length = max((len(x) for x in input_sequences))
    
    # padd sequences to maxlength
    sequences_padded = np.array(tensorflow.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                              maxlen=max_seq_length,
                                              padding='pre'))
    return sequences_padded, max_seq_length 



def split_XY(sequences_padded, total_words):
    """
    Split padded sequenzes into X and Y. 
    """
    # in all rows get all indices but the last one
    X = sequences_padded[:, :-1]

    # save the last indices spots as y-labels
    Y = sequences_padded[:, -1]
    
    # make Y labels categorical
    Y = tensorflow.keras.utils.to_categorical(Y, num_classes=total_words, dtype='uint8')
    return X, Y
    

    
    
def train_model(X, Y, max_seq_length, total_words, epochs):
    """
    Train CNN with embedding layer and an LSTM layer. 
    Return fit model object. 
    """
    model = Sequential()
    model.add(Embedding(total_words, 240, input_length=max_seq_length-1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(total_words, activation='softmax'))

    adam = Adam(lr=0.01)

    model.compile(loss="categorical_crossentropy",
                 optimizer=adam,
                 metrics=['accuracy'])
    
    # train model
    model.fit(X, Y, epochs=epochs, verbose=1)
    return model

def model_structure_viz(model, theme1, theme2):
    """
    Save visualization of model architecture.
    """
    impath = os.path.join("..","output",f"{theme1}_{theme2}_model_architecture.png")
    tensorflow.keras.utils.plot_model(model, to_file = impath, show_shapes = True)


def plot_train_hist(model, epochs, theme1, theme2):
    """
    Plot network training history. 
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), model.history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), model.history.history["accuracy"], label="train_acc")
    plt.title("Training loss/accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join("..","output", f"training_history_{theme1}_{theme2}.png"))

    
    
def get_initators(filepath, n_lines):
    """
    Open text file with iniator words and sample random iniator for each line in the poem. 
    """
    with open(filepath, "r", encoding = "utf-8") as file:
        # save indices of all keywords
        loaded_text = file.read() # load text file
        lines = loaded_text.splitlines() # seperate initiator lines
        initiators_list = list(random.sample(lines, n_lines)) # sample random initators
    return initiators_list



def generate_line(model_fit, seed_text, next_words, tokenizer, max_seq_length):
    """
    Take a iniator text bit and generate n words to finish the sentence. 
    Return text line. 
    """
    for _ in range(next_words): # initiate iteration to predict 20 words
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tensorflow.keras.preprocessing.sequence.pad_sequences([token_list], 
                                   maxlen=max_seq_length-1,
                                  padding="pre")
        # predict the next word 
        predicted = np.argmax(model_fit.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text



def get_poem(model_fit, initiators_list, n_words, tokenizer, max_seq_length):
    """
    For all the sentence initators create a textline. 
    Return list of poem-lines. 
    """
    the_poem = []
    for initiator in initiators_list:
        the_poem.append(generate_line(model_fit, str(initiator), n_words, tokenizer, max_seq_length))
    return the_poem



def save_poem(text_lines, theme1, theme2, number):
    """
    Write poem to txt file. 
    """
    # define output name
    outfile = os.path.join("..", "output", f"Poem_of_{theme1}_and_{theme2}_no_{number}.txt")

    # save poem text
    with open(outfile, "a", encoding="utf-8") as results:
        results.write(f"A poem of {theme1} and {theme2}: \n")
        for line in text_lines:
            results.write(line+"\n")
    return(print("[INFO] NEW POEM GENERATED!"))







