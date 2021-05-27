#!/usr/bin/env python

"""
Automatic poem generator based on tensorflow.keras and 
Parameters:
    theme1: str <theme-one>
    theme2: str <theme-two>
    lines: int <num-lines-poem>
    n_words: int <nwords-each-line>
    epochs: int <num-epochs-training>
    
Usage:
    poem-generator.py --theme1 <theme-one> --theme2 <theme-two> --lines <num-lines-poem> --n_words <nwords-each-line>
Example:
    $ python poem-generator.py --theme1 love --theme2 football --lines 8 --n_words 5 --epochs 10
"""


# load dependencies
import sys, os
sys.path.append(os.path.join(".."))
import argparse

# load utility functions
from utils.poem_utils import get_initators, load_poems, tokenize_poems, create_sequences, pad_seqs, split_XY, train_model, model_structure_viz, plot_train_hist, generate_line, get_poem, save_poem

# define main function 
def main():
 
    # initialise argumentparser
    ap = argparse.ArgumentParser()
    
    # define arguments
    ap.add_argument("-t1", 
                    "--theme1", 
                    type = str, 
                    required=False, 
                    help="Theme 1 of your poem.", 
                    default="nature")
    ap.add_argument("-t2", 
                    "--theme2", 
                    type = str, 
                    required=False, 
                    help="Theme 2 of your poem.", 
                    default="dance")
    ap.add_argument("-l", 
                    "--lines", 
                    type = int, 
                    required=False, 
                    help="Number of lines in your poem.", 
                    default = 12)
    ap.add_argument("-w", 
                    "--n_words", 
                    type=int, 
                    required=False, 
                    help="Number of words in each line of your poem.", 
                    default = 6)
    ap.add_argument("-np", 
                    "--n_poems", 
                    type = int, 
                    required=False, 
                    help="Number of poems you want.", 
                    default = 5)
    ap.add_argument("-e", 
                    "--epochs", 
                    type = int, 
                    required=False, 
                    help="Number of epochs for model training.", 
                    default = 10)

    # parse arguments to args
    args = vars(ap.parse_args())
    
    # load poems of themes 1+2 
    alltexts = load_poems(args["theme1"], args["theme2"])
    print(f"[INFO] Loaded {len(alltexts)} poems for model training.")
   
    # tokenize poems
    corpus, total_words, tokenizer = tokenize_poems(alltexts)

    # create sequences
    sequences = create_sequences(tokenizer, corpus)
    
    # pad sequences
    sequences_padded, max_seq_length = pad_seqs(sequences)
    
    # split sequences in X and Y
    X, Y = split_XY(sequences_padded, total_words)
    
    # train model
    model = train_model(X, Y, max_seq_length = max_seq_length, total_words = total_words, epochs = args["epochs"])
    
    # save model architecture
    model_structure_viz(model, args["theme1"], args["theme2"])
    
    # plot training history
    plot_train_hist(model, epochs = args["epochs"], theme1 = args["theme1"], theme2 = args["theme2"])
    
    
    # create n_poems
    for i in range(1, args["n_poems"]+1):
        
        # sample an iniator text for n_lines
        initiators_list = get_initators(filepath = os.path.join ("..", "utils", "initiaters.txt"), n_lines = args["lines"])
        
        # generate poem
        the_poem = get_poem(model, initiators_list, args["n_words"], tokenizer, max_seq_length)
        
        # save poem
        save_poem(text_lines = the_poem, theme1 = args["theme1"], theme2 = args["theme2"], number = str(i))
    
# define behaviour when called from command line
if __name__=="__main__":
    main()    
    
    
    
    
    
    
    
    
