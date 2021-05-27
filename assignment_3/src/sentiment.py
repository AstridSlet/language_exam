#!/usr/bin/env python

"""
Calculating mean sentiment scores over a set time periods. 

Parameters:
    infile: str <path-to-images>
    batch_size: int <batch-size-doc>
    
Usage:
    sentiment.py --batch_size <batch-size-doc>
Example:
    $ python sentiment.py --batch_size 300
"""


# load dependencies
from pathlib import Path 
import pandas as pd 
import numpy as np 
import os
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import matplotlib.pyplot as plt
import argparse

# create plot function
def plot_func(sentiment_df, window_size):
    # make rolling mean
    smooth_sent = sentiment_df.rolling(window_size).mean()
    # get the dates for the x-xis
    x = sentiment_df["date"]
    # create figure
    plt.figure()
    # plot the data
    plt.plot(x, smooth_sent, label="sentiment scores") 
    # title of plot
    plt.title(f"Sentiment scores: {window_size} days rolling average")
    # labelling x-axis
    plt.xlabel("Date")
    # labelling y-axis
    plt.ylabel("Sentiment")
    # rotate x-axis labels
    plt.xticks(rotation=40)
    # add legend
    plt.legend()
    # save figure 
    plt.savefig(os.path.join("..", "output", f"{window_size}days_sentiment.png"), bbox_inches='tight')


# define main function
def main():
    
    # initialise argumentparser
    ap = argparse.ArgumentParser()
    
    # define arguments
    ap.add_argument("-i", 
                    "--infile", 
                    type = str, 
                    required=False, 
                    help="Input filename", 
                    default="abcnews-date-text.csv")
    ap.add_argument("-b", 
                    "--batch_size", 
                    type = int, 
                    required=False, 
                    help="Batch size for loading data into spacy docs.", 
                    default = 500)


    # parse arguments to args
    args = vars(ap.parse_args())  
    
    # load data
    data = pd.read_csv(os.path.join("..", "data", args["infile"]))
    
    # make publish date date format 
    data["publish_date"] = pd.to_datetime(data["publish_date"], format="%Y%m%d")
        
    #initialise spaCyTextBlob in nlp pipeline
    nlp = spacy.load("en_core_web_sm")
    spacy_text_blob = SpacyTextBlob()
    nlp.add_pipe(spacy_text_blob)
    
    # create lists for storing polarity scores 
    sentiment_list = []
    mean_sent_list = []
    date_list = []

    # calculate sentiment scores 
    for doc in nlp.pipe(data["headline_text"], batch_size=500):
        print("[INFO] Loading data batch...")
        sentiment_list.append(doc._.sentiment.polarity)

    # append scores to dataframe 
    data["sent_scores"] = sentiment_list

    # create mean for each date 
    for name, group in data.groupby("publish_date"):
        print("[INFO] Creating mean scores...")
        date_list.append(name)
        mean_sent_list.append(group["sent_scores"].mean())

    # make df
    df = pd.DataFrame(zip(date_list, mean_sent_list), 
                      columns =["date", "mean_sentiment"])
    
    # make plots and save them
    plot_func(sentiment_df = df, window_size = 7)
    plot_func(sentiment_df = df, window_size = 30)

    
# Define behaviour when script is called from command line
if __name__=="__main__":
    main()
          
          
