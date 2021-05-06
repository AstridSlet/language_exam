# ! /usr/bin/python

import re #regex
import string #regex
from pathlib import Path # for fetching path 
import pandas as pd # for creating df 
from collections import Counter # for counting unique collocates
import numpy as np 
import string

# define path to fetch files
data_path = os.path.join("data")

# define output path
outpath = os.path.join("output","collocates_file.csv")

# define tokenizer function (and convert to lower case)
def tokenize(input_text):
    # Split on any non-alphanumeric character
    tokenizer = re.compile(r"\W+")
    # Tokenize
    token_list = tokenizer.split(input_text)
    # convert to lower case
    words_clean = [word.lower() for word in token_list]
    return words_clean


# make function for calculating how many times a collocate appears in text w.o. the keyword (O21)
def collo_function(collocate, raw_freqenzy, text):
    #count how many times collocate appears in total
    number_appearances = text.count(collocate)
    # calculate O12
    O12 = number_appearances - raw_freqenzy
    # calculate O21
    O21 = number_appearances - raw_freqenzy
    return O12, O21


# create function for getting the collocates of keyword and calculate rawfreq and MI for each collocate
def big_function(filepath, keyword, window_size):
    # create empty lists for all texts, text windows (collocates) and MI scores
    all_texts_list = []
    window_words = []
    MI_list = []
    for filepath in Path(filepath).glob("*.txt"):
        with open(filepath, "r", encoding = "utf-8") as file:
            # save indices of all keywords
            indice_list = []
            loaded_text = file.read() # load text file
            clean_text = tokenize(loaded_text) # clean text w tokenize function
            all_texts_list.extend(clean_text) # combine all loaded texts 
            # count how many times in total keyword appears in text
            keyword_app_total = all_texts_list.count(keyword)
            # find keyword indices in all_texts 
            for i, j in enumerate(all_texts_list):
                if j == keyword:
                    index = i
                    # save indices in list
                    indice_list.append(index)
    # loop over indice list 
    for index in indice_list:
        # define window with the collocates 
        window_start = max(0, index - window_size)
        window_end = index + window_size
        full_window = all_texts_list[window_start : window_end + 1]
        window_words.extend(full_window)
        # remove keyword from the window
        window_words.remove(keyword)
        # find unique collocates
    unique_collocates_dict = Counter(window_words)
    # save collocates and their raw_freq in df 
    df = pd.DataFrame.from_dict(unique_collocates_dict, orient="index", columns=["raw_frequency"])
    # fix the df so the collocate name is not the index collumn
    df["collocate"] = df.index
    df.reset_index(drop=True, inplace=True)
    # loop through the df with the unique collocates and their raw freq - and calculate the MI
    for a, b in df.itertuples(index=False):
                       raw_freq = a
                       collocate = b
                       O12, O21 = collo_function(collocate = collocate, raw_freqenzy = raw_freq, text = all_texts_list)
                       O11 = raw_freq
                       N = len(all_texts_list)
                       R1 = O11 + O12
                       C1 = O11 + O21
                       E11 = R1 * C1/N
                       MI = np.log(O11/E11)
                       MI_list.append(MI)
    df["MI"] = MI_list
    # add MI to df 
    df = df.sort_values("MI", ascending = False) 
    # save df
    df.to_csv(outpath, index=False)                   


# check if the funciton works
big_function(filepath = data_path, keyword = "love", window_size = 3)


