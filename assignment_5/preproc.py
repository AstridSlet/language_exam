#!/usr/bin/python
"""
Cleaning twitter data using re

"""

# standard library
import sys,os

# text cleaning
import re

# for storing cleaned data
import pandas as pd

# for parsing arguments
import argparse


# define emoji patterns 
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"  # below is specific emojis that weren't included in the above
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\U0001f923"
        u'\U0001f914'
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)

# define cleaning function no1
def clean_text1(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove "add mentions" ie @ followed by any upper/lower case letter/any number 
    text = re.sub(r'#', '', text)              # remove hashtag
    text = re.sub(r'RT[\s]+', '', text)        # remove RT's (retweets) followed by one or more wide spaces
    text = re.sub(r'https?:\/\/\S+', '', text) # remove hyperlinks
    text = re.sub(r'pic\.twitter\.com.[A-Za-z0-9]{10}', '', text) # remove url's for images (e.g. pic.twitter.com/hnXii4gXrH)
    text = emoji_pattern.sub(r'', text)        # remove emoji_pattern 
    return text    

# define clean function no2
def clean_text2(text):
    text = text.lower() # text make lower case 
    text = re.sub(r'[^a-z\s]', ' ', text) # remove special characters NB this means that e.g. "it's" will become "it s"
    text = re.sub(r'vaccination', ' ', text) 
    text = re.sub(r'vaccine', ' ', text)     
    text = re.sub(r'vaccinate', ' ', text)
    text = re.sub(r'vaccinate', ' ', text)
    return text


def main():
    # initialise argumentparser
    ap = argparse.ArgumentParser()
    # define arguments
    ap.add_argument("-i", 
                    "--infile", 
                    required=False, 
                    help="Input filename", 
                    type = str, 
                    default="vaccination2.csv", 
                    help="Input filename")
    ap.add_argument("-oc", 
                    "--outfile", 
                    required=False, 
                    type = str,
                    default="prepro_subset.csv",
                    help="Output filename of preprocessed data")

    # parse arguments to args
    args = vars(ap.parse_args())
    
    # get input filepath
    input_file = os.path.join("data", args["infile"])    
      
    # define collumns names we want to use
    col_list = ["id","date","tweet"]
    
    # read data 
    data = pd.read_csv(input_file, usecols=col_list, dtype={"id": "object", "conversation_id": "object", "created_at": "object", "date": "object","tweet": "object"}) 
    
    # apply cleaning functions to tweets
    data["tweet"] = data["tweet"].apply(clean_text1)
    data["tweet"] = data["tweet"].apply(clean_text2)

    
    # get output filepath
    output_file = os.path.join("data", args["outfile"])
    
    # save df as csv in the data folder
    data.to_csv(output_file, index=True, encoding="utf-8")
    
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
