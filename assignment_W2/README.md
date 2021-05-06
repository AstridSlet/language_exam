## Repository description 

The files found at this repo relates to the assignment of W2 (see below). To run the script download this folder and install the necessary packages found at the top of in the script:
re, string, pathlib, pandas, collections, numpy, string

You can then run the script __collocate_script.py__ with the sample data ( 4 .txt files placed in the 'data' folder)
When the script has been run it should produce a .csv file (which will be placed in the 'output' folder) which contains the collocates, their raw_frequency and MI.

## Assignment description 
__String processing with Python__

Using a text corpus found on the cds-language GitHub repo or a corpus of your own found on a site such as Kaggle, write a Python script which calculates collocates for a specific keyword.

The script should take a directory of text files, a keyword, and a window size (number of words) as input parameters, and a file called out/{filename}.py
These parameters can be defined in the script itself
Find out how often each word collocates with the target across the corpus
Use this to calculate mutual information between the target word and all collocates across the corpus
Save result as a single file consisting of four columns: collocate, raw_frequency, MI
