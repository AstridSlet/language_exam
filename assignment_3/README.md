
# Assignment 3: Sentiment analysis

## Project description 

Download the following CSV file from Kaggle:
https://www.kaggle.com/therohk/million-headlines

This is a dataset of over a million headlines taken from the Australian news source ABC (Start Date: 2003-02-19; End Date: 2020-12-31).
* Calculate the sentiment score for every headline in the data. You can do this using the spaCyTextBlob approach that we covered in class or any other dictionary-based approach in Python.
* Create and save a plot of sentiment over time with a 1-week rolling average
* Create and save a plot of sentiment over time with a 1-month rolling average
* Make sure that you have clear values on the x-axis and that you include the following: a plot title; labels for the x and y axes; and a legend for the plot


## Methods
For this assignment a single command line script was developed. The script loads the csv from the data folder and batches the headlines together in a Spacy doc object in batches specified with the command line argument batch_size which defaults to 500. The headlines are grouped by publishing date, then an average sentiment score is calculated for each day and from this the script calculates the 7 day- and the 30-day rolling means and saves a plot for each in the output folder. 


## Usage
For this assignment a single command line scripts was created: 

* sentiment.py
    * "--infile", type = str, required=False, help="Input filename", default="abcnews-date-text.csv"
    * "--batch_size", type = int, required=False, help="Batch size for loading data into spacy docs.", default = 500

If you have successfully cloned this repository and created the virtual environment lang_venv you can run the script from command line with:

```
$ cd language_exam
$ source lang_venv/bin/activate
$ cd assignment_3/src
$ python sentiment.py

```

When the script is run it produces two plots to the output folder: a plot of the sentiment scores with a 7-day rolling mean average and a plot of the sentiment scores with a 30-day rolling mean average.


## Discussion of results
The rolling average is a calculation to analyze data points by creating a series of averages of different subsets of the full data set to smooth out smaller fluctuations in the data. It is hard to say anything about the bigger tendencies when simply computing the rolling average over 7 days. In this plot you see a lot of changes in the graph, but it seems that there has been a peak in sentiment scores between 2014 and 2016. In the plot with the 30-day rolling the same tendencies can be seen but the graph has fewer spikes, as these values are averaged over a more days. 

