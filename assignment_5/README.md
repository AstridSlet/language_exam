
# Assignment 5: Applying (un) supervised machine learning to text data

## Project description 

For this task, you pick your own dataset to study. When you've chosen the data, you choose one of two tasks to perform with the data (one of them is a supervised learning task; the other is unsupervised). I chose the following task: 

* Train an LDA model on your data to extract structured information that can provide insight into your data.

For this you should formulate a short research statement explaining why you have chosen this dataset and what you hope to investigate. 


## Methods
Research question:

* Which topics have played a significant role in the debate on vaccinations in the past years?

I chose this topic, as I thought it would be nice to develop a pipeline that would enable investigation of the discourse related to a topic on twitter and how it changes over time. Additionally, I found vaccinations to be a highly relevant topic. To investigate this question, I am using a data set from Kaggle, which consists in all tweets containing the string 'vaccination' from the year 2006 to 30th of November 2019. The data set can be found here: https://www.kaggle.com/keplaxo/twitter-vaccination-dataset

## Usage
### Data
If you want to redo this analysis you need to unzip the data file 'vaccination2.csv' in the data folder after cloning this repository. This file is a subset (~90.000 tweets from between 20-06-2019 and 03-10-2019) of the full data set from Kaggle, and both files can be downloaded from the Kaggle link above. If you wish to run the analysis on the full data set from Kaggle, you simply download the full data set and place it in the data folder and add the argument -i < name-of-data-file.> when running the LDA script from command line. For computational reasons the results of this analysis is obtained from the subset data. 

### Workflow
To investigate the research question, I used unsupervised machine learning (latent Dirichlet allocation). In order to run the analysis, I also created a script for preprocessing twitter data.
There are therefore two scripts at this repo:
* preproc.py (for preprocessing)
    * arguments:
        *  "--infile", required = False, type = str, default = "vaccination2.csv", help = "Path to input images"
        *  "--outfile", required = False, type = str, default = "prepro_subset.csv ", help = "Output filename of preprocessed data"
* LDA_model.py (for topic modelling)
    * arguments:
        * "--infile", required = False, type = str, default  = "prepro_subset.csv", help="Input filename (file needs to be placed in the data folder)"
        * "--threshold", required = False, type = int, default  = 50, help = "Threshold value for the bigram and trigram models"
        * "--stepsize", required = False, type = int,default  = 5, help = "Stepsize when determining the optimal number of topics for the model"

If you have successfully cloned this repository and created the virtual environment lang_venv you can run the preprocessing script from command line with:

```
$ cd language_exam
$ source lang_venv/bin/activate
$ cd assignment_5/src
$ python preproc.py

```
This will produce a preprocessed version of the sample data (vaccination2.csv) called prepro_subset.csv, which will also be placed in the data folder. You can also choose to skip this step and simply unzip prepro_subset.csv.zip in the data folder. 

You can now run the topic modelling on this preprocessed data:

```
$ cd assignment_5/src
$ python -W ignore::DeprecationWarning LDA_model.py
```

The script initially tries to find the optimal number of topics based on coherence score of the topics. The model therefore outputs to the output folder:
1. A visualization the coherence scores when the model has different number of topics (out/topic_number_plot.jpg).
2. A textfile with the coherence scores for the different number of topics (out/topic_number_coherence_val.txt).

The script automatically chooses the number of topics which yields the highest coherence score and trains a model. This model training will produce:

1. A plot of the prevalence of the different topics over time for the trained model (out/topics_over_time.jpg).
2.  A textfile (final_output.txt) with the number of topics, the coherence and perplexity scores of the model, and the features of the different topics.


## Discussion of results
Answer to research question:
* In the textfile final_output.txt it seems that 10 dominant topics have been prevalent on the topic of vaccines in the past years (see also out/topic_number_plot.jpg). The three most prevalent topics when looking at the plot that shows topic prevalence over time (out/topics_over_time.jpg) are topics 6, 3 and 0.
* From the textfile final_output.txt one can get the most prevalent features of these three topics.
* Topic 3 seems to be a bit broad with both words “child", "flu", "school" and “risk”, “need but also the terms “rabie” and "dog".
* Topic 6 is also a bit broad with words like “child", "people", “get", “rate", “measle", "disease and “health" 
Interestingly both topic 3 and 6 include the word “child” suggesting that people tweet a lot about children and vaccinations.  
* Topic 0 seems to be related to the HPV vaccine with words like "hpv", "woman", "girl", “cervical”, “cancer”, "infection" and “prevent" but also include the words “boy” and “man”.

* When looking at the plot (topics_over_time.jpg) it seems that topics 6 (pink line) and 3 are (red line) are relative prevalent in the discussion at all times. And then topic no 0 (blue line) is generally less prevalent but there are some distinct spikes in this topic at some points.

The coherence score of a topic is the average of the distances between words. In this script this measure is used to determine the best model, but as can be seen in topic_number_coherence_val.txt
in the out folder, the coherence score of different number of topics can be relatively similar and sometimes it can be better to have fewer topics to avoid having the same keywords in different topics. It is thus problematic to just use coherence score as the only means of determining the optimal number of topics. 

As e.g. topics 3 and 6 do not seem to be clearly separated it seems that this model can still use some tweaking. One possible thing to look at is to make the step size more fine grained. The default model tries to have between 5 and 30 topics with a step size of 5, and though it substantially increases computation time, this step size could be made as small as 1. 

Another thing that might make it harder for the model to localize topics is the preprocessing of the data. In the preproc.py removes quite many things such as special characters. This e.g. means that “it’s” will become “it”  “s”. Additionally, the preproc.py script uses re to remove a lot of emojis which entails that some of the semantics of the original text is removed. In a better program it would e.g. be possible to incorporate the semantics of emojis in the text that is fed into the LDA model.  

