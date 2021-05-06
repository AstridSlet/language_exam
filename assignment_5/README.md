## Repo description 

__Assignment description__

Research question: 

* Which topics have played a significant role in the debate on vaccinations in the past years?

To investigate this question I have downloaded a data set from kaggle which is constituted by all tweets containing the string 'vaccination' from the year 2006 to 30th of November 2019. The data set, which is also placed in the data folder, can be found here: https://www.kaggle.com/keplaxo/twitter-vaccination-dataset

If you want to redo this analysis you need to unzip the data subset called 'vaccination2.csv' in the data folder of this repo - or download the full data set from kaggle set and place it in the folder and add the argument -i "name_of_file.csv" when running the main script at this repo.  


__Work flow__

To investigate the research question I used unsupervised machine learning (latent dirichlet allocation). In order to run the analysis I also created a script for preprocessing twitter data.

There are therefore two scripts at this repo: 
* preproc.py (for preprocessing)
    * arguments:
        *  -i : input file name
        *   -oc : output file name 
* LDA_model.py (for topic modelling) 
    * arguments:
        *  -i : input file name
        *  -t : threshold value for the bigram and trigram models
        *  -s : stepsize used for determining the optimal number of topics for the LDA model. The lower value, the more fine grained a search, default = 5


If you have successfully cloned this repository and created the virtual environment lang_venv you can run the preprocessing script from command line with:
`
$ cd language_exam
$ source lang_venv/bin/activate
$ cd assignment_5
$ python preproc.py`

This will produce a preprocessed version of the sample data (vaccination2.csv) called prepro_subset.csv, which will also be placed in the data folder. You can also choose to skip this step and simply unzip prepro_subset.csv.zip in the data folder. 

You can now run the topic modelling on this preprocessed data set by running: 

`$ python -W ignore::DeprecationWarning LDA_model.py`

The script initially tries to find the optimal number of topics based on coherence score. The model therefore outputs to the out folder:

1. A visualization the coherence scores when the model has different number of topics (out/topic_number_plot.jpg).
2. A textfile with the coherence scores for the different number of topics (out/topic_number_coherence_val.txt).
The script automatically chooses the number of topics which yields the highest coherence score and trains a model. This model training will produce:
1. A plot of the different topics prevalence over time (out/topics_over_time.jpg).
2. A textfile (final_output.txt) with the number of topics, the coherence and perplexity scores of the model, and the features of the different topics.

