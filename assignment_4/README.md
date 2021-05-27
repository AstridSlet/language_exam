
# Assignment 4: Network analysis

## Project description 

This command-line tool will take a given dataset and perform simple network analysis. In particular, it will build networks based on entities appearing together in the same documents, like we did in class.
* Your script should be able to be run from the command line
* It should take any weighted edgelist as an input, providing that edgelist is saved as a CSV with the column headers "nodeA", "nodeB"
* For any given weighted edgelist given as an input, your script should be used to create a network visualization, which will be saved in a folder called viz.
* It should also create a data frame showing the degree, betweenness, and eigenvector centrality for each node. It should save this as a CSV in a folder called output.



## Methods
For this assignment a single command line script was developed. The script takes as its input a weighted edgelist placed in the data folder, which has been constructed using an open source data set from Kaggle: https://www.kaggle.com/ekatra/fake-or-real-news?select=fake_or_real_news.csv 

The original data set includes approximately 6000 news stories of which half are real and half fake news. The first column in the edge list constructed from the original data set (nodeA) holds all persons mentioned in data set. These names have been extracted with named entity recognition using spaCy. For all nodes A, it has been calculated how many times they occur with another name (nodeB). This co-occurrence of the nodes is the weight of the connection between these two nodes, and this weight is stored in the third column (weight).   

The command line script loads the weighted edge list from the data folder and filters out the node connections that have a weight lower than the specified weight cutoff. Thus the weaker node connections are excluded from the analysis. The default weight cutoff of the script is 500 but can be changed with the command line argument -w as specified below. The script makes then make the filtered edge list into a graph object using the package networkx and saves visualization of the graph to the folder ‘viz’. Subsequently, the script calculates the degree, betweenness, and eigenvector centrality values for the node pairs – also using the networkx package - and saves these values to the output folder. 


## Usage
For this assignment a single command line scripts was created: 

* network.py
    * "--infile", type = str, required=False, help="Input filename", default="weighted_edgelist.csv"
    * "--outfile_plot", type = str, required=False, help="Output plot filename", default = os.path.join("..", "viz", "sentiment_over_time.png")
    * "--outfile_csv", type = str, required=False, help="Output csv filename", default = os.path.join("..", "output", "degree_betweenness_eigenvector.csv")
    * "--weight_cutoff", type=int, required=False, help "Weight cutoff for determining which edges to include in the graph and when calculating degree, betweenness and eigenvector values." default = 500


If you have successfully cloned this repository and created the virtual environment lang_venv you can run the script from command line with:

```
$ cd language_exam
$ source lang_venv/bin/activate
$ cd assignment_4/src
$ python network.py
```

## Discussion of results
The degree of a node measures how many other nodes the node connected to. In the case of this data set, the degree expresses how many other persons/mentions of persons, the person in question (nodeA) is connected to (Brede, 2012). From the output csv it can e.g. be seen, that the nodes ‘Clinton’ and ‘Bush’ are highly connected to mentions of other persons. 

The eigen vector centrality measures the centrality for a node based on the centrality of its neighbors, thus, if a node has many well connected neighbor nodes it will have a high eigen vector centrality (Brede, 2012). The ‘Donald Trump node’ e.g. almost has just as well connected neighbors (eigenvector = 0.26) as the ‘Bush’ node (eigenvector = 0.28), while it generally has fewer neighbor nodes than the bush node (based on degree). To put things to a head one could argue that the Trump node has ‘few but powerful friends’ in the network.    

The last measure calculated from the script is the betweenness degree: For two given nodes in a network there will be a ‘shortest path’ i.e. the path between the nodes that has the lowest number of vertices. The betweenness centrality of a node thus expresses how many of these ‘shortest paths’ that run through the node. A node with a higher betweenness centrality value will have more ‘control’ over the network, because more information will pass through that node (Brede, 2012). From the output one can see that apart from having the highest degree and eigenvector values the ‘Clinton’ node also has the highest betweenness centrality score, which makes sense, as it has so many connections compared to the other nodes of the network.

Generally, it is important to note that it is a limitation of this small analysis, that there are so many versions of the same name. From the output csv it can be seen that Hillary Clinton goes by many names in this data set though all the names relate to the same person, which skews the interpretation of how important this name is in relation to the other nodes. A natural development of this analysis would be to make sure that the data included as few as possible versions of the same name.  

### References: 
Brede, M. (2012). Networks—An Introduction. Mark EJ Newman.(2010, Oxford University Press.) ISBN-978-0-19-920665-0. MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info
