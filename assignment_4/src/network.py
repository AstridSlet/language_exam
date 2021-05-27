#!/usr/bin/env python

"""
Predicting season of GOT from sentences from the manuscripts using tensorflow.keras.
Parameters:
    infile: str <path-to-images>
    outfile_plot: str <output-plot-name>
    outfile_csv: str <output-csv-name>
    weight_cutoff: float <num-epochs-training>
    
Usage:
    network.py --weight_cutoff <num-epochs-training>
Example:
    $ python network.py --epochs 15
"""


# load dependencies
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz
import argparse
from collections import defaultdict


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
                    default="weighted_edgelist.csv")
    ap.add_argument("-op", 
                    "--outfile_plot", 
                    type = str, 
                    required=False, 
                    help="Output plot filename", 
                    default = os.path.join("..", "viz", "sentiment_over_time.png"))
    ap.add_argument("-oc", 
                    "--outfile_csv", 
                    type = str, 
                    required=False, 
                    help="Output csv filename", 
                    default = os.path.join("..", "output", "degree_betweenness_eigenvector.csv"))
    ap.add_argument("-w", 
                    "--weight_cutoff", 
                    type=int, 
                    required=False, 
                    help="Weight cutoff for determining which edges to include when calculating degree, betweenness and eigenvector values.", 
                    default = 500)

    # parse arguments to args
    args = vars(ap.parse_args())  
    
    # make file into df
    data = pd.read_csv(os.path.join("..", "data", args["infile"]))
        
    # filter out just the ones with weights higher than the chosen weight cutoff
    filtered_df = data[data["weights"] > args["weight_cutoff"]]

    # create a graph object 
    graph_object = nx.from_pandas_edgelist(filtered_df, "nodeA", "nodeB", ["weights"])  
     
    # plot the graph object 
    pos = nx.nx_agraph.graphviz_layout(graph_object, prog="neato")
    nx.draw(graph_object, pos, with_labels=True, node_size=20, font_size=10)

    # save plot  
    plt.savefig(os.path.join("..", "viz", args["outfile_plot"]), dpi=300, bbox_inches='tight')
    
    
    #### get centrality betweenness and eigenvector values ###
    
    # calculate metrics
    degree = nx.degree_centrality(graph_object)
    betweenness = nx.betweenness_centrality(graph_object)
    eigenvector = nx.eigenvector_centrality(graph_object)
    
    # make into dataframe
    df = pd.DataFrame({
        "degree":round(pd.Series(degree), 2),
        "betweenness":round(pd.Series(betweenness), 2),
        "eigenvector":round(pd.Series(eigenvector), 2)  
        })
    
    
    # save df 
    df.sort_values("degree", ascending = False).to_csv(os.path.join("..", "output", args["outfile_csv"]), sep='\t', encoding='utf-8')

    
# define behaviour when called from command line
if __name__=="__main__":
    main()
    
    
    
    
    
    