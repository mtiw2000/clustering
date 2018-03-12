# -*- coding: utf-8 -*-
from __future__ import print_function
__version__ = "0.3.0"

"""
Created on Thu Oct 19 12:19:05 2017

@author: tiwari
"""
#https://github.com/arthur-e/Programming-Collective-Intelligence
#http://pythonexample.com/code/feedparser%20example/
#scaping with beautiful soup
#http://web.stanford.edu/~zlotnick/TextAsData/Web_Scraping_with_Beautiful_Soup.html

#https://github.com/arthur-e/Programming-Collective-Intelligence/blob/master/chapter3/clusters.py

#tdif analysis
#https://buhrmann.github.io/tfidf-analysis.html


import feedparser
import time
import sys

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from math import sqrt
from sklearn.metrics import pairwise_distances
from PIL import Image,ImageDraw
from scipy.cluster.hierarchy import ward, dendrogram



#multi dimensional scaling
#http://brandonrose.org/clustering

#from bs4 import BeautifulSoup
#from BeautifulSoup import BeautifulSoup
import pandas as pd
import numpy as np
#import plotly.plotly as py
#from plotly.graph_objs import *
#import plotly.figure_factory as FF

from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot  as plt
from pandas import ExcelWriter
    
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


def tokenize_and_stem(text):

    txt = re.compile(r'<[^>]+>').sub('', text)
    txt = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', txt)

    tokens=[word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [ stemmer.stem(t) for t in filtered_tokens ]  
    return stems
 

    
def tokenize_only(text):
    
    txt = re.compile(r'<[^>]+>').sub('', text)
    txt = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', txt)

    tokens = [word.lower() for sent in nltk.sent_tokenize(txt) for word in nltk.word_tokenize(sent)]

    filtered_tokens = []
    for token in tokens:
         if re.search('[a-zA-Z]', token):
             filtered_tokens.append(token)
    return filtered_tokens
         

#tokenize_only(a[0])

#returns title and dictionary of wordcounts from url
def getwordcounts(url):
    # Parse the feed
    d=feedparser.parse(url)
    wc={}
    # Loop over all the entries
    for e in d.entries:
        if 'summary' in e: 
            summary=e.summary
        else: 
            summary=e.description
    # Extract a list of words
        words=getwords(e.title+' '+summary)

    for word in words:
        wc.setdefault(word,0)
        wc[word]+=1

    return d.feed.title,wc


def getwords(html):
    # Remove all the HTML tags
    txt = re.compile(r'<[^>]+>').sub('', html)
    txt = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', txt)

    # Split words by all non-alpha characters
    words = re.compile(r'[^A-Z^a-z]+').split(txt)

    # Convert to lowercase
    return [word.lower() for word in words if word != '']

#define custom toolbar location
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}
        

   
    
def cluster_rss_feed():
    print("Executing get RSS feeds")

    start_time = time.time()
    print( os.getcwd())
    os.chdir('C:/python_code/clustering/')
    print( os.getcwd())

    feedlist = [line for line in file('data/feedlist.txt')]

#    url='http://newsrss.bbc.co.uk/rss/newsonline_uk_edition/front_page/rss.xml'
#    url='https://wonkette.com/feed'
#    
#    d=feedparser.parse(url)
#    c=[]    
#    a=''
#    for e in d.entries:
#        if 'summary' in e: 
#            summary=e.summary
#        else: 
#            summary=e.description
#        print summary
#        a+=''.join(summary)
#    print a
#    c.append(a)    
#    print summary    


    feeddata = []
    feed_title=[]
    try:
        for url in feedlist:
            try:
                d=feedparser.parse(url)
                feed_title.append(d.feed.title)
                a=''
                for e in d.entries:
                    if 'summary' in e: 
                        summary=e.summary
                    else: 
                        summary=e.description
                    a+=''.join(summary)
                feeddata.append(a)                    
            except:
                print ('Failed to parse feed %s' % url)
    except:
        print ('Failed to parse feed %s' % url)
                
#    totalvocab_stemmed = []
#    totalvocab_tokenized = []
#    for i in feeddata:
#        allwords_stemmed = tokenize_and_stem(i) #for each item in 'feeddata', tokenize/stem
#        totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
#    
#        allwords_tokenized = tokenize_only(i)
#        totalvocab_tokenized.extend(allwords_tokenized)
   
#    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized})
#    vocab_stem_frame = pd.DataFrame({'stem_words': totalvocab_stemmed})


#    print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
             min_df=0.2, stop_words='english',
             use_idf=True, tokenizer=tokenize_only,lowercase=False, ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(feeddata)

    print(tfidf_matrix.shape)
    terms = tfidf_vectorizer.get_feature_names()

#    dense_tfidf_matrix = tfidf_matrix.toarray()
    
    
#    feed_title = a
    
    
    num_clusters = 6
    km = KMeans(n_clusters=num_clusters)
#    %time km.fit(tfidf_matrix)
    km.fit(tfidf_matrix)
    
    clusters = km.labels_.tolist()    

    rss_clusters = pd.DataFrame({'title':feed_title,'cluster':clusters}, index=clusters )
    rss_clusters['cluster'].value_counts()
    print("Top terms per cluster:")
    print()
    #sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
    
#    news_clusters.ix[5]['title'].values.tolist()

    file1 = open("data/rss_cluster_report.html","w")
    
    for i in range(num_clusters):
        print ("Cluster %d words:" % i, end='')
        
        for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
            print(' %s' % terms[ind], end=',')
        print() #add whitespace
        print() #add whitespace
        
        print("Cluster %d news titles:" % i, end='')
        for title in rss_clusters.ix[i]['title'].values.tolist():
            print(' %s,' % title, end='')
        print() #add whitespace
        print() #add whitespace
        
    print()
    print()

    MDS()
    
    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    dist = 1 - cosine_similarity(tfidf_matrix)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    
    xs, ys = pos[:, 0], pos[:, 1]
    print()
    print()



#    set up colors per clusters using a dict
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
    
    #set up cluster names using a dict
    cluster_names = {0: 'Cluster 0', 
                     1: 'Cluster 1', 
                     2: 'Cluster 2', 
                     3: 'Cluster 3', 
                     4: 'Cluster 4'}


    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df_cluster = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=feed_title)) 
    
    #group by cluster
    groups = df_cluster.groupby('label')
    
    #define custom css to format the font and to remove the axis labeling
    css = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }
    
    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }
    
    svg.mpld3-figure {
    margin-left: -200px;}
    """
    
    # Plot 
    fig, ax = plt.subplots(figsize=(14,6)) #set plot size
    ax.margins(0.03) # Optional, just adds 5% padding to the autoscaling
    
    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, 
                         label=cluster_names[name], mec='none', 
                         color=cluster_colors[name])
        ax.set_aspect('auto')
        labels = [i for i in group.title]
        
        #set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels ,
                                           voffset=10, hoffset=10, css=css)
        #connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())    
        
        #set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        
        #set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    
        
    ax.legend(numpoints=1) #show legend with only one dot
    
#    mpld3.display() #show the plot
#    mpld3.save_html()
    file1 = open("data/rss_cluster.html","w")
    
    #uncomment the below to export to html
#    html = mpld3.fig_to_html(fig)
    
    mpld3.save_html(fig,file1)
    file1.close()
#    print(html)
    
    linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
    
    fig, ax = plt.subplots(figsize=(15, 20)) # set size
    ax = dendrogram(linkage_matrix, orientation="left", labels=feed_title);
    
    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    
#    plt.tight_layout() #show plot with tight layout
    
    #uncomment below to save figure
    
    fig.savefig('data/ward_clusters.png', dpi=200) #save figure as ward_clusters



    end_time = time.time()

    print (end_time - start_time)

 
def main():
 
    cluster_rss_feed()
    
    
if __name__ == '__main__':
    main()


       