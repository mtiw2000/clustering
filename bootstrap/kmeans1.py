# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 18:41:32 2018

@author: Manish
"""
import sys
import os
import numpy as np
import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns;
from sklearn.metrics import accuracy_score


    sns.set()
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.cluster import KMeans
    
    X, y_true = make_blobs(n_samples=300, centers=4,
                           cluster_std=0.60, random_state=0)
    
    plt.scatter(X[:, 0], X[:, 1], s=50);
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    
    #iris data example 
    from sklearn.datasets import load_iris
    iris = load_iris()
    features = iris.data.T
    plt.scatter(features[0],features[1],alpha=0.2,s=100*features[3],c=iris.target,cmap='viridis')

#example 2

    os.getcwd()
    os.chdir('C:/python_code/clustering')
    header = ['years','gross','gender','country']
    
    salary_data= pd.read_csv('data/salary.csv',names=header, skiprows=1)
    
    salary_data.shape
    salary_data.head()

    data=np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)
    data = pd.DataFrame(data,columns=['x','y'])
    
    for col in ['years','gross']:
        plt.hist(salary_data[col],normed=True,alpha=.5)
    

    plt.hist(np.log(salary_data['gross']),normed=True,alpha=.5)
    sns.kdeplot(np.log(salary_data['gross']),shade=True)
    sns.kdeplot(salary_data['gross'],shade=True)
    sns.kdeplot(salary_data.groupby(['country','years'])['gross'].mean(),shade=True)
    sns.pairplot()                   
    
#measures of dispersion
    dispersion_salary_data=salary_data[['gross','years']].describe()
    salary_data[['gross','years']].mean(axis=0)
    b=salary_data.groupby('country')['gross'].describe()
    plt.scatter(salary_data['years'],salary_data['gross'])

    salary_data['country'].unique()

#model
    kmeans = KMeans(n_clusters=3)
    X= salary_data[['gross','years']]
    y_true =  salary_data['gender']
    a=kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    accuracy_score()
    a=accuracy_score(y_true,y_kmeans,normalize=True)
    