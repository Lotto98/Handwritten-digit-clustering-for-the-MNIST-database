from typing import Union, List, Tuple, Dict

from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, SpectralClustering

from tqdm import tqdm

import numpy as np
import pandas as pd

import random
import matplotlib.pyplot as plt

import pickle

import seaborn as sns

from IPython.display import display

pd.set_option('display.max_rows', 500)

import time

import os

import math

def slip_models_file(model_name:str)->None:
    """
    Function that splits the fitted model file in multiple files.

    Args:
        model_name (str): name of the model.
    """
    
    if model_name not in ["GaussianMixture","NormalizedCut","MeanShift"]:
        print("wrong model name...")
        return
            
    with open(model_name+"/fitted_models.pkl", 'rb') as inp:
        fitted_models=pickle.load(inp)
    
    division=2
    size=math.ceil(len(fitted_models)/division)
    
    while(len( pickle.dumps(dict(list(fitted_models.items())[0:size]),-1) ) >= 104857600):
        division+=1
        size=math.ceil(len(fitted_models)/division)
    
    for count, i in enumerate(range(0, len(fitted_models), size)):
        data=dict(list(fitted_models.items())[i: i + size])
    
        with open(model_name+"/fitted_models_part"+str(count)+".pkl", 'wb') as out:
            pickle.dump(data, out, pickle.HIGHEST_PROTOCOL)
        
    os.remove(model_name+"/fitted_models.pkl")
    
def merge_models_files(model_name:str, parts:int)->None:
    """
    Function that merges fitted models files into one file.

    Args:
        model_name (str): model name.
        parts (int): number of parts to merge.
    """
    
    if model_name not in ["GaussianMixture","NormalizedCut","MeanShift"]:
        print("wrong model name...")
        return
    
    d={}
    
    for count in range(0, parts):
    
        with open(model_name+"/fitted_models_part"+str(count)+".pkl", 'rb') as inp:
            p=pickle.load(inp)
            
        d.update(p)
        
        os.remove(model_name+"/fitted_models_part"+str(count)+".pkl")
    
    with open(model_name+"/fitted_models.pkl", 'wb') as out:
        pickle.dump(d, out, pickle.HIGHEST_PROTOCOL)

def load_results(model_name:str)->Tuple[Dict[int,pd.DataFrame],
                                        Dict[int,int],
                                        Dict[int,Union[GaussianMixture, MeanShift, SpectralClustering]],
                                        Dict[int,float]]:
    """
    Function that loads from files the results for the specified model.

    Args:
        model_name (str): results to load.

    Returns:
        Tuple[  Dict[int,pd.DataFrame],
                Dict[int,int], 
                Dict[int,Union[GaussianMixture, MeanShift, SpectralClustering]], 
                Dict[int,float]]: tuple of:
                                    -dict of:
                                        *key: PCA dimension.
                                        *value: result dataframe for this dimension.
                                        
                                    -dict of:
                                        *key: PCA dimension.
                                        *value: best result index for the result dataframe for this dimension.
                                    
                                    -dict of:
                                        *key: PCA dimension.
                                        *value: fitted model with the best hyperparameter/s for this dimension.
                                    
                                    -dict of:
                                        *key: PCA dimension.
                                        *value: fitting time for the best model for this dimension.
    """
    
    if model_name not in ["GaussianMixture","NormalizedCut","MeanShift"]:
        print("wrong model name...")
        return
    
    with open(model_name+"/results.pkl", 'rb') as inp:
        results=pickle.load(inp)
    
    with open(model_name+"/best_indexes.pkl", 'rb') as inp:
        best_indexes=pickle.load(inp)
            
    with open(model_name+"/timings.pkl", 'rb') as inp:
        timings=pickle.load(inp)
    
    with open(model_name+"/fitted_models.pkl", 'rb') as inp:
        fitted_models=pickle.load(inp)
    
    return results,best_indexes,fitted_models,timings

def to_frame(results: Dict[int,pd.Series])->pd.DataFrame:
    """
    Function that given the results dict converts it to pandas dataframe.

    Args:
        results (Dict[int,pd.Series]): results dict.

    Returns:
        pd.DataFrame: converted result
    """
    results_df = pd.concat(results, axis=0).reset_index(1,drop=True)
    
    results_df.index.name="PCA dimension"
    
    return results_df

def to_latex(results: Dict[int,pd.Series], best_indexes: Dict[int,int])->None:
    """
    Function that converts the results in latex code.

    Args:
        results (Dict[int,pd.Series]): tuning results.
        best_indexes (Dict[int,int]): best indexes for the tuning result. (used to highlight the best row for each sub-table)
    """
    
    to_print="\\begin{longtable}{rrr}\n\\endhead\n\\midrule\n\\multicolumn{2}{r}{Continued on next page} \\\\\n\\midrule\n\\endfoot\n\\endlastfoot\n"
    count=0
    
    for PCA_dim in results.keys():
        
        pre = "\\begin{tabular}{|c|c|}\n\\hline\n\\multicolumn{2}{|c|}{PCA dimension "+str(PCA_dim)+"} \\\\\n\\hline\nn\\_compon. & rand index \\\\\n\\hline\n"
        
        df = to_frame(results).groupby("PCA dimension").get_group(PCA_dim)
        
        latex_output:str = df.to_latex(index=False,header=False)
        
        latex_output_cleaned = "\n".join([ "\\rowcolor{lightgray}\n"+o if i==best_indexes[PCA_dim] else "\\rowcolor{white}\n"+o for i,o in enumerate(latex_output.split("\n")[3:-3]) ])
        
        post = "\n\\hline\n\\end{tabular}\n"
        
        div=''
        
        if count!=0:
            div='& \n'
        
        if count==3:
            div='\\\\\\\\ \n'
            count=0
            
        to_print=to_print+div+pre+latex_output_cleaned+post
        count=count+1

    to_print=to_print+"\n\\end{longtable}"
    
    print(to_print)

def load_PCA_dfs()->Tuple[Dict[int,pd.DataFrame], pd.Series]:
    """
    Function that loads from file all the PCA reduced datasets.

    Returns:
        Tuple[Dict[int,pd.DataFrame], pd.Series]: tuple of:
                                                    -dict of:
                                                        *key: pca dimension.
                                                        *value: PCA reduced dataframe.
                                                    -ground truth dataset.
    """
    
    y = pd.read_parquet('dataset/y.parquet').squeeze()
    
    dfs={}
    
    for i in tqdm(range(2,200,10),leave=False):
        
        dfs[i]=pd.read_parquet("dataset/PCA_"+str(i)+".parquet")
        
    return dfs,y

def load_df(key:int)->pd.DataFrame:
    """
    Function that loads a specific PCA reduced dataset.
    
    Args:
        key (int): dimension of the PCA reduced dataset to load.

    Returns:
        pd.DataFrame: PCA reduced dataset.
    """
    return pd.read_parquet("dataset/PCA_"+str(key)+".parquet")

def best_worst_pca_dimension(results:Dict[int, pd.DataFrame],
                             best_indexes:Dict[int,int],
                             hyperparameter_name:str)->Tuple[int,int,pd.DataFrame]:
    """
    Function that:
        -display the best hyperparameter value for each PCA dimension.
        -find the best PCA dimension and the worst PCA dimension.
        -plot the effects of PCA dimension over the rand index, hyperparameter value and n_cluster (Mean Shift only).

    Returns:
        Tuple[int,int,pd.Dataframe]: best PCA dimension, worst PCA dimension and 'Best n_components for each PCA dimension' dataframe.
    """
    
    max_ = 0
    min_ = float("inf")
    
    data=[]

    for x in results.keys():
        s = results[x].iloc[best_indexes[x]]
        
        hyperparameter = s[hyperparameter_name]
        rand_index = s["rand index"]
        
        if len(s.index)==3:
            n_clusters = s['n_clusters']
            data.append([hyperparameter,n_clusters, rand_index])
        else:
            data.append([hyperparameter,rand_index])
        
        if rand_index>max_:
            max_=rand_index
            best_pca=x
            best_hyperparameter=hyperparameter
            
        if rand_index<min_:
            min_=rand_index
            worst_pca=x
            worst_hyperparameter=hyperparameter
    
    if len(s.index)==3:
        columns=["best "+hyperparameter_name, 'n_clusters', "rand_index"]
    else:
        columns=["best "+hyperparameter_name,"rand_index"]
        
    data = pd.DataFrame(data, index=[x for x in results.keys()], columns=columns)
    to_print = data.style.set_caption("Best n_components for each PCA dimension")
    
    display(to_print)
    
    print(f"\nThe best PCA dimension is {best_pca} ({hyperparameter_name}={best_hyperparameter}) with a rand score of {max_}")
    print(f"\nThe worst PCA dimension is {worst_pca} ({hyperparameter_name}={worst_hyperparameter}) with a rand score of {min_}")
    
    size=6
    
    fig, axs = plt.subplots(1,2,figsize=(2*size,size))
    
    axs[0].plot([e for e in results.keys()],data["rand_index"])
    axs[0].set_xlabel("PCA dim")
    axs[0].set_ylabel("best rand index")
    axs[0].set_title("best rand index vs PCA dim")
    
    axs[1].plot([e for e in results.keys()],data["best "+hyperparameter_name])
    axs[1].set_xlabel("PCA dim")
    axs[1].set_ylabel("best hyperparameter")
    axs[1].set_title("best hyperparameter vs PCA dim")
    
    if len(s.index)==3:
        fig, ax=plt.subplots(figsize=(2*size,size))
        
        ax.plot([e for e in results.keys()],data["n_clusters"])
        ax.set_xlabel("PCA dim")
        ax.set_ylabel("best n_clusters")
        ax.set_title("best n_clusters vs PCA dim")
    
    return best_pca, worst_pca, data


def plot_clustering(model_name:str,
                    fitted_estimator_PCA2:Union[GaussianMixture,MeanShift,SpectralClustering])->None:
    """
    Function that plots the clustering of PCA dimension 2.

    Args:
        model_name (str): name of the model.
        fitted_estimator_PCA2 (Union[GaussianMixture,MeanShift,SpectralClustering]): fitted model to use.
    """

    X=load_df(2)
    
    match model_name:
        case "GaussianMixture":
            n_clusters_ = fitted_estimator_PCA2.get_params()["n_components"]
            labels = fitted_estimator_PCA2.predict(X)
            cluster_centers = fitted_estimator_PCA2.means_
            
        case "MeanShift":
            labels = fitted_estimator_PCA2.labels_
            n_clusters_ = len(np.unique(labels))
            cluster_centers = fitted_estimator_PCA2.cluster_centers_
            
        case "NormalizedCut":
            n_clusters_=fitted_estimator_PCA2.get_params()["n_clusters"]
            labels=fitted_estimator_PCA2.labels_
            cluster_centers = None

        case _:
            print("Wrong model name...")
            return
        
    plt.figure(figsize=(20,10))
    plt.clf()

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_clusters_)]
    markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        
        plt.scatter(X[my_members]["PC_1"], X[my_members]["PC_2"], marker=markers[k%len(markers)], color=col) # type: ignore
        
        if cluster_centers is not None:
            cluster_center = cluster_centers[k]
        else:
            cluster_center=X[my_members].mean()
        
        plt.scatter(cluster_center[0],cluster_center[1],marker=markers[k%len(markers)], edgecolor="black", s=200, color =col) # type: ignore
            
        
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()

def cluster_composition(model_name:str,
                        fitted_estimators:Dict[int,Union[GaussianMixture,MeanShift,SpectralClustering]],
                        pca_dimensions:Union[List[int],int],
                        plot_heatmap:bool=True)->None:
    """
    Function that analyses the content of each cluster for the given PCA dimensions. 
    
    Args:
        model_name (str): name of the model.
        fitted_estimators (Dict[int,Union[GaussianMixture,MeanShift,SpectralClustering]]): fitted estimator for each PCA dim.
        pca_dimensions (Union[List[int],int]): PCA dimensions to use.
    """
    
    dfs, y = load_PCA_dfs()
    
    if isinstance(pca_dimensions, int):
        pca_dimensions=[pca_dimensions]
    
    for pca_dimension in pca_dimensions:
        
        match model_name:
            case "GaussianMixture":
                n_clusters_=fitted_estimators[pca_dimension].get_params()["n_components"]
                labels=fitted_estimators[pca_dimension].predict(dfs[pca_dimension])
                
            case "MeanShift":
                labels=fitted_estimators[pca_dimension].labels_
                n_clusters_=len(np.unique(labels))
                
            case "NormalizedCut":
                n_clusters_=fitted_estimators[pca_dimension].get_params()["n_clusters"]
                labels=fitted_estimators[pca_dimension].labels_

            case _:
                print("Wrong model name...")
                return

        rows=[]
        
        for k in range(n_clusters_):
            my_members = labels == k
            
            indexes=dfs[pca_dimension][my_members].index.values.tolist()
            
            row=y[indexes].value_counts(True).sort_values(ascending=False)
            
            zero_indexes=set([x for x in range(0,10)])-set(row.index.values.tolist())
            
            row=pd.concat([row,pd.Series({i:0 for i in zero_indexes}) ]).sort_index()
            
            rows.append(row)
        
        df=pd.DataFrame(rows)
        
        if plot_heatmap:
            
            fig, ax = plt.subplots(figsize=(10,10))
        
            ax.set_title("Clusters composition for PCA dimension "+str(pca_dimension))
            
            ax=sns.heatmap(df, annot=True, ax=ax, cmap="rocket_r")
            
            ax.set_xlabel("Digit")
            ax.set_ylabel("Cluster id")
            
        else:
            
            df.columns=["Digit "+str(d) for d in range(0,10)]
            df.index.name="Cluster ids"
            display(df)
            
        almost_one_clusters = len(df[df.max(axis=1)>0.85])/len(df)*100
        
        above_half_clusters = len(df[(df.max(axis=1)<=0.85) & (df.max(axis=1)>0.5)])/len(df)*100
        
        below_half_clusters = len(df[(df.max(axis=1)<=0.5)])/len(df)*100
        
        print(f"PCA dimension {pca_dimension}:")
        print(f"-{almost_one_clusters:.6f}% of clusters are near one digit composed")
        print(f"-{above_half_clusters:.6f}% of clusters are above half composed by one digit")
        print(f"-{below_half_clusters:.6f}% of clusters are below half or half composed by one digit\n")
    
def plot_images_per_cluster(model_name:str,
                            pca_dimensions:Union[List[int],int],
                            fitted_estimators:Dict[int,Union[GaussianMixture,MeanShift,SpectralClustering]])->None:
    """
    Function that plots 4 members of each cluster for the given PCA dimensions.

    Args:
        model_name (str): the name of the model.
        pca_dimensions (Union[List[int],int]): PCA dimensions to use.
        fitted_estimators (Dict[int,Union[GaussianMixture,MeanShift,SpectralClustering]]): fitted estimator for each PCA dim.
    """
    
    dfs,_=load_PCA_dfs()
    
    if isinstance(pca_dimensions, int):
        pca_dimensions=[pca_dimensions]
    
    for key in pca_dimensions:
        
        match model_name:
            case "GaussianMixture":
                n_clusters_=fitted_estimators[key].get_params()["n_components"]
                labels=fitted_estimators[key].predict(dfs[key])
                
            case "MeanShift":
                labels=fitted_estimators[key].labels_
                n_clusters_=len(np.unique(labels))
                
            case "NormalizedCut":
                n_clusters_=fitted_estimators[key].get_params()["n_clusters"]
                labels=fitted_estimators[key].labels_

            case _:
                print("Wrong model name...")
                return
    
        fig,axs=plt.subplots(n_clusters_,4 ,figsize=(4*2,n_clusters_*2))
    
        fig.suptitle(f"Digits plot for PCA dimension {key} with {n_clusters_} clusters")
    
        for item in [item for sublist in axs for item in sublist]:
            item.set_yticklabels([])
            item.set_xticklabels([])
            
        with open("pca/pca_"+str(key)+".pkl", 'rb') as inp:
            pca=pickle.load(inp)
            
        for k in range(n_clusters_):
            my_members = labels == k
            
            data=pca.inverse_transform(dfs[key][my_members])
        
            if data.shape[0]>=4:
                random_indexes=np.random.choice(data.shape[0], size=4, replace=False)
            else:
                random_indexes=np.random.choice(data.shape[0], size=data.shape[0], replace=False)
                
            for i,imag in enumerate(data[random_indexes, :]):
                    axs[k,i].imshow(imag.reshape(28, 28))
                    
            if data.shape[0]==0:
                i=-1
        
            for j in range(i+1,4):
                axs[k,j].imshow(np.zeros(28*28).reshape(28,28))
                
        for ax,c in zip(axs[:,0],range(n_clusters_)):
            ax.set_ylabel(str(c), rotation=0, size='large')
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.show()
        
def timings_analysis(timings: Dict[int,float], isFit:bool=True)->Tuple[pd.DataFrame,str]:
    """
    Function that does an analysis for the fitting/prediction times. In particular:
        -plot timings vs PCA dimension.
        -create a timings dataframe.
        -generate latex output.

    Args:
        timings (Dict[int,float]): timings dataframe.
        isFit (bool, optional): if True generates titles for the fitting times, else generates titles for prediction times.
                                Defaults to True.

    Returns:
        Tuple[pd.DataFrame,str]: timings dataframe e latex output.
    """
    
    if isFit:
        col_name="Fitting time"
    else:
        col_name="Prediction time"
    
    df= pd.DataFrame(timings.values(), index=timings.keys(),columns=[col_name])

    df.index.name="PCA dim"
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    ax.plot(df.index, df[col_name])
    
    ax.set_title(col_name+" (seconds) vs PCA dimension")
    
    ax.set_xlabel("PCA dimension")
    ax.set_ylabel(col_name+" (seconds)")
    
    latex_output=df.to_latex(longtable=True, formatters=[lambda s: str(round(s, 6))+"s"])
    
    return df, latex_output

def predict_timings(fitted_models:Dict[int, Union[GaussianMixture,MeanShift,SpectralClustering]])->Dict[int,float]:
    """
    Function that generate prediction times using fitted models.

    Args:
        fitted_models (Dict[int, Union[GaussianMixture,MeanShift,SpectralClustering]]): fitted models.

    Returns:
        Dict[int,float]: prediction timings.
    """
    dfs, _= load_PCA_dfs()
    
    timings={}
    
    for dim in fitted_models.keys():
        
        start =time.time()
        fitted_models[dim].predict(dfs[dim])
        end =time.time()
        
        timings[dim]=(end-start)
        
    return timings