import pickle
import copy

from tqdm import tqdm

import pandas as pd
import numpy as np

from typing import Union, List, Tuple, Dict

from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, SpectralClustering

from sklearn.metrics.cluster import rand_score

from sys import argv, exit
import os

import time

from utility import load_PCA_dfs

def score_calculation(model: Union[GaussianMixture, MeanShift, SpectralClustering],
                  X:pd.DataFrame,
                  y:pd.Series,
                  hyperparameter_val:Union[int,float])->Union[Tuple[Union[int,float], int, float], Tuple[Union[int,float], float]]:
    
    """
    Function that fits the model with the given hyperparameter value and calculates the rand score.
    If a value for "k" is passed then it is used as an additional hyperparameter for the model. 

    Args:
        model (Union[GaussianMixture, MeanShift, SpectralClustering]): model to use.
        X (pd.DataFrame): feature data-set.
        y (pd.Series): ground truth data-set.
        hyperparameter_val (Union[int,float]): hyperparameter value.

    Returns:
        Union[Tuple[Union[int,float], int, float], Tuple[Union[int,float], float]]:
            tuple of:
                -hyperparameter value
                -number of clusters
                -rand score
            or tuple of:
                -hyperparameter value
                -rand score
    """
    
    model=model.set_params(**{hyperparameter_name:hyperparameter_val})
        
    cluster_labels=model.fit_predict(X)
        
    score=float(rand_score(y, cluster_labels))
    
    if isinstance(model, MeanShift):
        
        return (hyperparameter_val, model.cluster_centers_.shape[0], score)
    
    else:
        
        return (hyperparameter_val, score)

def hyperparameter_tuning(desc:str,
                          model: Union[GaussianMixture, MeanShift, SpectralClustering],
                          hyperparameter_name:str,
                          hyperparameter_values:List[Union[float, int]], 
                          X:pd.DataFrame,
                          y:pd.Series)->Tuple[pd.DataFrame,int,Union[GaussianMixture, MeanShift, SpectralClustering],float]:
    """
    Function that tunes the given hyperparameter for the given model with the given data.
    The result is inserted in a pandas dataframe.

    Args:
        desc (str): description of the tuning.
        model (Union[GaussianMixture, MeanShift, SpectralClustering]): model to tune.
        hyperparameter_name (str): name of the hyperparameter to tune.
        hyperparameter_values (List[Union[float, int]]): list of hyperparameter values.
        X (pd.DataFrame): features data-set.
        y (pd.Series): responses data-set.

    Returns:
        Tuple[pd.DataFrame,int,Union[GaussianMixture, MeanShift, SpectralClustering],float]:
            tuple of:
                -result of the tuning.
                -best result index for the result dataframe.
                -fitted model with the best hyperparameter.
                -fitting time for the best model.
    """
    result=[]
    fitted_model=None

    for val in tqdm(hyperparameter_values,desc=desc,leave=False):
        
        result.append(score_calculation(model,X,y,val))
    
    if isinstance(model, MeanShift):
        columns=[hyperparameter_name, 'n_clusters','rand index']
    else:
        columns=[hyperparameter_name,'rand index']
        
    result=pd.DataFrame(result, columns=columns)
    
    best_result_index=result.index[result["rand index"]==result["rand index"].max()].to_list()[0]
    
    start = time.time()
    
    fitted_model=model.set_params(**{hyperparameter_name:result[hyperparameter_name].iloc[best_result_index].squeeze()}).fit(X)
    
    end = time.time()
    elapsed=end-start
    
    return result, best_result_index, fitted_model, elapsed

def get_results(dfs:Dict[int, pd.DataFrame],
                y:pd.Series,
                model:Union[GaussianMixture, MeanShift, SpectralClustering],
                hyperparameter_name:str,
                hyperparameter_values:List[Union[float, int]])->Tuple[Dict[int,pd.DataFrame],
                                                                    Dict[int,int],
                                                                    Dict[int,Union[GaussianMixture, MeanShift, SpectralClustering]],
                                                                    Dict[int,float]]:
    """
    Function that given the PCA reduced data-frames and the responses dataframe,
    for each PCA dimension, tunes the given hyperparameter of the given model using the given hyperparameter values.

    Args:
        dfs (Dict[int, pd.DataFrame]): PCA reduced dataframes.
        y (pd.Series): response dataframe.
        model (Union[GaussianMixture, MeanShift, SpectralClustering]): model to tune.
        hyperparameter_name (str): hyperparameter to tune.
        hyperparameter_values (List[Union[float, int]]): list of hyperparameter values.

    Returns:
        Tuple[  Dict[int,pd.DataFrame],
                Dict[int,int],
                Dict[int,Union[GaussianMixture, MeanShift, SpectralClustering]],
                Dict[int,float]]:
                    
                    tuple of:
                        -dict of:
                            -key: PCA dimension.
                            -value: result dataframe for this dimension.
                        
                        -dict of:
                            -key: PCA dimension.
                            -value: best result index for the result dataframe for this dimension.
                            
                        -dict of:
                            -key: PCA dimension.
                            -value: fitted model with the best hyperparameter for this dimension.
                            
                        -dict of:
                            -key: PCA dimension.
                            -value: fitting time for the best model for this dimension.
    """
    
    results={}
    best_indexes={}
    fitted_estimators={}
    timings={}

    #dfs.keys()
    
    for dim in tqdm(dfs.keys(),desc="Total result"):
    
        results[dim],best_indexes[dim],fitted_estimator,timings[dim]=hyperparameter_tuning("PCA_"+str(dim),
                                                                                            model,hyperparameter_name,
                                                                                            hyperparameter_values,
                                                                                            dfs[dim], y)
        
        fitted_estimators[dim]=copy.deepcopy(fitted_estimator)
        
    return results, best_indexes, fitted_estimators, timings

def save_results(model_name:str,
                 results:Dict[int,pd.DataFrame],
                 best_indexes:Dict[int,int],
                 fitted_models:Dict[int,Union[GaussianMixture, MeanShift, SpectralClustering]],
                 timings:Dict[int,float]):
    """
    Function that saves the results, the best indexes, fitted models and timings dicts for the given model.

    Args:
        model_name (str): model name.
        
        results (Dict[int,pd.DataFrame]):
            dict of:
                -key: PCA dimension.
                -value: result dataframe for this dimension.
                
        best_indexes (Dict[int,int]):
            dict of:
                -key: PCA dimension.
                -value: best result index for the result dataframe for this dimension.
                
        fitted_models (Dict[int,Union[GaussianMixture, MeanShift, SpectralClustering]]):
            dict of:
                -key: PCA dimension.
                -value: fitted model with the best hyperparameter for this dimension.
                
        timings (Dict[int,float]):
            dict of:
                -key: PCA dimension.
                -value: fitting time for the best model for this dimension.
    """
    
    PATH = os.getcwd()+"/"+model_name
    
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    
    with open(model_name+"/results.pkl", 'wb') as out:
        pickle.dump(results, out, pickle.HIGHEST_PROTOCOL)
    
    with open(model_name+"/best_indexes.pkl", 'wb') as out:
        pickle.dump(best_indexes, out, pickle.HIGHEST_PROTOCOL)
            
    with open(model_name+"/fitted_models.pkl", 'wb') as out:
        pickle.dump(fitted_models, out, pickle.HIGHEST_PROTOCOL)
    
    with open(model_name+"/timings.pkl", 'wb') as out:
        pickle.dump(timings, out, pickle.HIGHEST_PROTOCOL)


if len(argv)!=2:
    print("Error, parameter should be 1: 'GaussianMixture', 'MeanShift' or 'NormalizedCut'")
    exit(1)
    
_,name = argv

n_jobs=-1

#PCA loading
dfs,y=load_PCA_dfs()

hyperparameter_values:List[Union[int,float]]

#model
match name:
    case "GaussianMixture":
        estimator=GaussianMixture(covariance_type="diag", max_iter=3000, random_state=32)
        hyperparameter_name="n_components"
        hyperparameter_values=[x for x in range(5,16)]
        
    case "MeanShift":
        estimator=MeanShift(n_jobs=n_jobs)
        hyperparameter_name="bandwidth"
        hyperparameter_values=[0.2, 0.4, 0.6, 0.8, 1, 2, 5, 10, 15, 20]
        #hyperparameter_values=[x for x in np.arange(0.2, 2, 0.2)]
        
    case "NormalizedCut":
        estimator=SpectralClustering(affinity="nearest_neighbors", n_neighbors=40, n_jobs=n_jobs)
        hyperparameter_name="n_clusters"
        hyperparameter_values=[x for x in range(5,16)]

    case _:
        print("Wrong model name...")
        exit(1)

results,best_indexes,fitted_estimators,timings = get_results(dfs,y,estimator,hyperparameter_name,hyperparameter_values)

#save
save_results(name,results,best_indexes,fitted_estimators,timings)