import pandas as pd
import dask.dataframe as dd
import tarfile
import scipy
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import surprise
import pickle 
import seaborn as sns

from dask.delayed import delayed

from surprise import Dataset, Reader
from surprise import SVD, SVDpp, NMF
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNBaseline
from surprise.prediction_algorithms.co_clustering import CoClustering
from surprise import accuracy

from sklearn.manifold import TSNE
from sklearn.neighbors import DistanceMetric
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def sample_k_users(df, k, num_clusters=5):
    
    # select one from each cluster
    num_per = np.bincount(list(map(lambda x : x%num_clusters, np.arange(k))))
    
    # sample user from number
    return sum([list(df[df['label'] == i]['userid'].sample(n=x).values) for i,x in enumerate(num_per)], [])
    
class Metrics:
    
    def __init__(self, tracks_df, listens_df):
        self.listens_df = listens_df
        self.tracks_df  = tracks_df
        
    def compute_precision(self, user_id, predict_df, k=10):

        predicted_tracks = list(predict_df.head(k)['musicbrainz-track-id'])
        predicted_artists = set(self.tracks_df[self.tracks_df['musicbrainz-track-id'].isin(predicted_tracks)]['musicbrainz-artist-id'])
        listened_artists = set(self.listens_df[self.listens_df['user-id'] == user_id]['artist-id'])
        return len(predicted_artists.intersection(listened_artists))/len(predicted_artists)


    def compute_recall(self, user_id, predict_df, k=10):

        predicted_tracks = list(predict_df.head(k)['musicbrainz-track-id'])
        predicted_artists = set(self.tracks_df[self.tracks_df['musicbrainz-track-id'].isin(predicted_tracks)]['musicbrainz-artist-id'])
        listened_artists = set(self.listens_df[self.listens_df['user-id'] == user_id]['artist-id'])
        return len(predicted_artists.intersection(listened_artists))/len(listened_artists)

    def compute_map(self, user_id, predict_df, k=10):

        precisions = [self.compute_precision(user_id, predict_df, k=k_) for k_ in np.arange(k)+1]
        return np.mean(precisions)



