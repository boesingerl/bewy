import pandas as pd
import surprise
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CollabRecommender:
    
    def __init__(self, model_path, users, tracks_df, listens_df, gamma):
        self.users = users
        self.tracks_df = tracks_df
        self.listens_df = listens_df
        self.gamma = gamma

        with open(model_path, 'rb') as handle:
            self.model = pickle.load(handle)
        
    def _predictions_from_users(self):
        """Predicts ratings for each user and each track given.

        Args:
            df: the dataframe of tracks
            users: the list of users
        Returns:
            a dataframe of size (tracks,users) of ratings provided by the svd algorithm

        """

        # compute full ratings matrix
        ratings = self.model.user_factors[self.users].dot(self.model.item_factors.T)

        # compute prediction for all users, all tracks
        predicts_df = pd.DataFrame(ratings).T

        # normalize per user
        predicts_df = predicts_df.replace(-100, np.NaN)
        predicts_df = predicts_df.fillna(predicts_df.mean())
        predicts_df = pd.DataFrame(MinMaxScaler().fit_transform(predicts_df))

        return predicts_df

    def _disagreement_variance(self,predicts_df):
        # init value
        values = np.zeros(predicts_df.shape[0])

        # iterate over all pairs of users
        for col1 in predicts_df.columns:
            for col2 in predicts_df.columns:
                if col1 != col2:
                    # add difference
                    values += np.abs(predicts_df[col1] - predicts_df[col2])

        return values * 2/(predicts_df.shape[1] * (predicts_df.shape[1] - 1))

    def _group_ratings(self,predicts_df, relevance_coeff = 0.5, max_rating=1):
        
        # compute relevance
        average_relevance = predicts_df.mean(axis=1).to_frame('relevance') / max_rating
        
        # compute variance
        variance = self._disagreement_variance(predicts_df).to_frame('variance')
        
        # join back variance and relevance in a single rating
        group_ratings = average_relevance.join(variance)
        group_ratings['rating'] = ((relevance_coeff*group_ratings['relevance'])
                                + (1-relevance_coeff)*(1-group_ratings['variance']))
        return group_ratings

    def compute_playlist(self,n):

        # compute predictions for the given users
        predicts_df = self._predictions_from_users()

        # compute group ratings
        group_ratings = self._group_ratings(predicts_df, relevance_coeff=self.gamma)

        # compute top ratings
        topn_ratings = group_ratings.sort_values(by='rating', ascending=False).head(n).index

        # return topn
        todisp = self.listens_df[self.listens_df['cattrack'].isin(topn_ratings)][['track-id','artist-name','track-name']].drop_duplicates().rename(columns={'track-id':'musicbrainz-track-id'})

        return todisp[['musicbrainz-track-id','artist-name','track-name']]