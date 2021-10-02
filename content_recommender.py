import pandas as pd
import surprise
import pickle

class ContentRecommender:
    
    def __init__(self, users, tracks_df, listens_df):
        self.users = users
        self.tracks_df = tracks_df
        self.listens_df = listens_df
            
    def compute_playlist(self,n, topk=3):
        clusters = set()
        for user in self.users:
            track_ids = (self.listens_df[self.listens_df['user'] == user]
                           .sort_values(by='plays',ascending=False)
                           .head(topk)['track-id'])
            print(track_ids)
            clusters = clusters | set(self.tracks_df[self.tracks_df['musicbrainz-track-id'].isin(track_ids)]['cluster'])

        return self.tracks_df[self.tracks_df['cluster'].isin(clusters)].sample(n=n)[['musicbrainz-track-id','artist-name','track-name']]
