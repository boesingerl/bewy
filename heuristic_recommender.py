import pandas as pd
import numpy as np
import surprise
import pickle

class HeuristicRecommender:
    
    def __init__(self, users, tracks_df, listens_df, genre_artists_path):
        self.users = users
        self.tracks_df = tracks_df
        self.listens_df = listens_df
        
        with open(genre_artists_path, 'rb') as handle:
            self.genre_artists = pickle.load(handle)
    
    def compute_playlist(self,n, topk=3):
        return self.tracks_df.sample(n=n)[['musicbrainz-track-id','artist-name','track-name']]
    
    def _sample_per_artist(self, artists):
        columns = ['musicbrainz-track-id','artist-name','track-name']
        return pd.DataFrame(
            [self.tracks_df[self.tracks_df['musicbrainz-artist-id'] == artist].sample(1)[columns] for artist in artists],
            columns=columns)
    
    def _compute_playlist_common_artists(self, n):
        # compute plays for each track for each user
        df_play_per_user = self.listens_df[self.listens_df['user-id'].isin(self.users)]
        
        # compute plays for each artist for each user
        df_artist_per_user = pd.DataFrame(df_play_per_user.groupby('user-id')['artist-name'].apply(set).apply(list))

        # group plays per artist
        df_plays_by_artist = (df_play_per_user[['artist-name','plays','user-id']]
         .groupby(['user-id','artist-name'])
         .sum()
         .reset_index()[['artist-name','plays']]
         .groupby('artist-name').mean()
         .sort_values('plays',ascending=False))
        
        # number of users that listen to artist
        df_nb_users = (df_artist_per_user.reset_index()
                       .explode('artist-name')
                       .groupby('artist-name')
                       .count()
                       .sort_values('user-id',ascending=False))

        # return artists listened by enough people
        df_artists = (df_nb_users
                .merge(right=df_plays_by_artist,on='artist-name')
                .sort_values(by=['user-id','plays'],ascending = False)
                .head(n))
                
        artists = list(df_artists.index)
        
        return self._sample_per_artist(artists)
    
    def _compute_playlist_common_songs(self,n):
        # compute plays for each track for each user
        df_play_per_user = self.listens_df[self.listens_df['user-id'].isin(self.users)]
        
        # get all songs listened to by user
        df_song_per_user = pd.DataFrame(df_play_per_user.groupby('user-id')['track-name'].apply(set).apply(list))

        # get total number of plays per track in group
        df_plays_by_track = (df_play_per_user[['track-name','plays','user-id']]
         .groupby(['user-id','track-name'])
         .sum()
         .reset_index()[['track-name','plays']]
         .groupby('track-name').mean()
         .sort_values('plays',ascending=False))

        # get number of users that listen to track
        df_nb_users_play = (df_song_per_user
                            .reset_index()
                            .explode('track-name')
                            .groupby('track-name')
                            .count()
                            .sort_values('user-id',ascending=False))

        # merge both previous
        return (df_nb_users_play
                .merge(right=df_plays_by_track,on='track-name')
                .sort_values(by=['user-id','plays'],ascending = False)
                .head(n))
    
    def _compute_playlist_common_genres(self,n):
        # get all songs listened to by users
        df_users = self.listens_d[self.listens_d['user-id'].isin(self.users)]

        # per track genres
        df_music_genre = self.tracks_df[['musicbrainz-track-id','genres']]

        # merge genres by users
        df_merge = df_users.merge(right=df_music_genre,left_on='track-id',right_on='musicbrainz-track-id')
        df_merge.genres = df_merge.genres.apply(lambda x : list(eval(x)))

        # one entry per genre
        df_merge_exploded = df_merge.explode('genres')

        # get mean number of plays over the group of users
        df_plays_by_genre = (df_merge_exploded[['genres','plays','user-id']]
         .groupby(['user-id','genres'])
         .sum()
         .reset_index()[['genres','plays']]
         .groupby('genres').median()
         .sort_values('plays',ascending=False))

        # get number of users that listen to the genre
        df_nb_users_genre = (df_merge_exploded.groupby('user-id')['genres'].apply(set).apply(list)
                             .reset_index()
                             .explode('genres')
                            .groupby('genres')
                            .count()
                            .sort_values('user-id',ascending=False))

        # merge two previous df 
        df_top_genres = (df_nb_users_genre
                         .merge(right=df_plays_by_genre,on='genres')
                         .sort_values(by=['user-id','plays'],ascending = False)
                         .head(n))
        
        genres = list(df_top_genres.index)
        artists = self._sample_per_genre(genres)
        
        return self._sample_per_artist(artists)
    
    
    
    def _sample_per_genre(self, genres):
        return [np.random.choice(list(self.genre_artists[genre]), size=1) for genre in genres]