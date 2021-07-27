from math import pi, trunc
from flask import Flask,render_template,redirect,request
import pickle
from sklearn.model_selection import train_test_split
import time
import joblib
import requests
import pickle
from models import Recommenders
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import *
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process



app = Flask(__name__, template_folder='templates')

@app.route('/')
def main():

    #triplets_file = 'C:\\Users\\Chandana C L\\Documents\\MAJOR PROJECT\\10000.txt'
    #songs_metadata_file = 'C:\\Users\\Chandana C L\\Documents\\MAJOR PROJECT\\song_data.csv'
    data_file = 'https://s3.ap-geo.objectstorage.softlayer.net/musicdataset/majorsongs.csv'

    #song_df_1 = pandas.read_table(triplets_file, header=None)
    #song_df_1.columns = ['user_id', 'song_id', 'listen_count']

    #song_df_2 = pandas.read_csv(songs_metadata_file)
    song_df = pandas.read_csv(data_file, header=1, sep=",")
    song_df.columns = ['id','song','singer','language','genre','movie/album','listen_count','user id']
    #song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

    song_df.head()
    print(song_df.head())
    length = len(song_df)

    song_df = song_df.head(10000)
    song_df['song'] = song_df['song'].map(str)+"-"+song_df['singer']
    song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
    grouped_sum = song_grouped['listen_count'].sum()
    print(grouped_sum)
    song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum)*100
    print("\nMOST POPULAR SONGS IN DATASET")
    song_grouped = song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])
    print(song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1]))
    
    return render_template("main.html", songs=song_grouped['song'].head(15))

@app.route('/', methods=['GET','POST'])
def predict():

    #triplets_file = 'C:\\Users\\Chandana C L\\Documents\\MAJOR PROJECT\\10000.txt'
    #songs_metadata_file = 'C:\\Users\\Chandana C L\\Documents\\MAJOR PROJECT\\song_data.csv'
    data_file = 'https://s3.ap-geo.objectstorage.softlayer.net/musicdataset/majorsongs.csv'

    #song_df_1 = pandas.read_table(triplets_file, header=None)
    #song_df_1.columns = ['user_id', 'song_id', 'listen_count']

    #song_df_2 = pandas.read_csv(songs_metadata_file)
    song_df = pandas.read_csv(data_file, header=1, sep=",")
    song_df.columns = ['id','song','singer','language','genre','movie/album','listen_count','user id']
    #song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

    song_df.head()
    print(song_df.head())
    length = len(song_df)

    song_df = song_df.head(10000)
    song_df['song'] = song_df['song'].map(str)+"-"+song_df['singer']
    song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
    grouped_sum = song_grouped['listen_count'].sum()
    print(grouped_sum)
    song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum)*100
    print("\nMOST POPULAR SONGS IN DATASET")
    song_grouped = song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])
    print(song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1]))


    if request.method== "POST":
        int_features = request.form["song"]
        songs = pandas.read_csv("https://s3.ap-geo.objectstorage.softlayer.net/musicdataset/majorsongs.csv", header=1, sep=",")
        songs.columns = ['id','title','singer','language','genre','movie/album','listen count','user id']

        song_user = songs.groupby('user id')['id'].count()
        song_ten_id = song_user[song_user > 16].index.to_list()
        df_song_id_more_ten = songs[songs['user id'].isin(song_ten_id)].reset_index(drop=True)

        df_songs_features = df_song_id_more_ten.pivot(index='id', columns='user id', values='listen count').fillna(0)
        mat_songs_features = csr_matrix(df_songs_features.values)

        class Recommender():
            def __init__(self, metric, algorithm, k, data, decode_id_song):
        # .
                self.metric = metric
                self.algorithm = algorithm
                self.k = k
                self.data = data
                self.decode_id_song = decode_id_song
                #self.model = self._recommender().fit(data)
                self.model = NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1).fit(data)
      

            def _recommender(self):
                 return NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1)

            def make_recommendation(self, new_song, n_recommendations):
                recommended = self._recommend(new_song=new_song, n_recommendations=n_recommendations)
                print("... Done")
                return recommended 

            def _recommend(self, new_song, n_recommendations):
                # Get the id of the recommended songs
                recommendations = []
                recommendation_ids = self.get_recommendation(new_song=new_song, n_recommendations=n_recommendations)
                # return the name of the song using a mapping dictionary
                recommendations_map = self._map_indeces_to_song_title(recommendation_ids)
                # Translate this recommendations into the ranking of song titles recommended
                for i, (idx, dist) in enumerate(recommendation_ids):
                    recommendations.append(recommendations_map[idx])
                return recommendations

            def get_recommendation(self, new_song, n_recommendations):
                recom_song_id = self._fuzzy_matching(song=new_song)
                # Return the n neighbors for the song id
                distances, indices = self.model.kneighbors(self.data[recom_song_id], 
                                                           n_neighbors=n_recommendations+1)
                return sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), 
                              key=lambda x: x[1])[:0:-1]

            def _map_indeces_to_song_title(self, recommendation_ids):
                # get reverse mapper
                return {song_id: song_title for song_title, song_id in self.decode_id_song.items()}


            def _fuzzy_matching(self, song):
                match_tuple = []
                # get match
                for title, idx in self.decode_id_song.items():
                    ratio = fuzz.ratio(title.lower(), song.lower())
                    if ratio >= 60:
                        match_tuple.append((title, idx, ratio))
            
                match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    
                return match_tuple[0][1]

        df_unique_songs = songs.drop_duplicates(['id']).reset_index(drop=True)[['id', 'title']]
        decode_id_song = {
            song: i for i, song in 
            enumerate(list(df_unique_songs.set_index('id').loc[df_songs_features.index].title))
            }

        model = Recommender(metric='cosine', algorithm='brute', k=20, data=mat_songs_features, decode_id_song=decode_id_song)
        new_recommendations = model.make_recommendation(new_song=int_features, n_recommendations=10)

        return render_template("main.html", data=new_recommendations, songs=song_grouped['song'].head(15))
    else:
        return render_template("main.html")



if __name__ == '__main__':
    app.run(debug=True)