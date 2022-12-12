from flask import Flask, Blueprint, request, render_template, jsonify, make_response, redirect, url_for, session
from flask_login import login_user, current_user, logout_user
from control.playlist_management import Playlist
from control.song_management import Song
from control.model_vgg import VGG
from control.modelNlp import *
from control.colormapping import *
import pandas as pd
import numpy as np
import time
main1 = Blueprint('main1', __name__)

@main1.route('/', methods = ['GET','POST'])
def main1_():
    name = request.args.get('playlist_name')
    point = request.args.get('playlist_point')
    # print(name, point)
    if name and point :
        Playlist.make_new_playlist(name,point)
    else:
        return render_template('index.html')
    all_songlist = Song.get_all_song()
    return render_template('main1.html',playlist_name =name, all_songlist = all_songlist)

@main1.route('/<playlist_name>', methods = ['GET','POST'])
def custom_playlist(playlist_name):
    
    all_song_list = Song.get_all_song()
    custom_song_list = Song.get_custom_song_id(playlist_name)
    song_title_list = Song.get_custom_song_titles(custom_song_list)
    return render_template('main1.html',playlist_name = playlist_name, all_songlist = all_song_list, song_title_list = song_title_list)

@main1.route('/song_info', methods = ['GET','POST'])
def song_info():
    song_names = request.args.get('song_names').split(',')
    pl_name = request.args.get('pl_name')
    # print(pl_name,type(song_names), song_names)
    for song in song_names:
        Playlist.add_song(pl_name,song)
    custom_song_list = Song.get_custom_song_id(pl_name) # song_id list
    song_title_list = Song.get_custom_song_titles(custom_song_list) # song_title list
    
    # ss_lst = Song.get_chart_score(sound_sentiments) # all sound sentiments mean by song id
    # ls_lst = Song.get_chart_score(lyrics_sentiments) # all lyrics sentiments mean by song id
    vgg_model = VGG()
    ss_lst = vgg_model.voting(custom_song_list)
    
    # print(ss_lst)
    lyrics_df= pd.DataFrame(Song.get_custom_song_lyrics(custom_song_list), columns = ['SONG_ID','LYRICS'])
    # print(lyrics_df)
    ls_lst_temp = electra_model.predictAll(lyrics_df)

    ls_lst = []
    for i in range(len(ls_lst_temp)):
        ls_lst.append(np.fromstring(ls_lst_temp.loc[:,'total'].tolist()[i][1:-1], dtype=float, sep=' '))
    ls_lst_score = np.sum(np.array(ls_lst), axis=0) / len(ls_lst)
    # print(ls_lst_score)

    # print(pred_electra)
    playlist_color = make_color(ls_lst_score, ss_lst)
    
    # print(ss_lst)
    check_fns = find_nearest_song(ls_lst_score,ss_lst, counts=5)
    # print(check_fns)
    custom_lyrics = Song.get_custom_lyrics(custom_song_list)
    
    return render_template('result.html',
                            pl_name = pl_name,
                            custom_lyrics = custom_lyrics,
                            song_custom_list = song_title_list, 
                            sound_sentiments = ss_lst, 
                            lyrics_sentiments = ls_lst_score,
                            playlist_color = playlist_color,
                            enumerate = enumerate,
                            check_fns = check_fns
                            )

@main1.route('/similary<check_fns>',methods = ['GET','POST'] )
def similary(check_fns):
    check_fns = check_fns[1:-1]
    check_fns = list(map(lambda x : int(x.strip()) ,check_fns.split(',')))
    # print(check_fns)
    similar_songs = Song.get_custom_song_titles(check_fns)
    sound_sentiments = Song.get_sound_sentiment(check_fns)
    lyrics_sentiments = Song.get_lyrics_sentiment(check_fns)
    similar_colors = []
    for i in range(len(sound_sentiments)):
        similar_colors.append(make_color(lyrics_sentiments[i],sound_sentiments[i])[1])
    # print(similar_songs, similar_colors)
    return render_template('similary.html',
                            similar_songs=similar_songs,
                            similar_colors=similar_colors,
                            enumerate = enumerate)

@main1.route('/lyrics',methods = ['GET','POST'] )
def get_lyrics():
    custom_lyrics = request.args.get('custom_lyrics_id')
    return render_template('lyrics_view.html', custom_lyrics = custom_lyrics)