from flask import Flask, Blueprint, request, render_template, jsonify, make_response, redirect, url_for, session
from flask_login import login_user, current_user, logout_user
from control.playlist_management import Playlist
from control.song_management import Song
from control.model_vgg import VGG
from control.modelNlp import *
from control.colormapping import *
import numpy as np

main1 = Blueprint('main1', __name__)

@main1.route('/', methods = ['GET','POST'])
def main1_():
    name = request.args.get('playlist_name')
    point = request.args.get('playlist_point')
    print(name, point)
    if name and point :
        Playlist.make_new_playlist(name,point)
    else:
        return render_template('test.html')
    all_songlist = Song.get_all_song()
    return render_template('main1.html',playlist_name =name, all_songlist = all_songlist)

# @main1.route('/create', methods = ['GET','POST'])
# def main2_():
#     name = request.args.get('pl_valid_name')
#     point = request.args.get('pl_valid_point')
#     print(name, point)
#     if name and point :
#         Playlist.make_new_playlist(name,point)
#     else:
#         return render_template('index.html')
#     all_songlist = Song.get_all_song()
#     return render_template('main1.html',playlist_name =name, all_songlist = all_songlist)

@main1.route('/<playlist_name>', methods = ['GET','POST'])
def custom_playlist(playlist_name):
    # name = request.args.get('playlist_name')
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
    custom_song_list = Song.get_custom_song_id(pl_name)
    song_title_list = Song.get_custom_song_titles(custom_song_list)
    sound_sentiments = Song.get_sound_sentiment(custom_song_list)
    lyrics_sentiments = Song.get_lyrics_sentiment(custom_song_list)
    ss_lst = Song.get_chart_score(sound_sentiments)
    ls_lst = Song.get_chart_score(lyrics_sentiments)
    print(ss_lst, ls_lst)
    return render_template('result.html',pl_name = pl_name, song_custom_list = song_title_list, sound_sentiments = ss_lst, lyrics_sentiments = ls_lst)

# @main1.route('/duplicate', methods = ['GET','POST'])
# def duplicate():
#     name = request.get_json()
#     print(name)
#     return jsonify(result = "success",result2 = name)

