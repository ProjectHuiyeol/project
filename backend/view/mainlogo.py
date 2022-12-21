from flask import Flask, Blueprint, request, render_template, jsonify, make_response, redirect, url_for, session
from flask_login import login_user, current_user, logout_user
from control.playlist_management import Playlist
from control.song_management import Song

mainlogo = Blueprint('mainlogo', __name__)

@mainlogo.route('/')
def main():
    pl_list = Playlist.call_playlist()
    valid_temp = []
    if pl_list:
        for pl in pl_list:
            valid_temp.append(pl[1])
    print(pl_list)
    return render_template('index.html',pl_list = pl_list, valid_temp=valid_temp)

@mainlogo.route('/psych_test')
def psych_test():
    pl_lst = Playlist.valid_playlist()
    return render_template('test.html',pl_lst=pl_lst)

@mainlogo.route('/create')
def main2_():
    name = request.args.get('playlist_name')
    point = request.args.get('playlist_point')
    print(name, point)
    if name and point:
        Playlist.make_new_playlist(name,point)
    else:
        return render_template('index.html')
    all_songlist = Song.get_all_song()
    return render_template('main1.html',playlist_name =name, all_songlist = all_songlist)

@mainlogo.route('/delete<playlist_name>')
def delete_playlist(playlist_name):
    
    Playlist.delete_playlist(playlist_name)
    pl_list = Playlist.call_playlist()
    return render_template('index.html',pl_list = pl_list)