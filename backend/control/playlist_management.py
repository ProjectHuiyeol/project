from flask_login import UserMixin
from model.sql_connect import *

class Playlist():
    def __init__(self, data):
        self.playlist_id = data[0][0]
        self.song_list = []
        data = list(data)
        data.sort(key=lambda x:x[1])
        for i in range(len(data)):
            self.song_list.append(data[i][2])

    
    def get_songlist(self):
        return self.song_list
    
    @staticmethod
    def make_new_playlist(playlist_name, ratio):
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        sql = "INSERT INTO playlist_info (PLAYLIST_NAME, PLAYLIST_RATIO) VALUES ('%s', '%s')" % (str(playlist_name), str(ratio))
        db_cursor.execute(sql)
        try:
            mysql_db.commit()
            print('commit')
        except:
            print('error')
    
    @staticmethod
    def valid_playlist():
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        sql = "select playlist_name from playlist_info ;"
        db_cursor.execute(sql)
        data = db_cursor.fetchall()
        return data

    @staticmethod
    def call_playlist():
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        sql = "SELECT * FROM playlist_info"
        db_cursor.execute(sql)
        data = db_cursor.fetchall()
        print(data,'!!')
        if not data:
            return None
        return data
    
    @staticmethod
    def delete_playlist(playlist_name):
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        playlist_id = Playlist.get_playlist_id(playlist_name)[0][0]
        
        sql = "DELETE FROM playlist WHERE PLAYLIST_ID = '%s'" % (str(playlist_id))
        db_cursor.execute(sql)
        sql = "DELETE FROM playlist_info WHERE PLAYLIST_NAME = '%s'" % (str(playlist_name))
        db_cursor.execute(sql)
        try:
            mysql_db.commit()
            print('commit')
        except:
            print('error')
    
    @staticmethod
    def get_playlist_id(playlist_name=None):
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        if playlist_name is None:
            sql = "SELECT * FROM playlist_info"
        else:
            sql = "SELECT * FROM playlist_info WHERE PLAYLIST_NAME = '%s'" % (str(playlist_name))
        db_cursor.execute(sql)
        data = db_cursor.fetchall()
        print(data,'!!')
        if not data:
            return None
        return data
    
        
    
    @staticmethod
    def get(playlist_id):
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        sql = 'SELECT * FROM playlist WHERE PLAYLIST_ID = "' + str(playlist_id) +'"'
        db_cursor.execute(sql)
        data = db_cursor.fetchall()
        
        if not data:
            return None
        return Playlist(data)
    
    @staticmethod
    def add_song(playlist_name, song_title):
        print('ddddd ', playlist_name)
        # Playlist.delete_song(playlist_id, song_id)
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        sql = "insert into `song_db`.`playlist` (`song_db`.`playlist`.`playlist_id`  , `song_db`.`playlist`.`song_id` ) values ( (select `song_db`.`playlist_info`.`playlist_id` from `song_db`.`playlist_info` where `song_db`.`playlist_info`.`playlist_name` = '%s'), (select `song_db`.`song`.`song_id` from `song_db`.`song` where `song_db`.`song`.`song_title` = '%s'));" % (str(playlist_name), str(song_title))
        db_cursor.execute(sql)
        try:
            mysql_db.commit()
            print('commit')
        except:
            print('error')
    
    @staticmethod
    def delete_song(playlist_id, song_id):
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        sql = "DELETE from playlist WHERE PLAYLIST_ID='%s' AND SONG_ID='%s'" % (str(playlist_id), str(song_id))
        db_cursor.execute(sql)
        mysql_db.commit()