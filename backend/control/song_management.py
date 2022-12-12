from model.sql_connect import *
import numpy as np
import pandas as pd
class Song():
    def __init__(self, data):
        self.song_id_list = []
        self.song_title_list = []
        for i in range(len(data)):
            self.song_id_list.append(data[i][0])
            self.song_title_list.append(data[i][1])
        
    def get_song_ids(self):
        return self.song_id_list
    def get_song_title(self):
        return self.song_title_list
    
    @staticmethod
    def get_all_song():
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        sql = "SELECT * FROM song;"
        db_cursor.execute(sql)
        data = db_cursor.fetchall()
        if not data:
            return None
        return data
        
    @staticmethod
    def get_custom_song_id(playlist_name):
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        sql = "select `song_db`.`playlist`.`song_id`  from `song_db`.`playlist` where `song_db`.`playlist`.`playlist_id` = (select `song_db`.`playlist_info`.`playlist_id` from `song_db`.`playlist_info` where `song_db`.`playlist_info`.`playlist_name` = '%s');" %(str(playlist_name))
        db_cursor.execute(sql)
        data = db_cursor.fetchall()
        if not data:
            return None
        return data

    @staticmethod
    def get_custom_song_titles(song_id_list):
        song_title_list = []
        try : 
            for song_id in song_id_list:
                mysql_db = conn_mysqldb()
                db_cursor = mysql_db.cursor()
                sql = "select `song_db`.`song`.`song_title` from `song_db`.`song` where `song_db`.`song`.`song_id` = %s;" %(song_id)
                db_cursor.execute(sql)
                data = db_cursor.fetchone()
                song_title_list.append(data)
            if not song_title_list:
                return None
            return song_title_list
        except:
            return None

    @staticmethod
    def get_custom_song_lyrics(song_ids):
        song_id_lyrics = []
        for song_id in song_ids:
            mysql_db = conn_mysqldb()
            db_cursor = mysql_db.cursor()
            sql = "select `song_db`.`song_info_lyrics`.`song_id`,`song_db`.`song_info_lyrics`.`song_lyrics` from `song_db`.`song_info_lyrics` where `song_db`.`song_info_lyrics`.`song_id` = %s;" %(song_id)
            db_cursor.execute(sql)
            data = db_cursor.fetchone()
            song_id_lyrics.append(data)
        if not song_id_lyrics:
            return None
        return song_id_lyrics

    @staticmethod
    def get_song(string):
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        sql = "SELECT * FROM song WHERE SONG_TITLE like '%" + str(string) +"%'"
        db_cursor.execute(sql)
        data = db_cursor.fetchall()
        if not data:
            return None
        return Song(data)

    @staticmethod
    def get_sound_sentiment(song_id):
        mysql_db = conn_mysqldb()
        db_cursor = mysql_db.cursor()
        sql = '''select * from song_info_sound where song_id = %d;''' %(song_id)
        db_cursor.execute(sql)
        data = db_cursor.fetchall()
        if not data:
            return None
        return data 
    
    @staticmethod
    def get_sound_sentiment(song_ids):
        sentiment_list = []
        for song_id in song_ids:
            mysql_db = conn_mysqldb()
            db_cursor = mysql_db.cursor()
            sql = '''select * from song_info_sound where song_id = %d;''' %(song_id)
            db_cursor.execute(sql)
            data = db_cursor.fetchone()
            sentiment_list.append(data[2:])
        if not sentiment_list:
            return None
        return sentiment_list

    @staticmethod
    def get_lyrics_sentiment(song_ids):
        sentiment_list = []
        for song_id in song_ids:
            mysql_db = conn_mysqldb()
            db_cursor = mysql_db.cursor()
            sql = '''select * from song_info_lyrics where song_id = %d;''' %(song_id)
            db_cursor.execute(sql)
            data = db_cursor.fetchone()
            sentiment_list.append(data[2:])
        if not sentiment_list:
            return None
        return sentiment_list

    @staticmethod
    def get_chart_score(sentiments_from_db):
        sentiments_lst = []
        for sentiment in sentiments_from_db:
            sentiments_lst.append(list(sentiment))
        temp = []
        for i in range(len(sentiments_lst[0])):
            a=0
            for j in range(len(sentiments_lst)):
                a += sentiments_lst[j][i]

            temp.append(a)
        return list(np.array(temp)/len(sentiments_lst))

    @staticmethod
    def get_custom_lyrics(song_ids):
        lyrics_list = []
        for song_id in song_ids:
            mysql_db = conn_mysqldb()
            db_cursor = mysql_db.cursor()
            sql = '''select song_lyrics from song_info_lyrics where song_id = %d;''' %(song_id)
            db_cursor.execute(sql)
            data = db_cursor.fetchone()[0]
            lyrics_list.append(data[2:])
        if not lyrics_list:
            return None
        return lyrics_list