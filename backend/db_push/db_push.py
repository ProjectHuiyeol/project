import pandas as pd
import os
import pymysql

path = "C:/project_huiyeol/backend/db_push"
os.chdir(path)

MYSQL_HOST = 'localhost'
MYSQL_CONN = pymysql.connect(
    host=MYSQL_HOST,
    port=3306,
    user='root',
    passwd='',
    db='song_db',
    charset='utf8'
)

def conn_mysqldb():
    if not MYSQL_CONN.open: 
        MYSQL_CONN.ping(reconnect=True)
    return MYSQL_CONN

melody_score = pd.read_csv('score_of_melody.csv', encoding='cp949', index_col='Unnamed: 0')
# print(melody_score.head())
song_id = melody_score['SONG_ID']
song_title = melody_score['SONG_TITLE']
song_happy = melody_score['happy']
song_sad = melody_score['sad']
song_relaxed = melody_score['relaxed']
song_angry = melody_score['angry']

new_data = pd.read_csv('new_datas_1.csv', encoding='utf8')
lyrics_id = new_data['SONG_ID']
lyrics_lyrics = new_data['LYRICS']
lyrics_happy = new_data['happy']
lyrics_angry = new_data['angry']
lyrics_dislike = new_data['dislike']
lyrics_fear = new_data['fear']
lyrics_sad = new_data['sad']
lyrics_surprise = new_data['surprise']

# print(song_happy)
# data값 삭제
# for i in range(len(melody_score)):
#     mysql_db = conn_mysqldb()
#     db_cursor = mysql_db.cursor()
#     sql = '''delete from song_info_sound where SONG_ID =  "%s";''' % (song_id[i])
#     db_cursor.execute(sql)
#     try:
#         mysql_db.commit()
#         print('commit')
#     except:
#         print('error')

# song table code
for i in range(len(melody_score)):
    # print(song_id[i], song_title[i])
    mysql_db = conn_mysqldb()
    db_cursor = mysql_db.cursor()
    sql = '''INSERT INTO song (song_id, song_title) VALUES ("%s", "%s");''' % (song_id[i],str(song_title[i]))
    db_cursor.execute(sql)
    try:
        mysql_db.commit()
        print('commit')
    except:
        print('error')
# song_info_sound code
for i in range(len(melody_score)):
    mysql_db = conn_mysqldb()
    db_cursor = mysql_db.cursor()
    sql = '''INSERT INTO song_info_sound (song_id, song_mp3_path, HAPPY, SAD, ANGRY, RELAXED) VALUES ((select song_id from song where song_id = %d),"../static/music/%s.mp3",%f,%f,%f,%f);'''% (song_id[i],str(song_id[i]),song_happy[i],song_sad[i],song_angry[i],song_relaxed[i])
    db_cursor.execute(sql)
    try:
        mysql_db.commit()
        print('commit')
    except:
        print('error')

# lyrics
for i in range(len(new_data)):
    mysql_db = conn_mysqldb()
    db_cursor = mysql_db.cursor()
    sql = '''INSERT INTO song_info_lyrics (song_id, song_lyrics, happy, fear,angry, dislike ,surprise, sad) VALUES ((select song_id from song where song_id = %d),"%s",%f,%f,%f,%f,%f,%f);''' % (lyrics_id[i], str(lyrics_lyrics[i]) ,lyrics_happy[i],lyrics_fear[i],lyrics_angry[i],lyrics_dislike[i], lyrics_surprise[i], lyrics_sad[i])
    db_cursor.execute(sql)
    try:
        mysql_db.commit()
        print('commit')
    except:
        print('error')