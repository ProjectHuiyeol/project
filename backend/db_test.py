import pymysql
import numpy as np
import pandas as pd

# MYSQL_HOST = 'localhost'
# MYSQL_CONN = pymysql.connect(
#     host=MYSQL_HOST,
#     port=3306,
#     user='root',
#     passwd='1234',
#     db='song_db',
#     charset='utf8'
# )
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

MYSQL_HOST = 'database-1.cqxqtis1gqjp.ap-northeast-1.rds.amazonaws.com'
MYSQL_CONN = pymysql.connect(
    host=MYSQL_HOST,
    port=3306,
    user='huiyeol',
    passwd='gmlduf0000',
    db='song_db',
    charset='utf8'
)

def conn_mysqldb():
    if not MYSQL_CONN.open:
        MYSQL_CONN.ping(reconnect=True)
    return MYSQL_CONN




# mysql_db = conn_mysqldb()
# db_cursor = mysql_db.cursor()
# sql = "select song_info_lyrics.song_id, song_info_lyrics.song_lyrics from song_info_lyrics;"
# db_cursor.execute(sql)
# data = db_cursor.fetchall()
# print(data[:2])
playlist_name = '테스트1'
mysql_db = conn_mysqldb()
db_cursor = mysql_db.cursor()
playlist_id = get_playlist_id(playlist_name)[0][0]
print(playlist_id)

sql = "DELETE FROM playlist WHERE PLAYLIST_ID = '%s'" % (str(playlist_id))
db_cursor.execute(sql)
sql = "DELETE FROM playlist_info WHERE PLAYLIST_NAME = '%s'" % (str(playlist_name))
db_cursor.execute(sql)
try:
    mysql_db.commit()
    print('commit')
except:
    print('error')

