import pymysql
import numpy as np
import pandas as pd

MYSQL_HOST = 'localhost'
MYSQL_CONN = pymysql.connect(
    host=MYSQL_HOST,
    port=3306,
    user='root',
    passwd='1234',
    db='song_db',
    charset='utf8'
)

def conn_mysqldb():
    if not MYSQL_CONN.open:
        MYSQL_CONN.ping(reconnect=True)
    return MYSQL_CONN

mysql_db = conn_mysqldb()
db_cursor = mysql_db.cursor()
sql = "select song_info_lyrics.song_id, song_info_lyrics.song_lyrics from song_info_lyrics;"
db_cursor.execute(sql)
data = db_cursor.fetchall()
print(data[:2])

