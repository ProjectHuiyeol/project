import pymysql

# MYSQL_HOST = 'localhost'
# MYSQL_CONN = pymysql.connect(
#     host=MYSQL_HOST,
#     port=3306,
#     user='root',
#     passwd='1234',
#     db='song_db',
#     charset='utf8'
# )

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