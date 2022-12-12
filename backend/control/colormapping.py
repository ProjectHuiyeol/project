import numpy as np
from model import sql_connect
from control import song_management as sm
from control import playlist_management as pm

color_names = ['하양',
 '아이보리',
 '노랑',
 '금색',
 '분홍',
 '주황색',
 '골든옐로우',
 '코랄',
 '다홍',
 '장미',
 '카민',
 '빨강',
 '살구',
 '베이지',
 '귤색',
 '유황',
 '오페라',
 '세리스',
 '크림슨',
 '탠',
 '카키',
 '은색',
 '라임',
 '민트',
 '회적',
 '황토',
 '연두',
 '버건디',
 '하늘',
 '자주',
 '회색',
 '갈색',
 '보라',
 '올리브',
 '스페이스그레이',
 '울트라마린',
 '청록',
 '춘록',
 '초록',
 '바다',
 '코발트블루',
 '파랑',
 '남색',
 '검정']

color_rgb = [[255, 255, 255],
 [255, 255, 240],
 [255, 255, 0],
 [255, 215, 0],
 [255, 192, 203],
 [255, 165, 0],
 [255, 140, 0],
 [255, 127, 80],
 [255, 36, 0],
 [255, 0, 127],
 [255, 0, 64],
 [255, 0, 0],
 [253, 188, 180],
 [245, 245, 220],
 [245, 151, 0],
 [241, 221, 56],
 [236, 17, 143],
 [222, 49, 99],
 [220, 20, 60],
 [210, 180, 140],
 [189, 183, 107],
 [192, 192, 192],
 [191, 255, 0],
 [189, 252, 201],
 [206, 170, 173],
 [184, 134, 11],
 [154, 205, 50],
 [144, 0, 32],
 [135, 206, 235],
 [128, 0, 128],
 [128, 128, 128],
 [165, 42, 42],
 [127, 0, 255],
 [128, 128, 0],
 [103, 103, 103],
 [35, 71, 148],
 [13, 152, 186],
 [0, 255, 127],
 [0, 128, 0],
 [0, 127, 255],
 [0, 71, 171],
 [0, 0, 255],
 [0, 0, 128],
 [0, 0, 0]]

color_info = {
    'happy' : [253, 251, 84],
    'fear' : [3,149,6],
    'angry' : [179,6,7],
    'dislike' : [224,89,232],
    'surprise' : [91,190,255],
    'sad' : [82,83,225]
}

music_color = {
    'happy' : [253, 251, 84],
    'sad' : [82,83,225],
    'angry' : [179,6,7],
    'relaxed' : [82,252,81]
}

rgb_dict = {0:[0,75,142], 1:[251,171,76], 2:[203,233,243], 3:[198,230,121], 4:[255,78,109], 5:[208,28,31]}

def make_color(lyric, melody):
    multiple = np.array([max(melody[0], melody[3]), melody[2], melody[1]])
    c_r, c_g, c_b = 0, 0 ,0
    for i in range(6):
        c_r += list(color_info.values())[i][0] * lyric[i]
        c_g += list(color_info.values())[i][1] * lyric[i]
        c_b += list(color_info.values())[i][2] * lyric[i]
    
    result = np.array([c_r, c_g, c_b]) * np.add(multiple, 1)
    result = np.array([255 if x >= 255 else x for x in result.reshape(-1)]).reshape(-1,3)
    ans = np.argmin(np.power(np.array(result) - np.array(color_rgb), 2).sum(axis=1))
    return color_names[ans], result


def find_nearest_song(lyric, sound, counts=1):
    mysql_db = sql_connect.conn_mysqldb()
    db_cursor = mysql_db.cursor()
    sql = '''select * from song_info_sound'''
    db_cursor.execute(sql)
    all_sound = db_cursor.fetchall()
    
    mysql_db = sql_connect.conn_mysqldb()
    db_cursor = mysql_db.cursor()
    sql = '''select * from song_info_lyrics'''
    db_cursor.execute(sql)
    all_lyrics = db_cursor.fetchall()
    # print(all_lyrics[0],all_sound[0])
    concat = np.concatenate((np.array(all_lyrics)[:,2:].astype(np.float), np.array(all_sound)[:,2:].astype(np.float)), axis=1)
    concat_input = list(lyric) + list(sound)

    nearest = np.argsort(np.sum(np.power(concat - np.array(concat_input), 2), axis=1))[:counts]
    
    result_song_id = []
    for el in nearest:
        result_song_id.append(all_sound[el][0])
    
    return result_song_id
    
def find_song_from_test(result_num, counts=1):
    test_rgb = rgb_dict[result_num]
    
    mysql_db = sql_connect.conn_mysqldb()
    db_cursor = mysql_db.cursor()
    sql = '''select * from song_info_sound'''
    db_cursor.execute(sql)
    all_sound = db_cursor.fetchall()
    
    mysql_db = sql_connect.conn_mysqldb()
    db_cursor = mysql_db.cursor()
    sql = '''select * from song_info_lyrics'''
    db_cursor.execute(sql)
    all_lyrics = db_cursor.fetchall()
    
    
    rgbs = []
    for i in range(len(all_sound)):
        rgbs.append(make_color(lyric=list(all_lyrics[i])[2:], melody=list(all_sound[i])[2:])[1][0])
    nearest = np.argsort(np.sum(np.power(np.array(rgbs) - np.array(test_rgb), 2), axis=1))[:5*counts]
    rand_mask = np.random.choice([i for i in range(counts*5)], counts, replace=False)
    pick = nearest[rand_mask]
    
    result_song_id = []
    for el in pick:
        result_song_id.append(all_sound[el][0])
    
    return result_song_id

if __name__ == '__main__':
    print(find_song_from_test(3,5))

# 화창한 날에
# 멀어진 이유
# 다시 사랑 할 수 있을까
# 헤어지는 중입니다
# 눈 떠보니 이별이더라

# Honesyt
# As I am
# Wonder
# Wraith
# Can you Feel the Love Tonight

# 기다리는 시간
# 갈바람
# 너에게 간다
# 가을의 전설
# I Can't Never Stop

# 상남자의 고백
# 누군가의 마음이 되면
# 웃음꽃
# 시계 바늘
# Falling

# Lonely
# Gangsta Gangsta
# Nothing But Trouble
# Stand Up And Shout
# Testify

# Single Ladies
# World of the Forgotten
# Who's Got You Singing Again
# Unshaken
# Elephant
