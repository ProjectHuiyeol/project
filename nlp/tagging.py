import pandas as pd
import numpy as np
import re
import platform
if platform.system()!='Linux':
    from msvcrt import getch
else:
    from getch import getch

filename = r'C:\project_huiyeol\project\nlp\data_out\sliced20\sliced20Lyrics10401~10450.csv'

try:
    df=pd.read_csv(filename, encoding='utf-8-sig')
except:
    df=pd.read_csv(filename, encoding='cp949')

total=len(df)-1
for idx, row in df.iterrows():
    try:
        judge = np.isnan(row['대분류1'])
    except:
        judge = False
    if judge:
        
        print(f"\n\n{idx}/{total} {row['Sliced_LYRICS']}")
        print("angry:1 dislike:2 fear:3 happy:4 neutral:5 sad:6 surprise:7 complex:8 exit:9")
    
        while True:
            byte_arr=getch()
            if ord(byte_arr) == 49:
                print(" angry")
                df.iat[idx, df.columns.get_loc('대분류1')] = "angry"
                break
            elif ord(byte_arr) == 50:
                print(" dislike")
                df.iat[idx, df.columns.get_loc('대분류1')] = "dislike"
                break
            elif ord(byte_arr) == 51:
                print(" fear")
                df.iat[idx, df.columns.get_loc('대분류1')] = "fear"
                break
            elif ord(byte_arr) == 52:
                print(" happy")
                df.iat[idx, df.columns.get_loc('대분류1')] = "happy"
                break
            elif ord(byte_arr) == 53:
                print(" neutral")
                df.iat[idx, df.columns.get_loc('대분류1')] = "neutral"
                break
            elif ord(byte_arr) == 54:
                print(" sad")
                df.iat[idx, df.columns.get_loc('대분류1')] = "sad"
                break
            elif ord(byte_arr) == 55:
                print(" surprise")
                df.iat[idx, df.columns.get_loc('대분류1')] = "surprise"
                break
            elif ord(byte_arr) == 56:
                print(" complex")
                df.iat[idx, df.columns.get_loc('대분류1')] = "complex"
                break
            elif ord(byte_arr) == 57:
                print(" 종료")
                exit()
            else:
                pass
        print()
    df.to_csv(filename, encoding='utf-8-sig', index=False)