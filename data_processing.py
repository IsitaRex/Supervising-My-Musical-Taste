import os
from pydub import AudioSegment

def cut_song(path, name, class_num):
    AudioSegment.ffmpeg = "/absolute/path/to/ffmpeg"
    sound = AudioSegment.from_mp3(path)
    #Selecting Portion we want to cut
    StrtMin = 0
    StrtSec = 0
    EndMin = 1
    EndSec = 1
    # Time to milliseconds conversion
    StrtTime = StrtMin*60*1000+StrtSec*1000
    EndTime = EndMin*60*1000+EndSec*1000
    # Opening file and extracting portion of it
    extract = sound[StrtTime:EndTime]
    # Saving file in required location
    if(class_num == 0):
        extract.export('Data/Like/'+name, format="mp3")
    else:
        extract.export('Data/Dislike/'+name, format="mp3")
    # new file portion.mp3 is saved at required location

class_ = ["Dataset2/Like/" , "Dataset2/Dislike/"]
class_num = 0
for cla in class_:
    cont = 1
    for fil in os.listdir(cla):
        print(fil)
        name = str(cont)
        while(len(name)<3):
            name = '0'+name
        cut_song(cla+fil, name+'.mp3', class_num)
        cont += 1
    class_num += 1