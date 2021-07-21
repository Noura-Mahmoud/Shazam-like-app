import pandas as pd
import os
from PIL import Image
import imagehash
import librosa 
from pydub import AudioSegment
from tempfile import mktemp
import librosa.display
import os
import pandas as pd
import numpy as np



Song = []
spectrograms = []
mfccHashList = []
melSpectroHashList=[]
chromaHashList=[]

   
for filename in os.listdir():
    if filename.endswith(".mp3"):
        
        mp3_audio = AudioSegment.from_file((filename), format="mp3")[:60000]  # read mp3
        wname = mktemp('.wav')  # use temporary file
        mp3_audio.export(wname, format="wav")  # convert to wav
        wavsong,samplingFrequency =librosa.load(wname)

        feature1= librosa.feature.mfcc(y=wavsong, sr=samplingFrequency)
        feature2= librosa.feature.melspectrogram(y=wavsong, sr=samplingFrequency)
        feature3= librosa.feature.chroma_stft(y=wavsong, sr=samplingFrequency)

        new_image = Image.fromarray(feature1)
        new_image2 = Image.fromarray(feature2)
        new_image3 = Image.fromarray(feature3)

        firstHash = imagehash.phash((new_image), hash_size=16).__str__()
        secondHash = imagehash.phash((new_image2), hash_size=16).__str__()
        thirdHash = imagehash.phash((new_image3), hash_size=16).__str__()

        Song.append(filename)
        mfccHashList.append(firstHash)
        melSpectroHashList.append(secondHash)
        chromaHashList.append(thirdHash)
    



dict = {'song': Song , 'mfcc hash': mfccHashList,'mel spectrogram hash':melSpectroHashList,'Chroma_stft hash':chromaHashList}
df = pd.DataFrame(dict)
print(df.head)
df.to_csv('CreateDB/NewSongsDataBase.csv',index=False,header=False)