from PyQt5 import QtWidgets
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *   
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from os import path
import numpy as np
import sys
import os
import librosa 
from pydub import AudioSegment
from tempfile import mktemp
import librosa.display
import numpy as np
from PIL import Image
import imagehash
import csv
from imagehash import hex_to_hash
import logging 


logging.basicConfig(level=logging.INFO, filename="logging.log", format='%(asctime)s:%(levelname)s:%(message)s', filemode='w') 

   
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MAIN_WINDOW,_=loadUiType(path.join(path.dirname(__file__),"main_1.ui"))

class MainApp(QMainWindow,MAIN_WINDOW):
  
    def __init__(self,parent=None):
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.Buttons= [self.Browse1 , self.Browse2 , self.Identify]
        self.Buttons[2].setDisabled(True) 
        self.songs= [None,None]
        self.outMix= None
        self.Buttons[0].clicked.connect(lambda : self.readSong(0) )
        self.Buttons[1].clicked.connect(lambda : self.readSong(1) )
        self.Buttons[2].clicked.connect(self.songMixer)
        self.mixerSlider.valueChanged.connect(self.songMixer)
        self.outMixList = [None , None , None, None]
        self.table = {}
        self.SongColumnDB = []
        self.MfccColumnDB = []
        self.melSpectroColumnDB = []
        self.Chroma_stftColumnDB = []
        self.SimilarityIndexes=[]
        self.songData=[None,None]
        self.samplingFrequencies=[None,None]
        self.SongNames=[self.firstSongName,self.secondSongName]
    def readSong(self,songNumber):
        fileName= QFileDialog.getOpenFileName( self, 'choose the signal', os.getenv('HOME') ,"mp3(*.mp3)" ) 
        self.path = fileName[0] 
        modifiedAudio = AudioSegment.from_file( self.path , format="mp3")[:60000]  # read mp3
        wname = mktemp('.wav')  # use temporary file
        modifiedAudio.export(wname, format="wav")  # convert to wav
        self.SongNames[songNumber].setText(os.path.splitext(os.path.basename(self.path))[0])
        self.songData[songNumber],self.samplingFrequencies[songNumber] =librosa.load(wname)
        self.songs[songNumber]= self.songData[songNumber]
        self.Buttons[2].setDisabled(False) 
    def songMixer(self) :
        sliderRatio = self.mixerSlider.value()/100

        if (self.songs[0] is not None) and (self.songs[1] is not None):
            logger.debug("Two different audio files are loaded ")
            self.outMix = self.songs[0] * sliderRatio + self.songs[1] * (1-sliderRatio)
        
        else:
            if self.songs[0] is not None : self.outMix = self.songs[0]
            if self.songs[1] is not None: self.outMix = self.songs[1]
           
        self.ReadFromDB()
    def ReadFromDB(self):
        DataFromDataBase = csv.reader(open('CreateDB/songsDataBase.csv', 'r'), delimiter=",")
        SongColumnDB = []
        MfccColumnDB = []
        melSpectroColumnDB = []
        Chroma_stftColumnDB = []
        
        for column in DataFromDataBase:
            SongColumnDB.append(column[0])
            MfccColumnDB.append(column[1])
            melSpectroColumnDB.append(column[2])
            Chroma_stftColumnDB.append(column[3])

        feature1a= librosa.feature.chroma_stft(y=self.outMix.astype ('float64') , sr=self.samplingFrequencies[0])
        feature1b= librosa.feature.melspectrogram(y=self.outMix.astype ('float64') , sr=self.samplingFrequencies[0])
        feature1c= librosa.feature.mfcc(y=self.outMix.astype ('float64') , sr=self.samplingFrequencies[0])
        feature1d= librosa.amplitude_to_db(np.abs(librosa.stft(self.outMix.astype('float64'))), ref=np.max)
        OutputSongFeatures=[feature1a,feature1b,feature1c,feature1d]
        FeatureHashs=[]
        
        for feature in OutputSongFeatures:
            ImageFeature=Image.fromarray(feature)
            FeatureHash=imagehash.phash((ImageFeature),hash_size=16).__str__()
            FeatureHashs.append(FeatureHash)
        i=0
        SimilarityIndexes=[]
        for hashnum in MfccColumnDB:
            Feature1_Diff=hex_to_hash(Chroma_stftColumnDB[i])-hex_to_hash(FeatureHashs[0])
            Feature2_Diff=hex_to_hash(melSpectroColumnDB[i])-hex_to_hash(FeatureHashs[1])
            Feature3_Diff=hex_to_hash(MfccColumnDB[i])-hex_to_hash(FeatureHashs[2])
            DifferencedHash=((Feature1_Diff+1.5*Feature2_Diff+Feature3_Diff)/3)
            SimilarityIndex=(1- DifferencedHash/255)*100
            SimilarityIndexes.append([SongColumnDB[i],SimilarityIndex])
            i+=1
        SimilarityIndexes.sort(key=lambda x:x[1])
        self.newSimilarityIndexes=list(reversed(SimilarityIndexes))
        self.startTable()
    def startTable(self):

            logger.debug("Showing similar audio files")
            self.resultsTable.setColumnCount(2)
            self.resultsTable.setRowCount(10)

            for row in range(10):
                self.resultsTable.setItem(row, 0, QtWidgets.QTableWidgetItem(self.newSimilarityIndexes[row][0]))
                self.resultsTable.setItem(row, 1, QtWidgets.QTableWidgetItem(str(round(self.newSimilarityIndexes[row][1], 2))+"%"))
                self.resultsTable.verticalHeader().setSectionResizeMode(row, QtWidgets.QHeaderView.Stretch)

            self.resultsTable.setHorizontalHeaderLabels(["Found Matches", "Percentage"])

            for col in range(2):
                self.resultsTable.horizontalHeader().setSectionResizeMode(col, QtWidgets.QHeaderView.Stretch)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()