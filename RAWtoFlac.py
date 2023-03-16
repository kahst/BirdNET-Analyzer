#!/usr/bin/python

import numpy as np
from datetime import datetime, timedelta
import soundfile as sf
import glob
from multiprocessing import Pool, Lock
import librosa
import io
import time
import os

###THIS IS ONLY EXTRACTING 1 HOUR in the middle of the recording
# Variables
samplingRate = 12000 #12k audio, but actual is a bit slower 

inputPath = "D:\\Audio Import\\Outdoor raw\\*.pcm"
#inputPath  = "D:\\24k PCM audio\\*.pcm" 
#outputPath = "E:\\BirdNet-Audio\\"
#outputPath = "E:\\BirdNet-Audio\\Outdoor Tests\\"
outputPath = "D:\\Audio Import\\QA Final\\"


threads = []

def processFile(file):
    #print(str(file))
    fileNameArr = str(file).split("\\")
    fileName = fileNameArr[len(fileNameArr) -1]
    #outputFile = outputPath + fileName.replace(".pcm", ".flac")#dateParts[0] + "_" + str(currentDatetime.strftime('%m-%d-%y_%H-%M-%S')) + ".flac"
    outputFile = outputPath + fileName.replace(".pcm",".flac")
    if glob.glob(outputFile):
        return
    offset = 0
    print("Loading: " + file)

    duration = samplingRate * 60 *60 *1 #1 hours
    start = samplingRate * 60 *60 *8

    #data, samplerate = sf.read(file, channels=1, samplerate=samplingRate,subtype='PCM_16', format='RAW', start=start, frames=duration)
    data, samplerate = sf.read(file, channels=1, samplerate=samplingRate,subtype='PCM_16', format='RAW')


    print("Writing: " + outputFile)
    sf.write(outputFile, data, samplerate=samplerate)
    modTime = os.path.getmtime(file)
    #date = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    #modTime = time.mktime(date.timetuple())

    os.utime(outputFile, (modTime, modTime))
    #data, samplerate = sf.read(file)
    return
    #return (outputFile, data, samplerate)


#def writeOutputFile(outputFile, data, samplerate):
    #sf.write(outputFile, data, samplerate=samplerate)

if __name__ == '__main__':
    #fileLoadPool = Pool(processes=2)
    #lock = Lock()
    #fileLoadPool = Pool(processes=1, initializer=init, initargs=(lock,))

    if not glob.glob(inputPath):
        print("RAW audio not found: " + inputPath)
    else:
        rawFiles = []
        for file in glob.glob(inputPath):
            processFile(file)
            #async_result = fileLoadPool.apply_async(loadFile, (file,))
            #threads.append(async_result)
