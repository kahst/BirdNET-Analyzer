#!/usr/bin/python

import numpy as np
from datetime import datetime, timedelta
import soundfile as sf
import glob
from multiprocessing import Pool, Manager

import librosa
import io
import time
import os

# Variables
samplingRate = 12000 #12k audio, but actual is a bit slower 

inputPath = "F:\\2023 GGOW audio raw import\\*.pcm"
#inputPath  = "D:\\24k PCM audio\\*.pcm" 
#outputPath = "E:\\BirdNet-Audio\\"
#outputPath = "E:\\BirdNet-Audio\\Outdoor Tests\\"
outputPath = "D:\\BirdNet Audio GGOW 2023\\Blue Mountain\\"


def readFile(file, lock):
    #try:
    print(file)
    fileNameArr = str(file).split("\\")
    #fileName = fileNameArr[len(fileNameArr) -1]
    #outputFile = outputPath + fileName.replace(".pcm", ".flac")#dateParts[0] + "_" + str(currentDatetime.strftime('%m-%d-%y_%H-%M-%S')) + ".flac"
    baseName = fileNameArr[len(fileNameArr) -1].replace(".pcm","")
    dateParts = baseName.split("_")
    currentDatetime = datetime.strptime(dateParts[1] + "_" + dateParts[2], '%m-%d-%y_%H-%M-%S')
    outputFile = outputPath + dateParts[0] + "_" + str(currentDatetime.strftime('%Y-%m-%d_T%H-%M-%S')) + ".flac"
    #outputFile = outputPath + fileName.replace(".pcm",".flac")
    if glob.glob(outputFile):
        return
    print("Loading: " + file)

    lock.acquire()
    data, samplerate = sf.read(file, channels=1, samplerate=samplingRate,subtype='PCM_16', format='RAW')
    lock.release()
    modTime = os.path.getmtime(file)
    print("Writing: " + outputFile)
    sf.write(outputFile, data, samplerate=samplerate)
    os.utime(outputFile, (modTime, modTime))
    #except:
        #print("error")



if __name__ == '__main__':
    manager = Manager()
    lock = manager.Lock()
    processPool = Pool(processes=3)
    if not glob.glob(outputPath):
        print("Creating: " + outputPath)
        os.mkdir(outputPath)

    if not glob.glob(inputPath):
        print("RAW audio not found: " + inputPath)
    else:
        rawFiles = []
        
        for file in glob.glob(inputPath):
            result = processPool.apply_async(readFile,(file,lock))

    processPool.close()
    processPool.join()
            #pool.map(lambda file: readFile(file, lock), glob.glob(inputPath))
            #for file in glob.glob(inputPath):
                    #print(file)
                    #fileLoadPool.apply_async(readFile,(file,))
            



