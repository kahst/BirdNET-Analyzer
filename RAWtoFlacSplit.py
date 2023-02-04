#!/usr/bin/python

import numpy as np
from datetime import datetime, timedelta
import soundfile as sf
import glob
from multiprocessing import Pool
import librosa
import io


# Variables
samplingRate = 12000 #12k audio, but actual is a bit slower 
splitHours = 24
duration = samplingRate * 60 *60 *splitHours #8 hours

inputPath = "D:\\Audio Import\\QA Audio\\*.pcm"
#inputPath  = "D:\\24k PCM audio\\*.pcm" 
#outputPath = "E:\\BirdNet-Audio\\"
outputPath = "E:\\BirdNet-Audio\\QA Audio"



threads = []


def processFile(file):
    #print(str(file))
    fileNameArr = str(file).split("\\")
    #fileName = fileNameArr[len(fileNameArr) -1]
    #outputFile = outputPath + fileName.replace(".pcm", ".flac")#dateParts[0] + "_" + str(currentDatetime.strftime('%m-%d-%y_%H-%M-%S')) + ".flac"
    baseName = fileNameArr[len(fileNameArr) -1].replace(".pcm","")
    dateParts = baseName.split("_")
    currentDatetime = datetime.strptime(dateParts[1] + "_" + dateParts[2], '%m-%d-%y_%H-%M-%S')
    outputFile = outputPath + dateParts[0] + "_" + str(currentDatetime.strftime('%m-%d-%y_%H-%M-%S')) + ".flac"
    if glob.glob(outputFile):
        return
    offset = 0
    print("Loading: " + file)
    while True:
        data, samplerate = sf.read(file, channels=1, samplerate=samplingRate,subtype='PCM_16', format='RAW', start=offset, frames=duration)
        if len(data) <= 0:
            break
        
        print("writing:" + outputFile)
        sf.write(outputFile, data, samplerate=samplerate)
        offset += duration
        currentDatetime = currentDatetime + timedelta(hours=splitHours)
        outputFile = outputPath + dateParts[0] + "_" + str(currentDatetime.strftime('%m-%d-%y_%H-%M-%S')) + ".flac"
    #data, samplerate = sf.read(file)
    return
    #return (outputFile, data, samplerate)



def writeOutputFile(outputFile, data, samplerate):
    sf.write(outputFile, data, samplerate=samplerate)

if __name__ == '__main__':
    fileLoadPool = Pool(processes=2)

    if not glob.glob(inputPath):
        print("RAW audio not found: " + inputPath)
    else:
        rawFiles = []
        for file in glob.glob(inputPath):
            fileLoadPool.apply_async(processFile,(file,))
            #async_result = fileLoadPool.apply_async(loadFile, (file,))
            #threads.append(async_result)

        fileLoadPool.close()
        fileLoadPool.join()

        #while len(threads) > 0:
        #    t = threads.pop(0)
        #    t.wait()
        #    result = t.get()
        #    print("writing output: " + result[0])
        #    writeOutputFile(result[0],result[1],result[2])

        #fileLoadPool.close()
        #fileLoadPool.join()