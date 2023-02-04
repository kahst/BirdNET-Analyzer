#!/usr/bin/python


from datetime import datetime, timedelta

import glob
import os
import shutil
import time



inputPath = "G:\\audio\\*.pcm"
#inputPath  = "D:\\24k PCM audio\\*.pcm" 
#outputPath = "E:\\BirdNet-Audio\\"
outputPath = "D:\\Audio Import\\QA HF12 12k 7mhz spi\\"






#def writeOutputFile(outputFile, data, samplerate):
    #sf.write(outputFile, data, samplerate=samplerate)

if __name__ == '__main__':

    while(True):
        if not glob.glob(inputPath):
            print("RAW audio not found: " + inputPath)
        else:
            rawFiles = sorted(glob.glob(inputPath), key=os.path.getmtime)
            for i in range(1, 3):
                file = rawFiles[len(rawFiles)-i]
                fileNameArr = str(file).split("\\")
                fileName = fileNameArr[len(fileNameArr) -1]
                outputFile = outputPath + fileName
                if not glob.glob(outputFile):
                    print("Copying: " + file)
                    shutil.copy2(file,outputFile)
                    print("Done")

        time.sleep(1)
    
