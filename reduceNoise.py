#!/usr/bin/python

from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf

# Variables
samplingRate = 12000 #12k audio, but actual is a bit slower 

inputPath = "D:\\Audio Import\\QA Audio\\*.pcm"
#inputPath  = "D:\\24k PCM audio\\*.pcm" 
#outputPath = "E:\\BirdNet-Audio\\"
outputPath = "E:\\BirdNet-Audio\\QA Audio\\"


#def writeOutputFile(outputFile, data, samplerate):
    #sf.write(outputFile, data, samplerate=samplerate)

if __name__ == '__main__':

# load data
    inputFile = "E:\\BirdNet-Audio\\QA Audio\\OS-15-1_1-26-23_17-30-11.flac"
    outputFile = inputFile.replace(".flac","-NR.flac")
    data, rate = sf.read(inputFile)
# perform noise reductionre
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    sf.write(outputFile,reduced_noise, rate)