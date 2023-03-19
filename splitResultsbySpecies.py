#!/usr/bin/python

import numpy as np
import soundfile as sf
import glob
import os





inputPath = "E:\\BirdNet Audio GGOW 2023\\Blue Mountain\\"


filters = {
    'Bubo virginianus_Great Horned Owl': .1,
    'Strix nebulosa_Great Gray Owl': .1,
    'Aegolius acadicus_Northern Saw-whet Owl': .4,
    'Aegolius funereus_Boreal Owl': .4,
    'Asio flammeus_Short-eared Owl': .4,
    'Asio otus_Long-eared Owl': .4,
    'Glaucidium gnoma_Northern Pygmy-Owl': .2,
    'Megascops asio_Eastern Screech-Owl': .4,
    'Megascops kennicottii_Western Screech-Owl': .4,
    'Psiloscops flammeolus_Flammulated Owl': .4,
    'Surnia ulula_Northern Hawk Owl': .4,
    'Tyto alba_Barn Owl': .4,
    'Strix varia_Barred Owl': .4
}


results = {}


def processLine(line):
    parts = line.split('  ')
    initial = parts[0]
    species = parts[1]
    probability = parts[2]
    if species not in results:
        results[species] = []
    
    if float(probability) > filters[species]:
        results[species].append(initial + "  " + species + "  " + probability)


if __name__ == '__main__':

    if not glob.glob(inputPath):
        print("result files not found:" + inputPath)
    else:
        rawFiles = []
        for file in glob.glob(inputPath + "*.txt"):
            results = {}
            print("processing: " + file)
            fileParts = file.split('\\')
            fileName = fileParts[len(fileParts)-1]
            with open(file, 'r') as f:
                for line in f:
                    processLine(line)
            #process results
            for key in results:
                outputPath = inputPath + key
                if not os.path.exists(outputPath):
                    os.makedirs(outputPath, exist_ok=True)
                    print("creating: " + outputPath)

                outputFile = inputPath + key + "\\" + fileName
                with open(outputFile, 'w') as outputResults:
                    print("writing: " + outputFile)
                    for row in results[key]:
                        outputResults.write(row)
                    

