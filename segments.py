import os
import sys
import argparse
import traceback

import numpy as np
from multiprocessing import pool

import config as cfg
import audio
from datetime import datetime
import functools
import librosa

# Set numpy random seed
np.random.seed(cfg.RANDOM_SEED)

def clearErrorLog():

    if os.path.isfile(cfg.ERROR_LOG_FILE):
        os.remove(cfg.ERROR_LOG_FILE)

def writeErrorLog(msg):

    with open(cfg.ERROR_LOG_FILE, 'a') as elog:
        elog.write(msg + '\n')

def detectRType(line):

    if line.lower().startswith('selection'):
        return 'table'
    elif line.lower().startswith('filepath'):
        return 'r'
    elif line.lower().startswith('start (s)'):
        return 'csv'
    else:
        return 'audacity'

def compareFiles(x, y):
    try:
        xfileNameParts = x[0:x.index(".")].split("\\")
        xbaseName = xfileNameParts[len(xfileNameParts) -1].replace(".flac","")
        xdateParts = xbaseName.split("_")
        #print(xdateParts)
        xcurrentDatetime = datetime.strptime(xdateParts[1] + "_" + xdateParts[2], '%Y-%m-%d_T%H-%M-%S')

        yfileNameParts = y[0:y.index(".")].split("\\")
        ybaseName = yfileNameParts[len(yfileNameParts) -1].replace(".flac","")
        ydateParts = ybaseName.split("_")
        ycurrentDatetime = datetime.strptime(ydateParts[1] + "_" + ydateParts[2], '%Y-%m-%d_T%H-%M-%S')

        if xcurrentDatetime >  ycurrentDatetime:
            return 1
        elif xcurrentDatetime < ycurrentDatetime:
            return -1
        else:
            return 0
    except:
        #print("comparison failed")
        return 1

def parseFolders(apath, rpath, allowed_filetypes={'audio': ['wav', 'flac', 'mp3', 'ogg', 'm4a'], 'results': ['txt', 'csv']}):

    data = {}
    # Get all audio files
    for root, dirs, files in os.walk(apath):
        orderedFiles = sorted(files,key=functools.cmp_to_key(compareFiles))
        for f in orderedFiles:
            if f.split('.')[-1].lower() in allowed_filetypes['audio']:
                data[f.rsplit('.', 1)[0]] = {'audio': os.path.join(root, f), 'result': ''}

    # Get all result files
    for root, dirs, files in os.walk(rpath):
        #orderedFiles = sorted(files,key=functools.cmp_to_key(compareFiles))
        for f in files:
            if f.split('.')[-1] in allowed_filetypes['results'] and f.find('.BirdNET.') != -1:
                data[f.split('.BirdNET.')[0]]['result'] = os.path.join(root, f)

    # Convert to list
    flist = []
    for f in data:
        if len(data[f]['result']) > 0:
            flist.append(data[f])

    print('Found {} audio files with valid result file.'.format(len(flist)))

    return flist

def parseFiles(flist, max_segments=100):

    outputList = []
    for f in flist:

        # Paths
        afile = f['audio'] 
        rfile = f['result']

        # Get all segments for result file
        segments = findSegments(afile, rfile)
        outputList.append((f, segments))

    return outputList

def findSegments(afile, rfile):

    segments = []

    # Open and parse result file
    lines = []
    with open(rfile, 'r') as rf:
        for line in rf.readlines():
            lines.append(line.strip())

    # Auto-detect result type
    if len(lines) == 0:
        return segments
    rtype = detectRType(lines[0])

    # Get start and end times based on rtype
    confidence = 0
    for i in range(len(lines)):
        if rtype == 'table' and i > 0:
            d = lines[i].split('\t')
            start = float(d[3])
            end = float(d[4])
            species = d[-2]
            confidence = float(d[-1])

        elif rtype == 'audacity':
            d = lines[i].split('\t')
            start = float(d[0])
            end = float(d[1])
            text = d[2].split("  ")
            species = text[1]
            #print(species)
            confidence = float(text[2])

        elif rtype == 'r' and i > 0:
            d = lines[i].split(',')
            start = float(d[1])
            end = float(d[2])
            species = d[4]
            confidence = float(d[5])

        elif rtype == 'csv' and i > 0:
            d = lines[i].split(',')
            start = float(d[0])
            end = float(d[1])
            species = d[3]
            confidence = float(d[4])

        # Check if confidence is high enough
        if confidence >= cfg.MIN_CONFIDENCE and (species in cfg.SPECIES_LIST or len(cfg.SPECIES_LIST) == 0):
            segments.append({'audio': afile, 'start': start, 'end': end, 'species': species, 'confidence': confidence, 'text': d[2]})
   
    return segments

def extractSegments(item, outputDict):

    # Paths and config
    afile = item[0][0]['audio']
    #afile = item['audio'] 
    segments = item[0][1]
    seg_length = item[1]
    cfg.setConfig(item[2])

    # Status
    print('Extracting segments from {}'.format(afile))
    # Open audio file
    seg_cnt = 1
    segmentOutputList = []
    segmentOutputResults = []
    startSegment = 0
    endSegment = cfg.SIG_LENGTH

    # Make output path
    outpath = cfg.OUTPUT_PATH + "_confidence_" + str(cfg.MIN_CONFIDENCE)
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    
    fileSampleRate = librosa.get_samplerate(afile)
    fileParts = afile.split("\\")
    #outFileName = outpath + "\\" + fileParts[len(fileParts)-1]
    filePrefix_Rate = fileParts[len(fileParts)-1].split("_")[0] + "_" + str(fileSampleRate)
    #print(outFileName)

    if filePrefix_Rate in outputDict:
        segmentList = outputDict[filePrefix_Rate][0]
        sampleCount = 0
        for seg in segmentList:
            sampleCount += len(seg)
            
        startSegment = sampleCount/fileSampleRate
        endSegment = startSegment + cfg.SIG_LENGTH


    for seg in segments:

        try:
            
            # Get start and end times
            #print("start:" + str(seg['start']))
            #print("end:" + str(seg['end']))
            #start = int(seg['start'] * cfg.SAMPLE_RATE)
            #end = int(seg['end'] * cfg.SAMPLE_RATE)
            #offset = ((seg_length * cfg.SAMPLE_RATE) - (end - start)) // 2
            #start = max(0, start - offset)
            #end = end + offset
            #print("extracting: " + str(end-start) + " " + str(start))
            start = seg['start']
            end = seg['end']
            sig, rate = audio.openAudioFileNoResample(afile, cfg.SAMPLE_RATE, duration=cfg.SIG_LENGTH, offset=start)  

            # Make sure segmengt is long enough
            if end > start:

                # Get segment raw audio from signal
                #seg_sig = sig[int(start):int(end)]

                # Save segment
                seg_name = '{:.3f}_{}_{}.wav'.format(seg['confidence'], seg_cnt, seg['audio'].split(os.sep)[-1].rsplit('.', 1)[0])
                #seg_path = os.path.join(outpath, seg_name)
                #audio.saveSignal(sig, seg_path)
                segmentOutputList.append(sig)
                if end - start < cfg.SIG_LENGTH: # signal is cut off by EOF
                    endSegment = startSegment + (end - start)
                    print(startSegment)
                    print(endSegment)
                segmentOutputResults.append(str(startSegment) + "\t" + str(endSegment) + "\t" + seg['text'] + "\n")
                startSegment = endSegment
                endSegment += cfg.SIG_LENGTH
                seg_cnt += 1

        except:

            # Print traceback
            print(traceback.format_exc(), flush=True)

            # Write error log
            msg = 'Error: Cannot extract segments from {}.\n{}'.format(afile, traceback.format_exc())
            print(msg, flush=True)
            writeErrorLog(msg)
            break


    if filePrefix_Rate not in outputDict:
        outputDict[filePrefix_Rate] = ([],[])

    outputDict[filePrefix_Rate][0].extend(segmentOutputList)
    outputDict[filePrefix_Rate][1].extend(segmentOutputResults)

    
    

        

    #audio.saveSignal(outputArray, outFileName.replace(".flac","_segments.flac"),rate)

    #out_string = ''
    #for s in segmentOutputResults:
    #    out_string += s
    #with open(outFileName.replace(".flac","_segments_results.txt"), 'w') as rfile:
    #    rfile.write(out_string)


def loadSpeciesList(fpath):

    slist = []
    if not fpath == None:
        with open(fpath, 'r') as sfile:
            for line in sfile.readlines():
                species = line.replace('\r', '').replace('\n', '')
                slist.append(species)

    return slist

if __name__ == '__main__':

    # Clear error log
    #clearErrorLog()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Extract segments from audio files based on BirdNET detections.')
    parser.add_argument('--audio', default='example/', help='Path to folder containing audio files.')
    parser.add_argument('--results', default='example/', help='Path to folder containing result files.')
    parser.add_argument('--o', default='example/', help='Output folder path for extracted segments.')
    parser.add_argument('--min_conf', type=float, default=0.1, help='Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to 0.1.')
    parser.add_argument('--max_segments', type=int, default=100, help='Number of randomly extracted segments per species.')
    parser.add_argument('--seg_length', type=float, default=3.0, help='Length of extracted segments in seconds. Defaults to 3.0.')
    parser.add_argument('--threads', type=int, default=4, help='Number of CPU threads.')
    parser.add_argument('--slist', default='', help='Path to species list file or folder. If folder is provided, species list needs to be named \"species_list.txt\".')
    

    args = parser.parse_args()

    # Parse audio and result folders
    cfg.FILE_LIST = parseFolders(args.audio, args.results)
    
    # Set output folder
    cfg.OUTPUT_PATH = args.o

    # Set number of threads
    cfg.CPU_THREADS = int(args.threads)

    # Set confidence threshold
    cfg.MIN_CONFIDENCE = max(0.01, min(0.99, float(args.min_conf)))

    cfg.SPECIES_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), args.slist)
    if os.path.isdir(cfg.SPECIES_LIST_FILE):
        cfg.SPECIES_LIST_FILE = os.path.join(cfg.SPECIES_LIST_FILE, 'species_list.txt')
    cfg.SPECIES_LIST = loadSpeciesList(cfg.SPECIES_LIST_FILE)

    if len(cfg.SPECIES_LIST) == 0:
        print('Species list contains {} species'.format(len(cfg.LABELS)))
    else:        
        print('Species list contains {} species'.format(len(cfg.SPECIES_LIST)))

    print(cfg.SPECIES_LIST)
    # Parse file list and make list of segments
    cfg.FILE_LIST = parseFiles(cfg.FILE_LIST, max(1, int(args.max_segments)))



    # Add config items to each file list entry.
    # We have to do this for Windows which does not
    # support fork() and thus each process has to
    # have its own config. USE LINUX!
    flist = []
    for entry in cfg.FILE_LIST:
        flist.append((entry, max(cfg.SIG_LENGTH, float(args.seg_length)), cfg.getConfig()))
    
    outputDict = {}
    # Extract segments   
    #if cfg.CPU_THREADS < 2:
    for entry in flist:
        extractSegments(entry, outputDict)
    
    for key in outputDict.keys():
        #print("processing output for: " + key)
        segmentOutputList, segmentOutputResults = outputDict[key]
        fileRate = key.split("_")[1]
        if len(segmentOutputList) == 0:
            continue
        #print("sol len: " + str(len(segmentOutputList)))
        #print("sor len: " + str(len(segmentOutputResults)))

        outpath = cfg.OUTPUT_PATH + "_confidence_" + str(cfg.MIN_CONFIDENCE)
        outFile = outpath + "\\" + key + ".flac"
        outputAudioArray = np.hstack(segmentOutputList)
        audio.saveSignal(outputAudioArray, outFile,int(fileRate))
        out_string = ''
        for s in segmentOutputResults:
            out_string += s
            with open(outFile.replace(".flac","_results.txt"), 'w') as rfile:
                rfile.write(out_string)


    #out_string = ''
    #for s in segmentOutputResults:
    #    out_string += s
    #with open(outFileName.replace(".flac","_segments_results.txt"), 'w') as rfile:
    #    rfile.write(out_string)


    #else:
    #    with Pool(cfg.CPU_THREADS) as p:
    #        p.map(extractSegments, flist)

    # A few examples to test
    # python3 segments.py --audio example/ --results example/ --o example/segments/ 
    # python3 segments.py --audio example/ --results example/ --o example/segments/ --seg_length 5.0 --min_conf 0.1 --max_segments 100 --threads 4
