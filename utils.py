import os
import traceback
import contextlib

import config as cfg

def list_subdirectories(path: str):
    return filter(lambda el: os.path.isdir(os.path.join(path, el)), os.listdir(path))


def clearErrorLog():
    if os.path.isfile(cfg.ERROR_LOG_FILE):
        os.remove(cfg.ERROR_LOG_FILE)


def writeErrorLog(ex: Exception):
    with open(cfg.ERROR_LOG_FILE, "a") as elog:
        elog.write("".join(traceback.TracebackException.from_exception(ex).format()) + "\n")

def loadSpeciesList(fpath):

    slist = []
    if not fpath == None:
        with open(fpath, 'r', encoding='utf-8') as sfile:
            for line in sfile.readlines():
                species = line.replace('\r', '').replace('\n', '')
                slist.append(species)

    return slist

def loadLabels(labels_file):

    labels = []
    with open(labels_file, 'r', encoding='utf-8') as lfile:
        for line in lfile.readlines():
            labels.append(line.replace('\n', ''))    

    return labels