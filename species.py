import sys
import os
import argparse

import config as cfg
import analyze
import model

def getSpeciesList(lat, lon, week, threshold=0.05, sort=False):

    print('Getting species list for {}/{}, Week {}...'.format(lat, lon, week), end='', flush=True)

    # Extract species from model
    pred = model.explore(lat, lon, week)

    # Make species list
    slist = []
    for p in pred:
        if p[0] >= threshold:
            slist.append(p[1])

    print('Done. {} species on list.'.format(len(slist)), flush=True)

    if sort:
        slist = sorted(slist)

    return slist

if __name__ == '__main__':    

    # Parse arguments
    parser = argparse.ArgumentParser(description='Get list of species for a given location with BirdNET. Sorted by occurrence frequency.')
    parser.add_argument('--o', default='example/', help='Path to output file or folder. If this is a folder, file will be named \'species_list.txt\'.')
    parser.add_argument('--lat', type=float, help='Recording location latitude.')
    parser.add_argument('--lon', type=float, help='Recording location longitude.')
    parser.add_argument('--week', type=int, default=-1, help='Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.')
    parser.add_argument('--threshold', type=float, default=0.05, help='Occurrence frequency threshold. Defaults to 0.05.')
    parser.add_argument('--sortby', default='freq', help='Sort species by occurrence frequency or alphabetically. Values in [\'freq\', \'alpha\']. Defaults to \'freq\'.')
    
    args = parser.parse_args()

    # Set paths relative to script path (requested in #3)
    cfg.LABELS_FILE = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), cfg.LABELS_FILE)
    cfg.MDATA_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), cfg.MDATA_MODEL_PATH)

    # Load eBird codes, labels
    cfg.LABELS = analyze.loadLabels(cfg.LABELS_FILE)

    # Set output path
    cfg.OUTPUT_PATH = args.o
    if os.path.isdir(cfg.OUTPUT_PATH):
        cfg.OUTPUT_PATH = os.path.join(cfg.OUTPUT_PATH, 'species_list.txt')

    # Set config
    cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK = args.lat, args.lon, args.week
    cfg.LOCATION_FILTER_THRESHOLD = args.threshold

    # Get species list
    species_list = getSpeciesList(cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK, cfg.LOCATION_FILTER_THRESHOLD, False if args.sortby == 'freq' else True)

    # Save species list
    with open(cfg.OUTPUT_PATH, 'w') as f:
        for s in species_list:
            f.write(s + '\n')

    # A few examples to test
    # python3 species.py --o example/ --lat 42.5 --lon -76.45 --week -1
    # python3 species.py --o example/species_list.txt --lat 42.5 --lon -76.45 --week 4 --threshold 0.05 --sortby alpha