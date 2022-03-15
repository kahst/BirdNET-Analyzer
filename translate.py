import os
import json
import urllib.request

import config as cfg
import analyze

# Locales for 25 common languages (according to GitHub Copilot) 
LOCALES = ['af', 'ar', 'cs', 'da', 'de', 'es', 'fi', 'fr', 'hu', 'it', 'ja', 'ko', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sv', 'th', 'tr', 'uk', 'zh']

# Sign up for your personal access token here: https://ebird.org/api/keygen
API_TOKEN = 'yourAPIToken'

def getLocaleData(locale):

    url = 'https://api.ebird.org/v2/ref/taxonomy/ebird?cat=species&fmt=json&locale=' + locale
    header = {'X-eBirdAPIToken': API_TOKEN}

    req = urllib.request.Request(url, headers=header)
    response = urllib.request.urlopen(req)

    return json.loads(response.read())

def translate(locale):

    print('Translating species names for {}...'.format(locale), end='', flush=True)

    # Get locale data
    data = getLocaleData(locale)

    # Create list of translated labels
    labels = []
    for l in cfg.LABELS:
        has_translation = False
        for entry in data:
            if l.split('_')[0] == entry['sciName']:
                labels.append('{}_{}'.format(l.split('_')[0], entry['comName']))
                has_translation = True
                break
        if not has_translation:
            labels.append(l)

    print('Done.', flush=True)

    return labels

def saveLabelsFile(labels, locale):

    # Create folder
    if not os.path.exists(cfg.TRANSLATED_LABELS_PATH):
        os.makedirs(cfg.TRANSLATED_LABELS_PATH)

    # Save labels file
    fpath = os.path.join(cfg.TRANSLATED_LABELS_PATH, '{}_{}.txt'.format(os.path.basename(cfg.LABELS_FILE).rsplit('.', 1)[0], locale))
    with open(fpath, 'w') as f:
        for l in labels:
            f.write(l + '\n')


if __name__ == '__main__':

    # Load labels
    cfg.LABELS = analyze.loadLabels(cfg.LABELS_FILE)

    # Translate labels
    for locale in LOCALES:
        labels = translate(locale)
        saveLabelsFile(labels, locale)