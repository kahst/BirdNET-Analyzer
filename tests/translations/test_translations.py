from birdnet.configuration import config
from birdnet.translations.labels_file_saving import save_labels_file
from birdnet.translations.species_names_translating import \
    translate_species_names
from birdnet.utils.lines_reading import read_lines

from tests._paths import ROOT_PATH


def test_translations():
    LOCALES = ['af', 'ar', 'cs', 'da', 'de', 'es', 'fi', 'fr', 'hu', 'it',
               'ja', 'ko', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl',
               'sv', 'th', 'tr', 'uk', 'zh']

    config.LABELS_FILE = str(ROOT_PATH / config.LABELS_FILE)
    config.LABELS = read_lines(config.LABELS_FILE)

    # Translate labels
    for locale in LOCALES[:1]:
        labels = translate_species_names(locale)
        save_labels_file(labels, locale)
