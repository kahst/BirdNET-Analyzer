"""Module for translating species labels.

Can be used to translate species names into other languages.

Uses the requests to the eBird-API.
"""
from birdnet.configuration import config
from birdnet.translations.constants import LOCALES
from birdnet.translations.labels_file_saving import save_labels_file
from birdnet.translations.species_names_translating import \
    translate_species_names
from birdnet.utils.lines_reading import read_lines


if __name__ == "__main__":
    # Load labels
    config.LABELS = read_lines(config.LABELS_FILE)

    # Translate labels
    for locale in LOCALES:
        labels = translate_species_names(locale)
        save_labels_file(labels, locale)
