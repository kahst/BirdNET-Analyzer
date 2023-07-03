from birdnet.configuration import config
from birdnet.translations.locale_data_getting import get_locale_data


def translate_species_names(locale: str):
    """Translates species names for a locale.

    Translates species names for the given language with the eBird API.

    Args:
        locale: Two character string of a language.

    Returns:
        The translated list of labels.
    """
    print(f"Translating species names for {locale}...", end="", flush=True)

    # Get locale data
    data = get_locale_data(locale)

    # Create list of translated labels
    labels: list[str] = []

    for l in config.LABELS:
        has_translation = False
        for entry in data:
            if l.split("_", 1)[0] == entry["sciName"]:
                labels.append(f'{l.split("_", 1)[0]}_{entry["comName"]}')
                has_translation = True
                break
        if not has_translation:
            labels.append(l)

    print("Done.", flush=True)

    return labels
