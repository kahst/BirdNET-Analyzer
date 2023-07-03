import json
import urllib.request

from birdnet.translations.constants import API_TOKEN


def get_locale_data(locale: str):
    """Download eBird locale species data.

    Requests the locale data through the eBird API.

    Args:
        locale: Two character string of a language.

    Returns:
        A data object containing the response from eBird.
    """
    url = \
        f"https://api.ebird.org/v2/ref/taxonomy/ebird" \
        f"?cat=species" \
        f"&fmt=json" \
        f"&locale={locale}"
    header = {"X-eBirdAPIToken": API_TOKEN}

    req = urllib.request.Request(url, headers=header)
    response = urllib.request.urlopen(req)

    return json.loads(response.read())
