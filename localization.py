import json

FALLBACK_LANGUAGE = "en"
LANGUAGE_DIR = "lang/"
LANGUAGE_LOOKUP = {}


def load_localization():
    global LANGUAGE_LOOKUP

    target_language = json.load(open("gui-settings.json"))["language-id"]

    try:
        with open(f"{LANGUAGE_DIR}/{target_language}.json", "r") as f:
            LANGUAGE_LOOKUP = json.load(f)
    except FileNotFoundError:
        print(
            f"Language file for {target_language} not found in {LANGUAGE_DIR}. Using fallback language {FALLBACK_LANGUAGE}."
        )

    if target_language != FALLBACK_LANGUAGE:
        with open(f"{LANGUAGE_DIR}/{FALLBACK_LANGUAGE}.json", "r") as f:
            fallback = json.load(f)

        for key, value in fallback.items():
            if key not in LANGUAGE_LOOKUP:
                LANGUAGE_LOOKUP[key] = value


def localize(key: str) -> str:
    return LANGUAGE_LOOKUP.get(key, key)
