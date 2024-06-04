import json

FALLBACK_LANGUAGE = "en"
LANGUAGE_DIR = "lang/"
LANGUAGE_LOOKUP = {}
TARGET_LANGUAGE = FALLBACK_LANGUAGE

def load_localization():
    global LANGUAGE_LOOKUP
    global TARGET_LANGUAGE

    try:
        TARGET_LANGUAGE = json.load(open("gui-settings.json", encoding="utf-8"))["language-id"]
    except FileNotFoundError:
        print(f"gui-settings.json not found. Using fallback language {FALLBACK_LANGUAGE}.")

    try:
        with open(f"{LANGUAGE_DIR}/{TARGET_LANGUAGE}.json", "r", encoding="utf-8") as f:
            LANGUAGE_LOOKUP = json.load(f)
    except FileNotFoundError:
        print(
            f"Language file for {TARGET_LANGUAGE} not found in {LANGUAGE_DIR}. Using fallback language {FALLBACK_LANGUAGE}."
        )

    if TARGET_LANGUAGE != FALLBACK_LANGUAGE:
        with open(f"{LANGUAGE_DIR}/{FALLBACK_LANGUAGE}.json", "r") as f:
            fallback = json.load(f)

        for key, value in fallback.items():
            if key not in LANGUAGE_LOOKUP:
                LANGUAGE_LOOKUP[key] = value


def localize(key: str) -> str:
    return LANGUAGE_LOOKUP.get(key, key)


def set_language(language: str):    
    if language:
        settings = {}

        try:
            with open("gui-settings.json", "r+", encoding="utf-8") as f:
                settings = json.load(f)
                settings["language-id"] = language
                f.seek(0)
                json.dump(settings, f, indent=4)
                f.truncate()

        except FileNotFoundError:
            pass
    
