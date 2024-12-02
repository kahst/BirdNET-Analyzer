import json
import os

import birdnet_analyzer.utils as utils

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
FALLBACK_LANGUAGE = "en"
LANGUAGE_DIR = os.path.join(SCRIPT_DIR, "lang")
LANGUAGE_LOOKUP = {}
TARGET_LANGUAGE = FALLBACK_LANGUAGE
GUI_SETTINGS_PATH = os.path.join(SCRIPT_DIR, "gui-settings.json")
STATE_SETTINGS_PATH = os.path.join(SCRIPT_DIR, "state.json")


def ensure_settings_file():
    if not os.path.exists(GUI_SETTINGS_PATH):
        try:
            with open(GUI_SETTINGS_PATH, "w") as f:
                settings = {"language-id": FALLBACK_LANGUAGE}
                f.write(json.dumps(settings, indent=4))
        except Exception as e:
            utils.writeErrorLog(e)


def get_state_dict() -> dict:
    try:
        with open(STATE_SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        try:
            with open(STATE_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump({}, f)
            return {}
        except Exception as e:
            utils.writeErrorLog(e)
            return {}


def get_state(key: str, default=None) -> str:
    return get_state_dict().get(key, default)


def set_state(key: str, value: str):
    state = get_state_dict()
    state[key] = value
    try:
        with open(STATE_SETTINGS_PATH, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        utils.writeErrorLog(e)


def load_local_state():
    global LANGUAGE_LOOKUP
    global TARGET_LANGUAGE

    ensure_settings_file()

    try:
        TARGET_LANGUAGE = json.load(open(GUI_SETTINGS_PATH, encoding="utf-8"))["language-id"]
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
        ensure_settings_file()
        settings = {}

        try:
            with open(GUI_SETTINGS_PATH, "r+", encoding="utf-8") as f:
                settings = json.load(f)
                settings["language-id"] = language
                f.seek(0)
                json.dump(settings, f, indent=4)
                f.truncate()

        except FileNotFoundError:
            pass
