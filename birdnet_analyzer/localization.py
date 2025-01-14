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
    """
    Ensures that the settings file exists at the specified path. If the file does not exist,
    it creates a new settings file with default settings.

    If the file creation fails, the error is logged.
    """
    if not os.path.exists(GUI_SETTINGS_PATH):
        try:
            with open(GUI_SETTINGS_PATH, "w") as f:
                settings = {"language-id": FALLBACK_LANGUAGE}
                f.write(json.dumps(settings, indent=4))
        except Exception as e:
            utils.writeErrorLog(e)


def get_state_dict() -> dict:
    """
    Retrieves the state dictionary from a JSON file specified by STATE_SETTINGS_PATH.
    
    If the file does not exist, it creates an empty JSON file and returns an empty dictionary.
    If any other exception occurs during file operations, it logs the error and returns an empty dictionary.
    
    Returns:
        dict: The state dictionary loaded from the JSON file, or an empty dictionary if the file does not exist or an error occurs.
    """
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
    """
    Retrieves the value associated with the given key from the state dictionary.

    Args:
        key (str): The key to look up in the state dictionary.
        default: The value to return if the key is not found. Defaults to None.

    Returns:
        str: The value associated with the key if found, otherwise the default value.
    """
    return get_state_dict().get(key, default)


def set_state(key: str, value: str):
    """
    Updates the state dictionary with the given key-value pair and writes it to a JSON file.

    Args:
        key (str): The key to update in the state dictionary.
        value (str): The value to associate with the key in the state dictionary.
    """
    state = get_state_dict()
    state[key] = value
    try:
        with open(STATE_SETTINGS_PATH, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        utils.writeErrorLog(e)


def load_local_state():
    """
    Loads the local language settings and populates the LANGUAGE_LOOKUP dictionary with the appropriate translations.
    This function performs the following steps:
    """
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
    """
    Translates a given key into its corresponding localized string.

    Args:
        key (str): The key to be localized.

    Returns:
        str: The localized string corresponding to the given key. If the key is not found in the localization lookup, the original key is returned.
    """
    return LANGUAGE_LOOKUP.get(key, key)


def set_language(language: str):
    """
    Sets the language for the application by updating the GUI settings file.
    This function ensures that the settings file exists, reads the current settings,
    updates the "language-id" field with the provided language, and writes the updated
    settings back to the file.

    Args:
        language (str): The language identifier to set in the settings file.
    """
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
