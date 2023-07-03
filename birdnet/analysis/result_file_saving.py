from birdnet.configuration import config
from birdnet.analysis.sorted_timestamps_getting import get_sorted_timestamps

import os


def save_result_file(r: dict[str, list], path: str, afile_path: str):
    """Saves the results to the hard drive.
    Args:
        r: The dictionary with {segment: scores}.
        path: The path where the result should be saved.
        afile_path: The path to audio file.
    """
    # Make folder if it doesn't exist
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Selection table
    out_string = ""

    if config.RESULT_TYPE == "table":
        # Raven selection header
        header = \
            "Selection" \
            "\tView\tChannel\tBegin Time (s)\tEnd Time (s)" \
            "\tLow Freq (Hz)\tHigh Freq (Hz)\tSpecies Code" \
            "\tCommon Name\tConfidence" \
            "\n"
        selection_id = 0

        # Write header
        out_string += header

        # Extract valid predictions for every timestamp
        for timestamp in get_sorted_timestamps(r):
            rstring = ""
            start, end = timestamp.split("-", 1)

            for c in r[timestamp]:
                if c[1] > config.MIN_CONFIDENCE and (
                    not config.SPECIES_LIST or c[0] in config.SPECIES_LIST
                ):
                    selection_id += 1
                    label = config.TRANSLATED_LABELS[config.LABELS.index(c[0])]
                    rstring += \
                        "{}\tSpectrogram " \
                        "1\t1\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4f}\n"\
                            .format(
                                selection_id,
                                start,
                                end,
                                150,
                                15000,
                                config.CODES[
                                    c[0]
                                ] if c[0] in config.CODES else c[0],
                                label.split("_", 1)[-1],
                                c[1],
                            )

            # Write result string to file
            out_string += rstring

    elif config.RESULT_TYPE == "audacity":
        # Audacity timeline labels
        for timestamp in get_sorted_timestamps(r):
            rstring = ""

            for c in r[timestamp]:
                if c[1] > config.MIN_CONFIDENCE and (
                        not config.SPECIES_LIST or c[0] in config.SPECIES_LIST
                ):
                    label = config.TRANSLATED_LABELS[config.LABELS.index(c[0])]
                    rstring += \
                        "{}\t{}\t{:.4f}\n".format(
                            timestamp.replace("-", "\t"),
                            label.replace("_", ", "), c[1],
                        )

            # Write result string to file
            out_string += rstring

    elif config.RESULT_TYPE == "r":
        # Output format for R
        header = \
            "filepath," \
            "start," \
            "end," \
            "scientific_name," \
            "common_name," \
            "confidence," \
            "lat," \
            "lon," \
            "week," \
            "overlap," \
            "sensitivity," \
            "min_conf," \
            "species_list," \
            "model"
        out_string += header

        for timestamp in get_sorted_timestamps(r):
            rstring = ""
            start, end = timestamp.split("-", 1)

            for c in r[timestamp]:
                if c[1] > config.MIN_CONFIDENCE and (
                    not config.SPECIES_LIST or c[0] in config.SPECIES_LIST
                ):
                    label = config.TRANSLATED_LABELS[config.LABELS.index(c[0])]
                    rstring += \
                        "\n{},{},{},{},{},{:.4f},{:.4f},{:.4f}," \
                        "{},{},{},{},{},{}".format(
                            afile_path,
                            start,
                            end,
                            label.split("_", 1)[0],
                            label.split("_", 1)[-1],
                            c[1],
                            config.LATITUDE,
                            config.LONGITUDE,
                            config.WEEK,
                            config.SIG_OVERLAP,
                            (1.0 - config.SIGMOID_SENSITIVITY) + 1.0,
                            config.MIN_CONFIDENCE,
                            config.SPECIES_LIST_FILE,
                            os.path.basename(config.MODEL_PATH),
                        )

            # Write result string to file
            out_string += rstring

    elif config.RESULT_TYPE == "kaleidoscope":
        # Output format for kaleidoscope
        header = \
            "INDIR," \
            "FOLDER," \
            "IN FILE," \
            "OFFSET," \
            "DURATION," \
            "scientific_name," \
            "common_name," \
            "confidence," \
            "lat," \
            "lon," \
            "week,overlap," \
            "sensitivity"
        out_string += header

        folder_path, filename = os.path.split(afile_path)
        parent_folder, folder_name = os.path.split(folder_path)

        for timestamp in get_sorted_timestamps(r):
            rstring = ""
            start, end = timestamp.split("-", 1)

            for c in r[timestamp]:
                if c[1] > config.MIN_CONFIDENCE and (
                    not config.SPECIES_LIST or c[0] in config.SPECIES_LIST
                ):
                    label = config.TRANSLATED_LABELS[config.LABELS.index(c[0])]
                    rstring += \
                        "\n{},{},{},{},{},{},{},{:.4f},{:.4f},{:.4f}," \
                        "{},{},{}".format(
                            parent_folder.rstrip("/"),
                            folder_name,
                            filename,
                            start,
                            float(end) - float(start),
                            label.split("_", 1)[0],
                            label.split("_", 1)[-1],
                            c[1],
                            config.LATITUDE,
                            config.LONGITUDE,
                            config.WEEK,
                            config.SIG_OVERLAP,
                            (1.0 - config.SIGMOID_SENSITIVITY) + 1.0,
                        )

            # Write result string to file
            out_string += rstring

    else:
        # CSV output file
        header = "Start (s),End (s),Scientific name,Common name,Confidence\n"

        # Write header
        out_string += header

        for timestamp in get_sorted_timestamps(r):
            rstring = ""

            for c in r[timestamp]:
                start, end = timestamp.split("-", 1)

                if c[1] > config.MIN_CONFIDENCE and (
                    not config.SPECIES_LIST or c[0] in config.SPECIES_LIST
                ):
                    label = config.TRANSLATED_LABELS[config.LABELS.index(c[0])]
                    rstring += \
                        "{},{},{},{},{:.4f}\n".format(
                            start,
                            end,
                            label.split("_", 1)[0],
                            label.split("_", 1)[-1],
                            c[1],
                        )

            # Write result string to file
            out_string += rstring

    # Save as file
    with open(path, "w", encoding="utf-8") as rfile:
        rfile.write(out_string)
