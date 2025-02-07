import argparse
import os

import birdnet_analyzer.config as cfg


ASCII_LOGO = r"""                        
                          .                                     
                       .-=-                                     
                    .:=++++.                                    
                 ..-======#=:.                                  
                .-%%%#*+=-#+++:..                               
              .-+***======++++++=..                             
                  .=====+==++++++++-.                           
                  .=+++====++++++++++=:.                        
                  .++++++++=======----===:                      
                   =+++++++====-----+++++++-.                   
                   .=++++==========-=++=====+=:.                
                     -++======---:::::-=++++***+:.              
                     ..---::::::::::::::::-=*****+-.            
                       ..--------::::::::::::--+##-.:.          
  ++++=::::::...         ..-------------::::::-::.::.           
           ..::-------:::.-=.:::::+-....   ....:--:..           
                    ..::-======--+::......      .:---:.         
                              ..:--==+++++==-..    .-+==-       
                                   ......::----:      **=--     
                                            ..-=-:.     *+=:=   
                                              ..-====  +++ =+** 
                                                 ========+      
                                                 **=====        
                                               ***+==           
                                              ****+             
"""


def io_args():
    """
    Creates an argument parser for input and output paths.
    Returns:
        argparse.ArgumentParser: The argument parser with input and output path arguments.
    Arguments:
        input (str): Path to the input file or folder. Defaults to the value specified in cfg.INPUT_PATH.
        output (str): Path to the output folder. Defaults to the input path if not specified.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "input",
        metavar="INPUT",
        default=cfg.INPUT_PATH,
        help="Path to input file or folder.",
        nargs="?",
    )
    p.add_argument("-o", "--output", help="Path to output folder. Defaults to the input path.")

    return p


def bandpass_args():
    """
    Creates an argument parser for bandpass filter frequency arguments.
    This function sets up an argument parser with two arguments:
    --fmin and --fmax, which define the minimum and maximum frequencies
    for the bandpass filter, respectively. The values are constrained
    to be within the range defined by cfg.SIG_FMIN and cfg.SIG_FMAX.
    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--fmin",
        type=lambda a: max(0, min(cfg.SIG_FMAX, int(a))),
        default=cfg.SIG_FMIN,
        help=f"Minimum frequency for bandpass filter in Hz. Defaults to {cfg.SIG_FMIN} Hz.",
    )
    p.add_argument(
        "--fmax",
        type=lambda a: max(cfg.SIG_FMIN, min(cfg.SIG_FMAX, int(a))),
        default=cfg.SIG_FMAX,
        help=f"Maximum frequency for bandpass filter in Hz. Defaults to {cfg.SIG_FMAX} Hz.",
    )

    return p


def species_args():
    """
    Creates an argument parser for species-related arguments.
    Returns:
        argparse.ArgumentParser: The argument parser with the following arguments:
            --lat (float): Recording location latitude. Set -1 to ignore. Default is -1.
            --lon (float): Recording location longitude. Set -1 to ignore. Default is -1.
            --week (int): Week of the year when the recording was made. Values in [1, 48] (4 weeks per month).
                          Set -1 for year-round species list. Default is -1.
            --slist (str): Path to species list file or folder. If folder is provided, species list needs to be named
                           "species_list.txt". If lat and lon are provided, this list will be ignored.
            --sf_thresh (float): Minimum species occurrence frequency threshold for location filter. Values in [0.01, 0.99].
                                 Defaults to cfg.LOCATION_FILTER_THRESHOLD.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--lat", type=float, default=-1, help="Recording location latitude. Set -1 to ignore.")
    p.add_argument("--lon", type=float, default=-1, help="Recording location longitude. Set -1 to ignore.")
    p.add_argument(
        "--week",
        type=int,
        default=-1,
        help="Week of the year when the recording was made. Values in [1, 48] (4 weeks per month). Set -1 for year-round species list.",
    )
    p.add_argument(
        "--slist",
        help='Path to species list file or folder. If folder is provided, species list needs to be named "species_list.txt". If lat and lon are provided, this list will be ignored.',
    )
    p.add_argument(
        "--sf_thresh",
        type=lambda a: max(0.01, min(0.99, float(a))),
        default=cfg.LOCATION_FILTER_THRESHOLD,
        help=f"Minimum species occurrence frequency threshold for location filter. Values in [0.01, 0.99]. Defaults to {cfg.LOCATION_FILTER_THRESHOLD}.",
    )

    return p


def sigmoid_args():
    """
    Creates an argument parser for sigmoid sensitivity.
    This function sets up an argument parser with a single argument `--sensitivity`.
    The sensitivity value is constrained to be within the range [0.5, 1.5], where higher
    values result in higher detection sensitivity. The default value is taken from
    `cfg.SIGMOID_SENSITIVITY`.
    Returns:
        argparse.ArgumentParser: The argument parser with the sensitivity argument configured.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--sensitivity",
        type=lambda a: max(0.5, min(1.0 - (float(a) - 1.0), 1.5)),
        default=cfg.SIGMOID_SENSITIVITY,
        help=f"Detection sensitivity; Higher values result in higher sensitivity. Values in [0.5, 1.5]. Defaults to {cfg.SIGMOID_SENSITIVITY}.",
    )

    return p


def overlap_args(help_string=f"Overlap of prediction segments. Values in [0.0, 2.9]. Defaults to {cfg.SIG_OVERLAP}."):
    """
    Creates an argument parser for the overlap of prediction segments.
    Args:
        help_string (str): A custom help string for the overlap argument. Defaults to a formatted string
                           indicating the range [0.0, 2.9] and the default value from cfg.SIG_OVERLAP.
    Returns:
        argparse.ArgumentParser: An argument parser with the overlap argument configured.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--overlap",
        type=lambda a: max(0.0, min(2.9, float(a))),
        default=cfg.SIG_OVERLAP,
        help=help_string,
    )

    return p


def audio_speed_args():
    """
    Creates an argument parser for audio speed configuration.
    This function sets up an argument parser with a single argument `--audio_speed`
    which allows the user to specify a speed factor for audio playback. The speed factor
    must be a float value where values less than 1.0 will slow down the audio and values
    greater than 1.0 will speed it up. The minimum allowed value is 0.01. The default
    value is taken from the configuration (`cfg.AUDIO_SPEED`).
    Returns:
        argparse.ArgumentParser: The argument parser with the `--audio_speed` argument configured.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--audio_speed",
        type=lambda a: max(0.01, float(a)),
        default=cfg.AUDIO_SPEED,
        help=f"Speed factor for audio playback. Values < 1.0 will slow down the audio, values > 1.0 will speed it up. Defaults to {cfg.AUDIO_SPEED}.",
    )

    return p


def threads_args():
    """
    Creates an argument parser for specifying the number of CPU threads to use.
    The parser adds an argument `--threads` (or `-t`) which accepts an integer value.
    The value is constrained to be at least 1. If not specified, the default value is
    set to half the number of available CPU cores, but not exceeding 8.
    Returns:
        argparse.ArgumentParser: The argument parser with the `--threads` argument.
    """
    import multiprocessing

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "-t",
        "--threads",
        type=lambda a: max(1, int(a)),
        default=min(8, max(1, multiprocessing.cpu_count() // 2)),
        help="Number of CPU threads.",
    )

    return p


def min_conf_args():
    """
    Creates an argument parser for the minimum confidence threshold.

    Returns:
        argparse.ArgumentParser: An argument parser with the --min_conf argument.

    The --min_conf argument:
        - Sets the minimum confidence threshold for predictions.
        - Accepts float values in the range [0.01, 0.99].
        - Defaults to the value specified in cfg.MIN_CONFIDENCE.
        - Ensures that the provided value is clamped between 0.01 and 0.99.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--min_conf",
        default=cfg.MIN_CONFIDENCE,
        type=lambda a: max(0.01, min(0.99, float(a))),
        help=f"Minimum confidence threshold. Values in [0.01, 0.99]. Defaults to {cfg.MIN_CONFIDENCE}.",
    )

    return p


def locale_args():
    """
    Creates an argument parser for locale settings.
    This function creates an argument parser with a single argument `--locale`
    (or `-l`) which specifies the locale for translated species common names.
    The default value is 'en' (US English). The available locale values include
    'af', 'en_UK', 'de', 'it', and others.
    Returns:
        argparse.ArgumentParser: An argument parser with the locale argument.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "-l",
        "--locale",
        default="en",
        help="Locale for translated species common names. Values in ['af', 'en_UK', 'de', 'it', ...] Defaults to 'en' (US English).",
    )

    return p


def bs_args():
    """
    Creates an argument parser for batch size configuration.
    Returns:
        argparse.ArgumentParser: An argument parser with a batch size argument.
    The parser includes the following argument:
        -b, --batchsize: An integer specifying the number of samples to process at the same time.
                         The value must be at least 1. Defaults to the value of cfg.BATCH_SIZE.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "-b",
        "--batchsize",
        type=lambda a: max(1, int(a)),
        default=cfg.BATCH_SIZE,
        help=f"Number of samples to process at the same time. Defaults to {cfg.BATCH_SIZE}.",
    )

    return p


def analyzer_parser():
    parents = [
        io_args(),
        bandpass_args(),
        species_args(),
        sigmoid_args(),
        overlap_args(),
        audio_speed_args(),
        threads_args(),
        min_conf_args(),
        locale_args(),
        bs_args(),
    ]

    if os.environ.get("IS_GITHUB_RUNNER", "false").lower() == "true":
        parser = argparse.ArgumentParser(description="analyze stuff", parents=parents)
    else:
        parser = argparse.ArgumentParser(
            description=ASCII_LOGO,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            usage="python -m birdnet_analyzer.analyze [options]",
            parents=parents,
        )

    class UniqueSetAction(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            setattr(args, self.dest, {v.lower() for v in values})

    parser.add_argument(
        "--rtype",
        default={"table"},
        choices=["table", "audacity", "kaleidoscope", "csv"],
        nargs="+",
        help="Specifies output format. Values in ['table', 'audacity',  'kaleidoscope', 'csv']. Defaults to 'table' (Raven selection table).",
        action=UniqueSetAction,
    )
    parser.add_argument(
        "--combine_results",
        help="Also outputs a combined file for all the selected result types. If not set combined tables will be generated. Defaults to False.",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "-c",
        "--classifier",
        help="Path to custom trained classifier. Defaults to None. If set, --lat, --lon and --locale are ignored.",
    )

    parser.add_argument(
        "--skip_existing_results",
        action="store_true",
        help="Skip files that have already been analyzed. Defaults to False.",
    )

    return parser
