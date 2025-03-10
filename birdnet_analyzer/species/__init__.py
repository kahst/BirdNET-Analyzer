def main():
    import birdnet_analyzer.cli as cli
    import birdnet_analyzer.utils as utils

    # Parse arguments
    parser = cli.species_parser()

    args = parser.parse_args()

    utils.ensure_model_exists()

    from birdnet_analyzer.species.utils import run # noqa: E402

    run(args.output, args.lat, args.lon, args.week, args.sf_thresh, args.sortby)