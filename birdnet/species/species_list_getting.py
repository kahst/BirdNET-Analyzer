from birdnet.model.main import explore


def get_species_list(
    lat: float,
    lon: float,
    week: int,
    threshold=0.05,
    sort=False,
) -> list[str]:
    """Predict a species list.

    Uses the model to predict the species list for the given coordinates and
    filters by threshold.

    Args:
        lat: The latitude.
        lon: The longitude.
        week: The week of the year [1-48]. Use -1 for year-round.
        threshold: Only values above or equal to threshold will be shown.
        sort: If the species list should be sorted.

    Returns:
        A list of all eligible species.
    """
    # Extract species from model
    pred = explore(lat, lon, week)

    # Make species list
    slist = [p[1] for p in pred if p[0] >= threshold]

    return sorted(slist) if sort else slist
