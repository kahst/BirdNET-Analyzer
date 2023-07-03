def get_sorted_timestamps(results: dict[str, list]):
    """Sorts the results based on the segments.
    Args:
        results: The dictionary with {segment: scores}.
    Returns:
        Returns the sorted list of segments and their scores.
    """
    return sorted(results, key=lambda t: float(t.split("-", 1)[0]))
