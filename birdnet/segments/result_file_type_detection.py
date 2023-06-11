def detect_result_file_type(line: str):
    """Detects the type of result file.
    Args:
        line: First line of text.
    Returns:
        Either "table", "r", "kaleidoscope", "csv" or "audacity".
    """
    if line.lower().startswith("selection"):
        return "table"
    elif line.lower().startswith("filepath"):
        return "r"
    elif line.lower().startswith("indir"):
        return "kaleidoscope"
    elif line.lower().startswith("start (s)"):
        return "csv"
    else:
        return "audacity"
