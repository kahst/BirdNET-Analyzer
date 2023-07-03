def save_as_embeddings_file(results: dict[str, str], fpath: str):
    """Write embeddings to file

    Args:
        results: A dictionary containing the embeddings at timestamp.
        fpath: The path for the embeddings file.
    """
    with open(fpath, "w") as f:
        for timestamp in results:
            f.write(
                timestamp.replace("-", "\t") +
                "\t" +
                ",".join(map(str, results[timestamp])) +
                "\n"
            )
