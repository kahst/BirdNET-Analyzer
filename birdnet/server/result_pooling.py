def pool_results(lines: list[str], num_results=5, pmode="avg"):
    """Parses the results into list of (species, score).
    Args:
        lines: List of result scores.
        num_results: The number of entries to be returned.
        pmode: Decides how the score for each species is computed.
               If "max" used the maximum score for the species,
               if "avg" computes the average score per species.
    Returns:
        A List of (species, score).
    """
    # Parse results
    results = {}

    for line in lines:
        d = line.split("\t")
        species = d[2].replace(", ", "_")
        score = float(d[-1])

        if not species in results:
            results[species] = []

        results[species].append(score)

    # Compute score for each species
    for species in results:
        if pmode == "max":
            results[species] = max(results[species])
        else:
            results[species] = sum(results[species]) / len(results[species])

    # Sort results
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    return results[:num_results]
