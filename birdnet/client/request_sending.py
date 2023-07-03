import os
import json
from time import time

import requests


def send_request(host: str, port: int, fpath: str, mdata: str):
    """Sends a classification request to the server.

    Args:
        host: Host address of the server.
        port: Port for the request.
        fpath: File path of the file to be analyzed.
        mdata: Additional json metadata.

    Returns:
        The json decoded response.
    """
    url = f"http://{host}:{port}/analyze"

    print(f"Requesting analysis for {fpath}")

    # Make payload
    multipart_form_data = {
        "audio": (fpath.rsplit(os.sep, 1)[-1], open(fpath, "rb")),
        "meta": (None, mdata),
    }

    # Send request
    start_time = time()
    response = requests.post(url, files=multipart_form_data)
    end_time = time()

    print(
        "Response: {}, Time: {:.4f}s".format(
            response.text, end_time - start_time
        ),
        flush=True,
    )

    # Convert to dict
    data = json.loads(response.text)

    return data
