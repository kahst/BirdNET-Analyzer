import os

def list_subdirectories(path: str):
    return filter(lambda el: os.path.isdir(os.path.join(path, el)), os.listdir(path))