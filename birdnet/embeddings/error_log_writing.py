from birdnet.configuration import config


def write_error_log(msg):
    with open(config.ERROR_LOG_FILE, "a") as error_log:
        error_log.write(msg + "\n")
