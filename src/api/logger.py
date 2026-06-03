# logger.py
# Central logging configuration for the fraud detection platform.
# Import get_logger() in any module to get a JSON-formatted logger.

# Why JSON logging?
# Production systems use log aggregators like Datadog, CloudWatch, or the ELK Stack (Elasticsearch, Logstash, Kibana) to collect and search logs from many services at once. These tools work by ingesting structured JSON where each log line is a parseable object with known fields.
# Plain print() output is unstructured text that cannot be queried, filtered, or alerted on programmatically.

import logging
import sys
from pythonjsonlogger.json import JsonFormatter


def get_logger(name: str, service: str = "fastapi") -> logging.Logger:
    """
    Creates and returns a JSON-formatted logger.

    name: the module name, pass __name__ from the calling module.
          This becomes the "name" field in the log output and helps identify which module produced the log line.

    service: a label identifying which service this log came from.
             Useful when multiple services write to the same log aggregator.

    How it works:
    1. Create a standard Python Logger object.
    2. Attach a StreamHandler that writes to stdout.
    3. Set a JsonFormatter from python-json-logger that serialises every log record as a single JSON line.
    4. The "extra" field lets us attach arbitrary key-value pairs to every log line produced by this logger, service name in this case.
    """
    logger = logging.getLogger(name)

    # To avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # The format string defines which standard LogRecord fields appear.
    # Fields not listed here but added via extra={} still appear in the output.
    formatter = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "module"
        },
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent log records from propagating to the root logger, which would cause duplicate output
    logger.propagate = False

    # Attach the service name to every log line from this logger
    logger = logging.LoggerAdapter(logger, {"service": service})

    return logger