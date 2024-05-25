import logging

# Define a new logging level for messages indicating no anomalies
NO_ANOMALIES = 25
logging.addLevelName(NO_ANOMALIES, "NO_ANOMALIES")

# Define the custom formatter with colors
class CustomFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages."""
    grey = "\x1b[38;21m"
    blue = "\x1b[38;21;94m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    green= "\x1b[32m"

    # Define log message formats with colors
    FORMATS = {
        NO_ANOMALIES: green + "NO_ANOMALIES" + reset + " %(asctime)s - %(message)s",
        logging.DEBUG: blue + "DEBUG" + reset + " %(asctime)s - %(message)s",
        logging.INFO: grey + "INFO" + reset + " %(asctime)s - %(message)s",
        logging.WARNING: yellow + "WARNING" + reset + " %(asctime)s - %(message)s",
        logging.ERROR: red + "ERROR" + reset + " %(asctime)s - %(message)s",
        logging.CRITICAL: bold_red + "CRITICAL" + reset + " %(asctime)s - %(message)s"
    }

    def format(self, record):
        log_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def setup_logging():
    """
    Set up logging configuration.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler('network_anomaly_detection.log')
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create custom formatter and set for handlers
    formatter = CustomFormatter()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_activity(activity, level=logging.INFO):
    """
    Log activity related to network anomaly detection.

    Args:
    - activity: String describing the activity to be logged.
    - level: Logging level (default: INFO).
    """
    logging.log(level, activity)