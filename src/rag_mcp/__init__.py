import logging
import sys

# Configure logging for the package
logging.basicConfig(
    level=logging.INFO,  # Set the default logging level
    format="%(levelname)s %(asctime)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output logs to console
    ],
)
