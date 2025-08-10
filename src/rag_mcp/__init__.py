import logging

# Configure logging for the package
logging.basicConfig(
    level=logging.INFO,  # Set the default logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Output logs to console
    ],
)
