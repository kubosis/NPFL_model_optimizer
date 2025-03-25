import sys
import logging
import ecs_logging


class EcsLogger:
    """
    EcsLogger is a custom logger class that integrates ECS (Elastic Common Schema) logging.

    Attributes:
        logger (logging.Logger): The logger instance.
        name (str): The name of the logger.
    """
    def __init__(self, name: str):
        self.logger = self.create_logger(name=name)
        self.name = name

    @staticmethod
    def create_logger(name: str):
        """
        Creates and configures a logger with ECS formatting.

        Args:
            name (str): The name of the logger.

        Returns:
            logging.Logger: The configured logger instance.
        """
        # Get the Logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Add an ECS formatter to the Handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ecs_logging.StdlibFormatter(exclude_fields=[
                # You can specify individual fields to ignore:
                "log.original",
            ]))
        logger.addHandler(handler)
        return logger
