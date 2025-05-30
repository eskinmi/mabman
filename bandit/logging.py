import logging
import sys
from typing import Optional

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> None:
    """Setup logging configuration for the bandit library.
    
    Args:
        level: Logging level
        log_file: Optional file path to write logs
        format_string: Log message format
    """
    root_logger = logging.getLogger('bandit')
    root_logger.setLevel(level)
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)