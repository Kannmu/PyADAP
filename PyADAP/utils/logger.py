"""Logging utilities for PyADAP 3.0

This module provides a comprehensive logging system with multiple output formats
and configurable log levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
from colorama import Fore, Back, Style, init

# Initialize colorama for Windows compatibility
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT
    }
    
    def format(self, record):
        # Get the original formatted message
        formatted = super().format(record)
        
        # Add color based on log level
        color = self.COLORS.get(record.levelname, '')
        if color:
            # Color the entire message
            formatted = f"{color}{formatted}{Style.RESET_ALL}"
        
        return formatted


class Logger:
    """Advanced logging system for PyADAP."""
    
    def __init__(self, 
                 name: str = "PyADAP",
                 level: Union[str, int] = logging.INFO,
                 log_file: Optional[Union[str, Path]] = None,
                 console_output: bool = True,
                 colored_output: bool = True):
        """Initialize the logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            console_output: Whether to output to console
            colored_output: Whether to use colored console output
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Set logging level
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Setup console handler
        if console_output:
            self._setup_console_handler(colored_output)
        
        # Setup file handler
        if log_file:
            self._setup_file_handler(log_file)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def _setup_console_handler(self, colored: bool = True) -> None:
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if colored:
            formatter = ColoredFormatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: Union[str, Path]) -> None:
        """Setup file logging handler."""
        log_path = Path(log_file)
        
        # Create directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, *args, **kwargs)
    
    def log_function_call(self, func_name: str, args: tuple = (), kwargs: dict = None) -> None:
        """Log function call with parameters."""
        kwargs = kwargs or {}
        args_str = ', '.join(str(arg) for arg in args)
        kwargs_str = ', '.join(f"{k}={v}" for k, v in kwargs.items())
        
        params = ', '.join(filter(None, [args_str, kwargs_str]))
        self.debug(f"Calling {func_name}({params})")
    
    def log_execution_time(self, func_name: str, execution_time: float) -> None:
        """Log function execution time."""
        self.info(f"{func_name} executed in {execution_time:.4f} seconds")
    
    def log_data_info(self, data_name: str, shape: tuple, dtypes: dict = None) -> None:
        """Log data information."""
        self.info(f"{data_name}: shape={shape}")
        if dtypes:
            dtype_summary = ', '.join(f"{k}: {v}" for k, v in dtypes.items())
            self.debug(f"{data_name} dtypes: {dtype_summary}")
    
    def log_statistical_result(self, test_name: str, statistic: float, p_value: float, 
                              effect_size: Optional[float] = None) -> None:
        """Log statistical test result."""
        result_msg = f"{test_name}: statistic={statistic:.4f}, p={p_value:.4f}"
        if effect_size is not None:
            result_msg += f", effect_size={effect_size:.4f}"
        self.info(result_msg)
    
    def log_progress(self, current: int, total: int, task_name: str = "Processing") -> None:
        """Log progress information."""
        percentage = (current / total) * 100 if total > 0 else 0
        self.info(f"{task_name}: {current}/{total} ({percentage:.1f}%)")
    
    def log_memory_usage(self, memory_mb: float, context: str = "") -> None:
        """Log memory usage information."""
        context_str = f" ({context})" if context else ""
        self.debug(f"Memory usage{context_str}: {memory_mb:.2f} MB")
    
    def log_file_operation(self, operation: str, file_path: Union[str, Path], 
                          success: bool = True, error_msg: str = "") -> None:
        """Log file operation."""
        if success:
            self.info(f"{operation}: {file_path}")
        else:
            self.error(f"Failed to {operation.lower()}: {file_path} - {error_msg}")
    
    def log_configuration(self, config_dict: dict, config_name: str = "Configuration") -> None:
        """Log configuration settings."""
        self.info(f"{config_name} loaded:")
        for key, value in config_dict.items():
            self.debug(f"  {key}: {value}")
    
    def create_section_separator(self, title: str, char: str = "=", width: int = 80) -> None:
        """Create a visual section separator in logs."""
        separator = char * width
        title_line = f" {title} ".center(width, char)
        
        self.info(separator)
        self.info(title_line)
        self.info(separator)
    
    def set_level(self, level: Union[str, int]) -> None:
        """Change logging level."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)
    
    def add_file_handler(self, log_file: Union[str, Path]) -> None:
        """Add additional file handler."""
        self._setup_file_handler(log_file)
    
    def remove_handlers(self) -> None:
        """Remove all handlers."""
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()


def get_logger(name: str = "PyADAP", **kwargs) -> Logger:
    """Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for Logger initialization
        
    Returns:
        Logger instance
    """
    return Logger(name=name, **kwargs)


def setup_logging(log_file: Optional[Union[str, Path]] = None,
                 level: Union[str, int] = logging.INFO,
                 console_output: bool = True,
                 colored_output: bool = True) -> Logger:
    """Setup default logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        console_output: Whether to output to console
        colored_output: Whether to use colored console output
        
    Returns:
        Configured Logger instance
    """
    # Create default log file if none specified
    if log_file is None and console_output is False:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"PyADAP_{timestamp}.log"
    
    return Logger(
        name="PyADAP",
        level=level,
        log_file=log_file,
        console_output=console_output,
        colored_output=colored_output
    )


class LogContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(self, logger: Logger, level: Union[str, int]):
        """Initialize log context.
        
        Args:
            logger: Logger instance
            level: Temporary logging level
        """
        self.logger = logger
        self.new_level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.original_level = logger.logger.level
    
    def __enter__(self):
        self.logger.set_level(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.set_level(self.original_level)


def with_logging_level(logger: Logger, level: Union[str, int]):
    """Decorator to temporarily change logging level.
    
    Args:
        logger: Logger instance
        level: Temporary logging level
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LogContext(logger, level):
                return func(*args, **kwargs)
        return wrapper
    return decorator