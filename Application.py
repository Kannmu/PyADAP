"""PyADAP 3.0 Application
========================

PyADAP: Python Automated Data Analysis Pipeline - Version 3.0

Main application entry point with modern GUI interface and enhanced
statistical analysis capabilities.

Key Features in 3.0:
- Modern GUI with intuitive workflow
- Enhanced statistical analysis with automatic assumption checking
- Intelligent test selection based on data characteristics
- Advanced data preprocessing and outlier detection
- Comprehensive effect size calculations
- Automated report generation with statistical interpretations
- Robust error handling and validation

Author: Kannmu
Date: 2024/12/19
License: MIT License
Repository: https://github.com/Kannmu/PyADAP
"""

import sys
import os
import warnings
from pathlib import Path
from traceback import print_exc

# Add PyADAP to path if needed
sys.path.insert(0, str(Path(__file__).parent))

# Import PyADAP 3.0 modules
try:
    from PyADAP import __version__
    from PyADAP.gui import MainWindow
    from PyADAP.config import Config
    from PyADAP.utils import get_logger, setup_logging
except ImportError as e:
    print_exc()
    print(f"Error importing PyADAP modules: {e}")
    print("Please ensure PyADAP 3.0 is properly installed.")
    sys.exit(1)

# Configure warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    """Main application entry point."""
    try:
        # Setup logging
        setup_logging()
        logger = get_logger("PyADAP.Application")
        
        # Print welcome message
        print(f"\n{'='*60}")
        print(f"  PyADAP {__version__} - Advanced Data Analysis Platform")
        print(f"{'='*60}")
        print("Loading application, please wait...\n")
        
        logger.info(f"Starting PyADAP {__version__}")
        
        # Load configuration
        config = Config()
        logger.info("Configuration loaded successfully")
        
        # Create and run main window
        app = MainWindow(config=config)
        logger.info("Main window initialized")
        
        print("Application ready! Opening main window...")
        
        # Start the GUI application
        app.run()
        
        logger.info("Application closed normally")
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("Please check the log files for more details.")
        
        # Try to log the error if possible
        try:
            logger = get_logger("PyADAP.Application")
            logger.error(f"Fatal application error: {str(e)}", exc_info=True)
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()
