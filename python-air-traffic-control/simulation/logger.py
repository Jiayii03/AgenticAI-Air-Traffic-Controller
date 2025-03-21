# simulation/logger.py
import sys
import os
import time

class Logger:
    """A logging utility that redirects print statements to both console and file."""
    
    def __init__(self, log_dir='logs', prefix='training_log'):
        """Initialize the logger with configurable directory and prefix.
        
        Args:
            log_dir (str): Directory to store log files
            prefix (str): Prefix for log filenames
        """
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a timestamped log filename
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_filename = f"{log_dir}/{prefix}_{self.timestamp}.txt"
        self.terminal = sys.stdout
        self.log_file = None
    
    def start(self):
        """Start logging by redirecting stdout to the logger."""
        self.log_file = open(self.log_filename, 'w')
        sys.stdout = self
        print(f"Logging started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {self.log_filename}")
        return self
    
    def stop(self):
        """Stop logging and restore stdout."""
        print(f"Logging stopped at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout = self.terminal
        if self.log_file:
            self.log_file.close()
    
    def write(self, message):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

    def get_filename(self):
        """Return the current log filename."""
        return self.log_filename