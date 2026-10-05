"""
BYBIT LIVE BOT - LOGGING SETUP
Daily-rotated file logs + colored console output.
"""
import os
import logging
import logging.handlers
from datetime import datetime
import config

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN if HAS_COLOR else "",
        "INFO": Fore.GREEN if HAS_COLOR else "",
        "WARNING": Fore.YELLOW if HAS_COLOR else "",
        "ERROR": Fore.RED if HAS_COLOR else "",
        "CRITICAL": Fore.MAGENTA + (Style.BRIGHT if HAS_COLOR else "") if HAS_COLOR else "",
    }
    RESET = Style.RESET_ALL if HAS_COLOR else ""

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        base = super().format(record)
        return f"{color}{base}{self.RESET}"

def setup_logging(level: str = "INFO"):
    os.makedirs(config.LOG_PATH, exist_ok=True)
    log_file = os.path.join(config.LOG_PATH, f"bot_{datetime.utcnow().strftime('%Y-%m-%d')}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Clear existing handlers
    for h in list(root_logger.handlers): root_logger.removeHandler(h)

    fmt = "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # File handler with daily rotation
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(fmt, date_fmt))
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter(fmt, date_fmt))
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    return root_logger
