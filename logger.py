import logging
import colorlog


class EmojiFormatter(colorlog.ColoredFormatter):
    def format(self, record):
        # Define emojis for each log level
        emoji_map = {
            "DEBUG": "🐛",
            "INFO": "💡",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🔥",
        }
        record.emoji = emoji_map.get(record.levelname, "❓")
        return super().format(record)


# Custom log colors: using bold and bright colors for extra flair
log_colors = {
    "DEBUG": "bold_blue",
    "INFO": "bold_green",
    "WARNING": "bold_yellow",
    "ERROR": "bold_red",
    "CRITICAL": "bold_purple",
}

# Define the log format with background color and emoji
log_format = "%(log_color)s%(asctime)s %(emoji)s %(levelname)s:%(reset)s " "%(message)s"

# Create the formatter using the EmojiFormatter class
formatter = EmojiFormatter(
    log_format, datefmt="%Y-%m-%d %H:%M:%S", log_colors=log_colors
)

# Setup the logger
logger = logging.getLogger("movie_rec_sys")
logger.setLevel(logging.DEBUG)  # Set to DEBUG for maximum detail
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
