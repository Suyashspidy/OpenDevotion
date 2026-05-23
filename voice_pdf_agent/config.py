from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Config:
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

config = Config()