from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Config:
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")

config = Config()