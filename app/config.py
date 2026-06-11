from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

APP_NAME = os.getenv("APP_NAME", "Aegis")
APP_ENV = os.getenv("APP_ENV", "development")

CVE_SERVER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mcp_servers", "cve_server.py")
)

A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", "http://localhost:8888")
A2A_TIMEOUT = int(os.getenv("A2A_TIMEOUT", "30"))
