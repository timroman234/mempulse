"""Force offline mode for every test: no API calls even if .env holds a key."""

import os

os.environ["MEMPULSE_OFFLINE"] = "1"
