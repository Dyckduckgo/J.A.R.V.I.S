"""PyInstaller entry point (the package itself is imported, not run as a script)."""

import sys

from jarvis.app import main

if __name__ == "__main__":
    sys.exit(main())
