import sys

from .cli import main

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):          # Windows: консоль може бути не UTF-8
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
