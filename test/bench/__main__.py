"""`python -m bench …` 等价于 `python -m bench.cli …`。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
