"""Run paper experiments and summarize freshly computed results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from gblcss.cli import main

if __name__ == '__main__':
    main()
