"""Allow 'python3 -m flux_index' to work."""
from flux_index.cli import main
import sys
sys.exit(main() or 0)
