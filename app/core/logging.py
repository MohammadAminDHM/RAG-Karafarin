import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure basic app logging.
    This is intentionally simple for Phase 0 and can be upgraded later.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # Reconfigure if already configured
    )
