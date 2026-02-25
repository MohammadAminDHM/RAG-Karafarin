import json

from app.core.config import get_settings
from app.services.ingestion_service import build_phase1_pipeline


def main() -> None:
    settings = get_settings()
    _, report = build_phase1_pipeline(settings)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
