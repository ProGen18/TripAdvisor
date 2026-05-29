"""Orchestrateur du pipeline complet TripAdvisor."""
import sys
import time
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.config import PROCESSED_DIR, ENRICHED_DIR, OUTPUT_DIR

STEPS = [
    ("build", "scripts/build_restaurants.py",
     PROCESSED_DIR / "restaurants.parquet"),
    ("classify", "scripts/classify_menu.py",
     PROCESSED_DIR / "menu_diversity.parquet"),
    ("enrich", "scripts/enrich_external.py",
     ENRICHED_DIR / "restaurants_enriched.parquet"),
    ("stats", "scripts/descriptive_stats.py",
     OUTPUT_DIR / "rating_distribution.png"),
]


def main():
    args = sys.argv[1:]
    from_step = None
    if "--from" in args:
        idx = args.index("--from")
        from_step = args[idx + 1] if idx + 1 < len(args) else None

    start = 0
    if from_step:
        for i, (name, _, _) in enumerate(STEPS):
            if name == from_step:
                start = i
                break
        else:
            print(f"Etape inconnue : {from_step}")
            print(f"Disponibles : {[s[0] for s in STEPS]}")
            sys.exit(1)

    for i, (name, script, output) in enumerate(STEPS):
        if i < start:
            print(f"[IGNORE] {name} (--from {from_step})")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(STEPS)}] {name}")
        print(f"{'='*60}")

        if output.exists():
            print(f"  Sortie existante : {output}")
            print(f"  [IGNORE] (supprimer pour re-executer)")
            continue

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, script],
            cwd=Path(__file__).resolve().parents[1],
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  [ECHEC] apres {elapsed:.0f}s")
            sys.exit(1)

        print(f"  [OK] {elapsed:.0f}s")


if __name__ == "__main__":
    main()
