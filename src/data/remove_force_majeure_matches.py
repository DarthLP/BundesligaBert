"""
Remove Force Majeure Matches: Exclude matches with excess_time_90 > 4 minutes

This script moves matches with excess_time_90 > 4 minutes from data/processed/ to 
data/removed_from_processed/. These matches represent structural breaks (riots, VAR 
confusion, technical failures, severe injuries) rather than normal referee bias or 
time-wasting, and must be excluded to preserve statistical validity in both regression 
and BERT models.

Rationale:
- OLS regression is highly sensitive to outliers. A single match with 16 minutes of 
  excess time has 1000x more influence than a normal match (squared error 256 vs 0.25).
- BERT models cannot generalize from rare, out-of-distribution events (pitch invasions, 
  police interventions). Including them causes memorization rather than learning.
- Normal referee bias operates in the -1 to +2 minute range. Excess > 4 minutes indicates 
  external interruptions not captured by model variables (goals, subs, cards).
- 4.0 minutes of excess time is still within plausible range for high-event matches or 
  VAR reviews, so only matches > 4.0 are excluded.

Usage:
    python src/data/remove_force_majeure_matches.py

Output:
    Moves files from data/processed/season_{season}/match_{match_id}.json
    to data/removed_from_processed/season_{season}/match_{match_id}.json
    
    Creates a log file at data/results/removed_force_majeure_matches.json with details.

Author: BundesligaBERT Project
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Matches to remove (excess_time_90 > 4.0 minutes)
# Format: (season, match_id, excess_time_90, description)
# Note: Matches with exactly 4.0 minutes are included in analysis (within plausible range)
FORCE_MAJEURE_MATCHES: List[Tuple[str, str, float, str]] = [
    ("2017-18", "hsv-gegen-mgladbach-2018-bundesliga-3827891", 16.00, "Hamburger SV vs Bor. Mönchengladbach (HSV relegation day - pitch invasion)"),
    ("2018-19", "dortmund-gegen-bayern-2018-bundesliga-4243356", 5.00, "Borussia Dortmund vs Bayern München"),
    ("2018-19", "hannover-gegen-dortmund-2018-bundesliga-4243281", 5.00, "Hannover 96 vs Borussia Dortmund"),
    ("2021-22", "dortmund-gegen-hoffenheim-2021-bundesliga-4721514", 6.00, "Borussia Dortmund vs TSG Hoffenheim"),
    ("2022-23", "bayern-gegen-hoffenheim-2023-bundesliga-4781675", 6.00, "Bayern München vs TSG Hoffenheim"),
    ("2022-23", "bochum-gegen-dortmund-2023-bundesliga-4781699", 6.00, "VfL Bochum vs Borussia Dortmund"),
    ("2022-23", "bochum-gegen-freiburg-2023-bundesliga-4781618", 5.00, "VfL Bochum vs SC Freiburg"),
    ("2023-24", "bochum-gegen-mgladbach-2023-bundesliga-4862017", 8.00, "VfL Bochum vs Bor. Mönchengladbach"),
    ("2024-25", "bochum-gegen-wolfsburg-2024-bundesliga-4936837", 5.00, "VfL Bochum vs VfL Wolfsburg"),
    ("2024-25", "dortmund-gegen-heidenheim-2024-bundesliga-4936804", 5.00, "Borussia Dortmund vs 1. FC Heidenheim"),
]


def remove_force_majeure_matches(
    processed_dir: Path = Path("data/processed"),
    removed_dir: Path = Path("data/removed_from_processed"),
    results_dir: Path = Path("data/results")
) -> Dict[str, any]:
    """
    Move force majeure matches (excess_time_90 > 4 minutes) from processed to removed.
    
    Args:
        processed_dir: Directory containing processed match files
        removed_dir: Directory to move removed matches to (organized by season)
        results_dir: Directory to save removal log
        
    Returns:
        Dictionary with removal statistics and details
    """
    # Create removed directory structure
    removed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "total_matches": len(FORCE_MAJEURE_MATCHES),
        "successfully_removed": 0,
        "not_found": 0,
        "errors": 0,
        "removed_matches": [],
        "missing_matches": [],
        "error_matches": []
    }
    
    logger.info(f"Starting removal of {results['total_matches']} force majeure matches...")
    
    for season, match_id, excess_time, description in FORCE_MAJEURE_MATCHES:
        # Construct source and destination paths
        season_dir_processed = processed_dir / f"season_{season}"
        season_dir_removed = removed_dir / f"season_{season}"
        
        source_file = season_dir_processed / f"match_{match_id}.json"
        dest_file = season_dir_removed / f"match_{match_id}.json"
        
        # Check if source file exists
        if not source_file.exists():
            logger.warning(f"Match not found: {source_file}")
            results["not_found"] += 1
            results["missing_matches"].append({
                "season": season,
                "match_id": match_id,
                "excess_time_90": excess_time,
                "description": description,
                "expected_path": str(source_file)
            })
            continue
        
        try:
            # Create destination season directory if it doesn't exist
            season_dir_removed.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(source_file), str(dest_file))
            
            logger.info(f"Removed: {season} - {match_id} (Excess: {excess_time} min, > 4.0 threshold)")
            results["successfully_removed"] += 1
            results["removed_matches"].append({
                "season": season,
                "match_id": match_id,
                "excess_time_90": excess_time,
                "description": description,
                "source_path": str(source_file),
                "destination_path": str(dest_file)
            })
            
        except Exception as e:
            logger.error(f"Error removing {source_file}: {e}")
            results["errors"] += 1
            results["error_matches"].append({
                "season": season,
                "match_id": match_id,
                "excess_time_90": excess_time,
                "description": description,
                "error": str(e)
            })
    
    # Save removal log
    log_file = results_dir / "removed_force_majeure_matches.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 70)
    logger.info("REMOVAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total matches to remove: {results['total_matches']}")
    logger.info(f"Successfully removed: {results['successfully_removed']}")
    logger.info(f"Not found: {results['not_found']}")
    logger.info(f"Errors: {results['errors']}")
    logger.info(f"Removal log saved to: {log_file}")
    
    return results


if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # Run removal
    results = remove_force_majeure_matches()
    
    # Exit with error code if any failures occurred
    if results["not_found"] > 0 or results["errors"] > 0:
        exit(1)

