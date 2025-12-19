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

def discover_force_majeure_matches(
    processed_dir: Path,
    threshold: float = 4.0
) -> List[Tuple[str, str, float]]:
    """
    Automatically discover all matches with excess_time_90 > threshold.
    
    Args:
        processed_dir: Directory containing processed match files
        threshold: Minimum excess_time_90 value to trigger removal (default: 4.0)
        
    Returns:
        List of tuples (season, match_id, excess_time_90)
    """
    matches_to_remove = []
    
    # Find all season directories
    season_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith('season_')])
    
    for season_dir in season_dirs:
        season = season_dir.name.replace('season_', '')
        match_files = sorted(season_dir.glob('match_*.json'))
        
        for match_file in match_files:
            try:
                with open(match_file, 'r', encoding='utf-8') as f:
                    match_data = json.load(f)
                
                excess_90 = match_data.get('targets', {}).get('excess_90')
                
                # Only process if excess_90 is available and > threshold
                if excess_90 is not None and excess_90 > threshold:
                    match_id = match_file.stem.replace('match_', '')
                    matches_to_remove.append((season, match_id, excess_90))
                    
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Error reading {match_file}: {e}")
                continue
    
    return matches_to_remove


def remove_force_majeure_matches(
    processed_dir: Path = Path("data/processed"),
    removed_dir: Path = Path("data/removed_from_processed"),
    results_dir: Path = Path("data/results"),
    threshold: float = 4.0
) -> Dict[str, any]:
    """
    Move force majeure matches (excess_time_90 > threshold minutes) from processed to removed.
    
    This function automatically discovers all matches with excess_time_90 > threshold
    and moves them to the removed directory. Matches with exactly threshold minutes
    are included in analysis (within plausible range for high-event matches).
    
    Args:
        processed_dir: Directory containing processed match files
        removed_dir: Directory to move removed matches to (organized by season)
        results_dir: Directory to save removal log
        threshold: Minimum excess_time_90 value to trigger removal (default: 4.0)
        
    Returns:
        Dictionary with removal statistics and details
    """
    # Create removed directory structure
    removed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Automatically discover matches to remove
    logger.info(f"Discovering matches with excess_time_90 > {threshold} minutes...")
    matches_to_remove = discover_force_majeure_matches(processed_dir, threshold)
    
    results = {
        "threshold": threshold,
        "total_matches": len(matches_to_remove),
        "successfully_removed": 0,
        "not_found": 0,
        "errors": 0,
        "removed_matches": [],
        "missing_matches": [],
        "error_matches": []
    }
    
    logger.info(f"Found {results['total_matches']} matches with excess_time_90 > {threshold} minutes")
    logger.info(f"Starting removal...")
    
    for season, match_id, excess_time in matches_to_remove:
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
                "expected_path": str(source_file)
            })
            continue
        
        try:
            # Create destination season directory if it doesn't exist
            season_dir_removed.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(source_file), str(dest_file))
            
            logger.info(f"Removed: {season} - {match_id} (Excess: {excess_time:.2f} min, > {threshold} threshold)")
            results["successfully_removed"] += 1
            results["removed_matches"].append({
                "season": season,
                "match_id": match_id,
                "excess_time_90": excess_time,
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
                "error": str(e)
            })
    
    # Save removal log
    log_file = results_dir / "removed_force_majeure_matches.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 70)
    logger.info("REMOVAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Threshold: excess_time_90 > {threshold} minutes")
    logger.info(f"Total matches found: {results['total_matches']}")
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

