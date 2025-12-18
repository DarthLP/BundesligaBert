"""
Restore 4.0 Minute Matches: Move matches with excess_time_90 == 4.0 back to processed

This script restores matches with exactly 4.0 minutes of excess time from 
data/removed_from_processed/ back to data/processed/. These matches are now
considered valid for analysis after updating the threshold from 3.0 to 4.0 minutes.

Usage:
    python src/data/restore_4min_matches.py

Output:
    Moves files from data/removed_from_processed/season_{season}/match_{match_id}.json
    back to data/processed/season_{season}/match_{match_id}.json
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def restore_4min_matches(
    removed_dir: Path = Path("data/removed_from_processed"),
    processed_dir: Path = Path("data/processed"),
    removal_log: Path = Path("data/results/removed_force_majeure_matches.json")
) -> Dict[str, any]:
    """
    Restore matches with excess_time_90 == 4.0 from removed back to processed.
    
    Args:
        removed_dir: Directory containing removed match files
        processed_dir: Directory to restore matches to
        removal_log: Path to removal log JSON file
        
    Returns:
        Dictionary with restoration statistics and details
    """
    if not removal_log.exists():
        logger.error(f"Removal log not found: {removal_log}")
        return {"error": "Removal log not found"}
    
    # Load removal log to find 4.0 minute matches
    with open(removal_log, 'r', encoding='utf-8') as f:
        removal_data = json.load(f)
    
    results = {
        "total_4min_matches": 0,
        "successfully_restored": 0,
        "not_found": 0,
        "errors": 0,
        "restored_matches": [],
        "missing_matches": [],
        "error_matches": []
    }
    
    # Find all matches with excess_time_90 == 4.0
    matches_to_restore = [
        match for match in removal_data.get('removed_matches', [])
        if match.get('excess_time_90') == 4.0
    ]
    
    results["total_4min_matches"] = len(matches_to_restore)
    logger.info(f"Found {results['total_4min_matches']} matches with excess_time_90 == 4.0 to restore")
    
    for match_info in matches_to_restore:
        season = match_info['season']
        match_id = match_info['match_id']
        excess_time = match_info['excess_time_90']
        
        # Construct source and destination paths
        season_dir_removed = removed_dir / f"season_{season}"
        season_dir_processed = processed_dir / f"season_{season}"
        
        source_file = season_dir_removed / f"match_{match_id}.json"
        dest_file = season_dir_processed / f"match_{match_id}.json"
        
        # Check if source file exists
        if not source_file.exists():
            logger.warning(f"Match not found in removed: {source_file}")
            results["not_found"] += 1
            results["missing_matches"].append({
                "season": season,
                "match_id": match_id,
                "expected_path": str(source_file)
            })
            continue
        
        try:
            # Create destination season directory if it doesn't exist
            season_dir_processed.mkdir(parents=True, exist_ok=True)
            
            # Check if destination already exists (shouldn't happen, but be safe)
            if dest_file.exists():
                logger.warning(f"Destination already exists, skipping: {dest_file}")
                results["not_found"] += 1
                continue
            
            # Move the file back
            shutil.move(str(source_file), str(dest_file))
            
            logger.info(f"Restored: {season} - {match_id} (Excess: {excess_time} min)")
            results["successfully_restored"] += 1
            results["restored_matches"].append({
                "season": season,
                "match_id": match_id,
                "excess_time_90": excess_time,
                "source_path": str(source_file),
                "destination_path": str(dest_file)
            })
            
        except Exception as e:
            logger.error(f"Error restoring {source_file}: {e}")
            results["errors"] += 1
            results["error_matches"].append({
                "season": season,
                "match_id": match_id,
                "excess_time_90": excess_time,
                "error": str(e)
            })
    
    logger.info("=" * 70)
    logger.info("RESTORATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total 4.0 minute matches: {results['total_4min_matches']}")
    logger.info(f"Successfully restored: {results['successfully_restored']}")
    logger.info(f"Not found: {results['not_found']}")
    logger.info(f"Errors: {results['errors']}")
    
    return results


if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    # Run restoration
    results = restore_4min_matches()
    
    # Exit with error code if any failures occurred
    if results.get("not_found", 0) > 0 or results.get("errors", 0) > 0:
        exit(1)

