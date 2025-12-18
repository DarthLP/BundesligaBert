"""
Export Script for Kaggle Dataset Preparation

This script exports processed match data into JSON files ready for Kaggle dataset upload.
It replicates the exact same data preparation and splitting logic as train_bert.py to ensure
reproducibility. The exported JSON files can be uploaded to Kaggle as a private dataset
and used in a Kaggle notebook for GPU-accelerated BERT training.

The script:
1. Loads processed match data from data/processed/season_*/ directories
2. Creates samples for both halves (45 and 90) from each match
3. Splits History data (< 2024-25) into train/val/test_history using 80/10/10 split with match-level splitting
4. Separates Future data (2024-25) as test_future
5. Saves each split as a JSON file with text, label, match_id, season, and half

Usage:
    # Run from project root
    python src/data/export_for_kaggle.py
    
    # Or as module
    python -m src.data.export_for_kaggle

Input:
    Processed JSON files from data/processed/season_{season}/match_{match_id}.json

Output:
    JSON files saved to data/kaggle_export/:
    - train.json: Training samples (80% of History data)
    - val.json: Validation samples (10% of History data)
    - test_history.json: Test History samples (10% of History data)
    - test_future.json: Test Future samples (all 2024-25 season data)
    
    Each JSON file contains an array of objects with structure:
    [
        {
            "text": "BERT input text...",
            "label": 2.0,
            "match_id": "match_12345",
            "season": "2023-24",
            "half": 45
        },
        ...
    ]

Author: BundesligaBERT Project
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility (same as train_bert.py)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_processed_matches(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all processed match files from season directories.
    
    Scans data/processed/ for all season directories matching pattern season_*,
    loads all match_*.json files, and validates required fields.
    
    Args:
        data_dir: Path to data/processed/ directory
        
    Returns:
        List of match dictionaries with all match data
        
    Raises:
        FileNotFoundError: If data_dir doesn't exist
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    matches = []
    season_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("season_")])
    
    logger.info(f"Found {len(season_dirs)} season directories")
    
    for season_dir in season_dirs:
        match_files = list(season_dir.glob("match_*.json"))
        logger.info(f"Loading {len(match_files)} matches from {season_dir.name}")
        
        for match_file in match_files:
            try:
                with open(match_file, 'r', encoding='utf-8') as f:
                    match_data = json.load(f)
                
                # Validate required fields
                required_fields = ['match_id', 'season', 'bert_input_45', 'bert_input_90', 'targets']
                missing_fields = [field for field in required_fields if field not in match_data]
                
                if missing_fields:
                    logger.warning(f"Skipping {match_data.get('match_id', 'unknown')}: missing fields {missing_fields}")
                    continue
                
                matches.append(match_data)
                
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading {match_file}: {e}")
                continue
    
    logger.info(f"Loaded {len(matches)} matches total")
    return matches


def prepare_samples(matches: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Prepare samples from matches by creating samples for both halves.
    
    For each match, creates two samples (one for each half). Filters by season
    to separate History (< 2024-25) from Future (2024-25).
    
    Args:
        matches: List of match dictionaries from load_processed_matches()
        
    Returns:
        Tuple of (history_samples, future_samples) - both are lists of sample dictionaries
    """
    history_samples = []
    future_samples = []
    
    # Separate History and Future
    history_matches = [m for m in matches if m['season'] < "2024-25"]
    future_matches = [m for m in matches if m['season'] == "2024-25"]
    
    logger.info(f"History matches: {len(history_matches)}, Future matches: {len(future_matches)}")
    
    # Create samples from History matches
    skipped_history = 0
    for match in history_matches:
        match_id = match['match_id']
        season = match['season']
        targets = match['targets']
        
        # Sample 1: Half 45
        bert_input_45 = match.get('bert_input_45')
        target_45 = targets.get('announced_45')
        if target_45 is None:
            target_45 = targets.get('actual_45')
        
        if bert_input_45 and target_45 is not None:
            history_samples.append({
                'text': bert_input_45,
                'label': float(target_45),
                'match_id': match_id,
                'season': season,
                'half': 45
            })
        else:
            skipped_history += 1
        
        # Sample 2: Half 90
        bert_input_90 = match.get('bert_input_90')
        target_90 = targets.get('announced_90')
        if target_90 is None:
            target_90 = targets.get('actual_90')
        
        if bert_input_90 and target_90 is not None:
            history_samples.append({
                'text': bert_input_90,
                'label': float(target_90),
                'match_id': match_id,
                'season': season,
                'half': 90
            })
        else:
            skipped_history += 1
    
    if skipped_history > 0:
        logger.warning(f"Skipped {skipped_history} History samples due to missing data")
    
    # Create samples from Future matches
    skipped_future = 0
    for match in future_matches:
        match_id = match['match_id']
        season = match['season']
        targets = match['targets']
        
        # Sample 1: Half 45
        bert_input_45 = match.get('bert_input_45')
        target_45 = targets.get('announced_45')
        if target_45 is None:
            target_45 = targets.get('actual_45')
        
        if bert_input_45 and target_45 is not None:
            future_samples.append({
                'text': bert_input_45,
                'label': float(target_45),
                'match_id': match_id,
                'season': season,
                'half': 45
            })
        else:
            skipped_future += 1
        
        # Sample 2: Half 90
        bert_input_90 = match.get('bert_input_90')
        target_90 = targets.get('announced_90')
        if target_90 is None:
            target_90 = targets.get('actual_90')
        
        if bert_input_90 and target_90 is not None:
            future_samples.append({
                'text': bert_input_90,
                'label': float(target_90),
                'match_id': match_id,
                'season': season,
                'half': 90
            })
        else:
            skipped_future += 1
    
    if skipped_future > 0:
        logger.warning(f"Skipped {skipped_future} Future samples due to missing data")
    
    logger.info(f"Created {len(history_samples)} History samples, {len(future_samples)} Future samples")
    
    return history_samples, future_samples


def split_history_samples(history_samples: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split History samples into train/val/test_history using match-level splitting.
    
    Uses the same 80/10/10 split logic as train_bert.py to ensure reproducibility.
    Splits by match_id to prevent data leakage (all samples from the same match
    go to the same split).
    
    Args:
        history_samples: List of History sample dictionaries
        
    Returns:
        Tuple of (train_samples, val_samples, test_history_samples)
    """
    # Split History by match_id (prevent leakage)
    history_match_ids = list(set([s['match_id'] for s in history_samples]))
    random.Random(RANDOM_SEED).shuffle(history_match_ids)
    
    n_train = int(0.8 * len(history_match_ids))
    n_val = int(0.1 * len(history_match_ids))
    
    train_match_ids = set(history_match_ids[:n_train])
    val_match_ids = set(history_match_ids[n_train:n_train + n_val])
    test_history_match_ids = set(history_match_ids[n_train + n_val:])
    
    train_samples = [s for s in history_samples if s['match_id'] in train_match_ids]
    val_samples = [s for s in history_samples if s['match_id'] in val_match_ids]
    test_history_samples = [s for s in history_samples if s['match_id'] in test_history_match_ids]
    
    logger.info(f"Split History: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_history_samples)}")
    
    return train_samples, val_samples, test_history_samples


def save_json(data: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: List of dictionaries to save
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(data)} samples to {output_path}")


def main():
    """Main function to orchestrate data export."""
    logger.info("Starting Kaggle export pipeline")
    
    # Setup paths
    data_dir = PROJECT_ROOT / "data" / "processed"
    output_dir = PROJECT_ROOT / "data" / "kaggle_export"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading processed matches...")
    matches = load_processed_matches(data_dir)
    
    if len(matches) == 0:
        raise ValueError("No matches loaded! Check data directory.")
    
    # Prepare samples
    logger.info("Preparing samples...")
    history_samples, future_samples = prepare_samples(matches)
    
    # Split History samples
    logger.info("Splitting History samples...")
    train_samples, val_samples, test_history_samples = split_history_samples(history_samples)
    
    # Save JSON files
    logger.info("Saving JSON files...")
    save_json(train_samples, output_dir / "train.json")
    save_json(val_samples, output_dir / "val.json")
    save_json(test_history_samples, output_dir / "test_history.json")
    save_json(future_samples, output_dir / "test_future.json")
    
    # Print summary
    logger.info("=" * 70)
    logger.info("Export Summary:")
    logger.info(f"  Train samples: {len(train_samples)}")
    logger.info(f"  Val samples: {len(val_samples)}")
    logger.info(f"  Test History samples: {len(test_history_samples)}")
    logger.info(f"  Test Future samples: {len(future_samples)}")
    logger.info(f"  Total samples: {len(train_samples) + len(val_samples) + len(test_history_samples) + len(future_samples)}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 70)
    logger.info("Export completed successfully!")
    logger.info("Next steps:")
    logger.info("  1. Zip the JSON files: cd data/kaggle_export && zip bundesliga_bert_data.zip *.json")
    logger.info("  2. Upload to Kaggle: Datasets → New Dataset → Upload")
    logger.info("  3. Use in Kaggle notebook: /kaggle/input/your-dataset-name/")


if __name__ == "__main__":
    main()

