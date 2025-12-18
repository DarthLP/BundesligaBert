"""
Find all matches that exceed 512 tokens in their BERT inputs.

This script scans all processed match files and identifies matches where either
bert_input_45_tokens or bert_input_90_tokens exceeds 512 tokens.

Usage:
    python src/data/find_over_512_tokens.py [--processed-dir data/processed] [--output output.json]

Output:
    Creates a JSON file listing all matches that exceed 512 tokens, including:
    - match_id
    - season
    - bert_input_45_tokens
    - bert_input_90_tokens
    - metadata (home, away, matchday)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_over_512_matches(processed_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan all processed match files and find those exceeding 512 tokens.
    
    Args:
        processed_dir: Path to processed data directory
        
    Returns:
        List of dictionaries containing match information for matches over 512 tokens
    """
    processed_dir = Path(processed_dir)
    over_limit_matches = []
    
    # Find all season directories
    season_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith('season_')])
    
    if not season_dirs:
        logger.warning(f"No season directories found in {processed_dir}")
        return over_limit_matches
    
    logger.info(f"Scanning {len(season_dirs)} seasons for matches over 512 tokens...")
    
    total_scanned = 0
    for season_dir in season_dirs:
        season = season_dir.name.replace('season_', '')
        match_files = list(season_dir.glob('match_*.json'))
        
        logger.info(f"Scanning {season}: {len(match_files)} matches")
        
        for match_file in match_files:
            total_scanned += 1
            try:
                with open(match_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                match_id = data.get('match_id', match_file.stem)
                tokens_45 = data.get('bert_input_45_tokens', 0)
                tokens_90 = data.get('bert_input_90_tokens', 0)
                
                # Check if either exceeds 512
                if tokens_45 > 512 or tokens_90 > 512:
                    match_info = {
                        'match_id': match_id,
                        'season': data.get('season', season),
                        'bert_input_45_tokens': tokens_45,
                        'bert_input_90_tokens': tokens_90,
                        'metadata': {
                            'home': data.get('metadata', {}).get('home', 'Unknown'),
                            'away': data.get('metadata', {}).get('away', 'Unknown'),
                            'matchday': data.get('metadata', {}).get('matchday', 'Unknown'),
                            'final_score': data.get('metadata', {}).get('final_score', 'Unknown')
                        },
                        'file_path': str(match_file.relative_to(processed_dir))
                    }
                    over_limit_matches.append(match_info)
                    
                    logger.warning(
                        f"Found match over 512 tokens: {match_id} ({season}) - "
                        f"45: {tokens_45}, 90: {tokens_90}"
                    )
            
            except Exception as e:
                logger.error(f"Error processing {match_file.name}: {e}")
                continue
    
    logger.info(f"Scanned {total_scanned} matches total")
    logger.info(f"Found {len(over_limit_matches)} matches exceeding 512 tokens")
    
    return over_limit_matches


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Find all matches exceeding 512 tokens')
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Path to processed data directory (default: data/processed)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/results/matches_over_512_tokens.json',
        help='Output JSON file path (default: data/results/matches_over_512_tokens.json)'
    )
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    output_file = Path(args.output)
    
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return
    
    # Find all matches over 512 tokens
    over_limit_matches = find_over_512_matches(processed_dir)
    
    # Group by season for summary
    by_season = {}
    for match in over_limit_matches:
        season = match['season']
        if season not in by_season:
            by_season[season] = []
        by_season[season].append(match)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY BY SEASON")
    logger.info(f"{'='*60}")
    for season in sorted(by_season.keys()):
        count = len(by_season[season])
        logger.info(f"{season}: {count} matches over 512 tokens")
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_matches_over_512': len(over_limit_matches),
            'by_season': by_season,
            'all_matches': over_limit_matches
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Print detailed list
    if over_limit_matches:
        logger.info(f"\n{'='*60}")
        logger.info("DETAILED LIST OF MATCHES OVER 512 TOKENS")
        logger.info(f"{'='*60}")
        for match in over_limit_matches:
            logger.info(
                f"{match['season']} - {match['match_id']}: "
                f"45={match['bert_input_45_tokens']}, 90={match['bert_input_90_tokens']} | "
                f"{match['metadata']['home']} vs {match['metadata']['away']}"
            )


if __name__ == '__main__':
    main()

