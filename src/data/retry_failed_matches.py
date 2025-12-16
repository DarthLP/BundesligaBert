"""
Retry Failed Matches Script

This script reads failed matches from data/failed_matches.json and retries scraping them.
Useful for retrying matches that failed during the initial scraping run.

Usage:
    # Retry all failed matches
    python src/data/retry_failed_matches.py
    
    # Retry only matches from a specific season
    python src/data/retry_failed_matches.py --season 2023-24
    
    # Retry with verbose logging
    python src/data/retry_failed_matches.py --verbose

Output:
    - Retries scraping failed matches
    - Updates failed_matches.json (removes successfully scraped matches)
    - Logs progress and results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.kicker_scraper import KickerScraper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_failed_matches(failed_file: Path) -> list:
    """Load failed matches from JSON file."""
    if not failed_file.exists():
        logger.warning(f"Failed matches file not found: {failed_file}")
        return []
    
    try:
        with open(failed_file, 'r', encoding='utf-8') as f:
            failed_matches = json.load(f)
        logger.info(f"Loaded {len(failed_matches)} failed matches from {failed_file}")
        return failed_matches
    except Exception as e:
        logger.error(f"Error loading failed matches: {e}")
        return []


def save_failed_matches(failed_file: Path, failed_matches: list):
    """Save failed matches to JSON file."""
    try:
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_matches, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(failed_matches)} remaining failed matches to {failed_file}")
    except Exception as e:
        logger.error(f"Error saving failed matches: {e}")


def retry_failed_matches(failed_file: Path, season_filter: str = None, max_retries: int = 3):
    """
    Retry scraping failed matches.
    
    Args:
        failed_file: Path to failed_matches.json
        season_filter: Optional season to filter by (e.g., "2023-24")
        max_retries: Maximum number of retry attempts per match
    """
    # Load failed matches
    failed_matches = load_failed_matches(failed_file)
    
    if not failed_matches:
        logger.info("No failed matches to retry.")
        return
    
    # Filter by season if specified
    if season_filter:
        original_count = len(failed_matches)
        failed_matches = [m for m in failed_matches if m.get('season') == season_filter]
        logger.info(f"Filtered to {len(failed_matches)} matches for season {season_filter} (from {original_count} total)")
    
    if not failed_matches:
        logger.info("No failed matches to retry after filtering.")
        return
    
    # Initialize scraper
    save_dir = project_root / "data" / "raw"
    scraper = KickerScraper(save_dir=save_dir, headless=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RETRYING {len(failed_matches)} FAILED MATCHES")
    logger.info(f"{'='*60}\n")
    
    successful_retries = 0
    still_failed = []
    
    for idx, match_info in enumerate(failed_matches, 1):
        match_url = match_info['url']
        season = match_info.get('season')
        matchday = match_info.get('matchday')
        original_error = match_info.get('error', 'Unknown error')
        
        logger.info(f"[{idx}/{len(failed_matches)}] Retrying: {match_url}")
        logger.info(f"  Season: {season}, Matchday: {matchday}")
        logger.info(f"  Original error: {original_error}")
        
        success = False
        last_error = None
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                result = scraper.scrape_full_match(
                    match_url,
                    season=season,
                    matchday=matchday,
                    force_rescrape=False  # Don't force - skip if already downloaded
                )
                
                if result:
                    successful_retries += 1
                    logger.info(f"  ✓ SUCCESS! Match {result['match_id']} scraped successfully")
                    success = True
                    break
                else:
                    # Check if it was skipped (already downloaded)
                    match_id = match_url.split('/')[-1].split('?')[0]
                    if scraper._is_match_downloaded(match_id, season):
                        successful_retries += 1
                        logger.info(f"  ✓ Match already downloaded (from previous successful retry)")
                        success = True
                        break
                    else:
                        # Still failed
                        if attempt < max_retries - 1:
                            logger.warning(f"  ✗ Attempt {attempt + 1}/{max_retries} failed, retrying...")
                            scraper._random_delay(5, 10)
                        else:
                            last_error = "scrape_full_match returned None"
            
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    logger.warning(f"  ✗ Attempt {attempt + 1}/{max_retries} failed: {e}")
                    logger.info(f"  Retrying in 5-10 seconds...")
                    scraper._random_delay(5, 10)
                else:
                    logger.error(f"  ✗ All {max_retries} attempts failed: {e}")
        
        # If still failed, keep it in the list
        if not success:
            match_info['error'] = last_error or original_error
            match_info['retry_attempts'] = match_info.get('retry_attempts', 0) + 1
            still_failed.append(match_info)
            logger.error(f"  ✗ Still failed after {max_retries} retry attempts")
        
        logger.info("")  # Blank line for readability
        
        # Delay between matches
        scraper._random_delay(4, 8)
    
    # Save updated failed matches list (remove successfully retried ones)
    save_failed_matches(failed_file, still_failed)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"RETRY SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total retried: {len(failed_matches)}")
    logger.info(f"Successfully retried: {successful_retries}")
    logger.info(f"Still failed: {len(still_failed)}")
    logger.info(f"Success rate: {successful_retries/len(failed_matches)*100:.1f}%")
    logger.info(f"{'='*60}")
    
    if still_failed:
        logger.info(f"\n⚠️  {len(still_failed)} matches still failed.")
        logger.info(f"   They remain in {failed_file} for future retry attempts.")
    else:
        logger.info(f"\n✓ All failed matches have been successfully retried!")
        logger.info(f"   {failed_file} has been cleared.")
    
    # Clean up
    try:
        scraper.driver.quit()
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retry failed matches from failed_matches.json",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--season",
        type=str,
        help="Filter by season (e.g., '2023-24'). If not specified, retries all failed matches."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts per match (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    failed_file = project_root / "data" / "failed_matches.json"
    
    retry_failed_matches(
        failed_file=failed_file,
        season_filter=args.season,
        max_retries=args.max_retries
    )

