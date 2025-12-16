"""
Smoke Test for KickerScraper: Live Integration Test

This script performs a live integration test on a dynamically discovered Bundesliga match
to verify that the KickerScraper implementation is working correctly.

Usage:
    # Run the smoke test with dynamic discovery (default)
    python tests/test_scraper_live.py
    
    # Or from project root
    python -m tests.test_scraper_live
    
    # Test specific known matches
    python -m tests.test_scraper_live --test-known
    python -m tests.test_scraper_live --test-bayern
    python -m tests.test_scraper_live --test-bremen

The test will:
1. Dynamically discover a match URL using get_match_urls(season="2023-24", matchday=34)
   OR test specific known matches (Bayern vs Hoffenheim, Bremen vs Bochum)
2. Scrape the match(es)
3. Save results to tests/artifacts/smoke_test_result.json
4. Perform comprehensive assertions on structure, targets, leakage, and data quality
5. Print a formatted report with test results

Output:
    - tests/artifacts/smoke_test_result.json: Full match data from the scrape
    - Console output: Formatted test report with pass/fail status

Note:
    This is a "self-healing" test that automatically finds a valid match URL,
    avoiding hardcoded URLs that may become invalid over time.
    Use --test-known to verify scraper logic with matches known to have data.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.kicker_scraper import KickerScraper

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)


def test_known_matches(scraper: KickerScraper) -> bool:
    """
    Test scraper with known good and problematic matches to verify logic.
    
    Tests:
    1. Bayern vs Hoffenheim (Known to have HTML fallback) - ID: 4862110
    2. Bremen vs Bochum (Problematic match) - ID: 4862269
    
    Args:
        scraper: Initialized KickerScraper instance
        
    Returns:
        bool: True if Bayern match works (Bremen failure is expected)
    """
    print("=" * 70)
    print("TESTING KNOWN MATCHES: Verifying Scraper Logic")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # 1. Test with the Bayern Match (Known to have HTML fallback)
    print(">>> TESTING KNOWN GOOD MATCH (Bayern vs Hoffenheim) <<<")
    bayern_url = "https://www.kicker.de/bayern-gegen-hoffenheim-2024-bundesliga-4862110/ticker"
    result1 = scraper.scrape_full_match(bayern_url, season="2023-24", matchday=17, force_rescrape=True)
    
    if result1:
        events_count = len(result1.get('ticker_data', []))
        print(f"✓ Bayern match scraped successfully: {events_count} events")
        if events_count > 0:
            print("  → Scraper logic is CORRECT (HTML parsing works)")
        else:
            print("  ⚠️  Warning: No events found despite successful scrape")
            all_passed = False
    else:
        print("✗ Bayern match failed to scrape")
        print("  → This indicates a problem with the scraper logic")
        all_passed = False
    
    print("\n" + "-" * 70 + "\n")
    
    # 2. Test with the Problematic Match (Bremen vs Bochum)
    print(">>> TESTING PROBLEMATIC MATCH (Bremen vs Bochum) <<<")
    bremen_url = "https://www.kicker.de/bremen-gegen-bochum-2024-bundesliga-4862269/ticker"
    result2 = scraper.scrape_full_match(bremen_url, season="2023-24", matchday=34, force_rescrape=True)
    
    if result2:
        events_count = len(result2.get('ticker_data', []))
        print(f"✓ Bremen match scraped successfully: {events_count} events")
        print("  → Match has data (unexpected but good)")
    else:
        print("✗ Bremen match failed to scrape")
        print("  → Expected if data missing on Kicker (edge case)")
    
    print("\n" + "=" * 70)
    print("KNOWN MATCHES TEST COMPLETE")
    print("=" * 70)
    print()
    
    return all_passed


def run_smoke_test(scraper: KickerScraper = None, test_url: str = None, season: str = None, matchday: int = None):
    """
    Execute the smoke test on a single Bundesliga match.
    
    Dynamically discovers a match URL using the crawler, then tests the scraper
    on that match. This makes the test "self-healing" and avoids hardcoded URLs.
    
    Args:
        scraper: Optional KickerScraper instance (creates new one if not provided)
        test_url: Optional specific URL to test (skips discovery if provided)
        season: Optional season string (used if test_url provided)
        matchday: Optional matchday number (used if test_url provided)
    
    Returns:
        bool: True if all tests pass, False otherwise
        
    Raises:
        RuntimeError: If the crawler fails to find any match URLs
    """
    # Create artifacts directory
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_file = artifacts_dir / "smoke_test_result.json"
    
    print("=" * 70)
    print("SMOKE TEST: KickerScraper Live Integration Test")
    print("=" * 70)
    print(f"Output file: {output_file}")
    print()
    
    try:
        # Instantiate scraper (if not provided)
        if scraper is None:
            print("Step 1: Initializing KickerScraper...")
            scraper = KickerScraper(save_dir=project_root / "data" / "raw")
            print("✓ Scraper initialized")
            print()
        else:
            print("Step 1: Using provided scraper instance")
            print()
        
        # Dynamic Discovery: Find match URLs (if not provided)
        if test_url:
            print("Step 2: Using provided test URL...")
            print(f"Testing URL: {test_url}")
            test_matchday = matchday
            if not season:
                season = "2023-24"
        else:
            # Try multiple matchdays (starting from the end) to find a valid match
            print("Step 2: Dynamically discovering match URLs...")
            season = "2023-24"
            match_urls = []
            test_matchday = None
            
            # Try matchdays from 34 down to 1 (most recent first)
            for matchday in range(34, 0, -1):
                print(f"   Trying matchday {matchday}...", end=" ")
                urls = scraper.get_match_urls(season=season, matchday=matchday)
                if urls:
                    match_urls = urls
                    test_matchday = matchday
                    print(f"✓ Found {len(urls)} match(es)")
                    break
                else:
                    print("✗ No matches found")
            
            if not match_urls:
                raise RuntimeError(
                    f"Crawler failed to find matches for {season} across all matchdays (1-34). "
                    "Cannot run smoke test."
                )
            
            # Pick the first URL
            test_url = match_urls[0]
            print(f"✓ Using matchday {test_matchday}: Found {len(match_urls)} match(es)")
            print(f"Testing dynamically found URL: {test_url}")
        print()
        
        # Execute scrape (force_rescrape=True for tests to ensure fresh data)
        print("Step 3: Scraping match data...")
        result = scraper.scrape_full_match(
            match_url=test_url,
            season=season,
            matchday=test_matchday,
            force_rescrape=True  # Force re-scraping for tests
        )
        
        if result is None:
            print("❌ FAILED: scrape_full_match returned None")
            print("   This could indicate:")
            print("   - The match page was not found (404)")
            print("   - The page failed validation")
            print("   - No ticker events were found")
            print()
            print("   The scraper correctly returns None, but this indicates a problem")
            print("   with the dynamically discovered URL or the scraper itself.")
            print()
            return False
        
        print("✓ Match data scraped successfully")
        print()
        
        # Save output
        print("Step 4: Saving results...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✓ Results saved to {output_file}")
        print()
        
        # Assertions: Structure
        print("Step 5: Running assertions...")
        print("-" * 70)
        
        # Assertion 1: Structure - Verify required keys exist
        required_keys = ['metadata', 'targets', 'score_timeline', 'ticker_data']
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            print(f"❌ FAILED: Missing required keys: {missing_keys}")
            return False
        print("✓ Structure: All required keys present")
        
        # Assertion 2: Targets - announced_time_90 is an INTEGER (not None)
        targets = result.get('targets', {})
        announced_time_90 = targets.get('announced_time_90')
        if announced_time_90 is None:
            print("❌ FAILED: targets['announced_time_90'] is None (expected integer)")
            return False
        if not isinstance(announced_time_90, int):
            print(f"❌ FAILED: targets['announced_time_90'] is {type(announced_time_90)}, expected int")
            return False
        print(f"✓ Targets: announced_time_90 = {announced_time_90} (integer)")
        
        # Assertion 3: Leakage Check (CRITICAL) - No "Nachspielzeit" in ticker_data text
        ticker_data = result.get('ticker_data', [])
        leakage_found = []
        for idx, event in enumerate(ticker_data):
            text = event.get('text', '')
            if text and 'nachspielzeit' in text.lower():
                leakage_found.append({
                    'index': idx,
                    'minute': event.get('minute', 'unknown'),
                    'text': text
                })
        
        if leakage_found:
            print("❌ FAILED: Data leakage detected!")
            print("   The word 'Nachspielzeit' (case-insensitive) was found in ticker_data:")
            for leak in leakage_found:
                print(f"   - Event {leak['index']} (minute {leak['minute']}): {leak['text'][:100]}...")
            raise AssertionError(
                f"Data leakage detected: 'Nachspielzeit' found in {len(leakage_found)} ticker event(s)"
            )
        print(f"✓ Leakage Check: No 'Nachspielzeit' found in {len(ticker_data)} ticker events")
        
        # Assertion 4: Score Logic - score_timeline is not empty
        score_timeline = result.get('score_timeline', {})
        if not score_timeline:
            print("❌ FAILED: score_timeline is empty")
            return False
        print(f"✓ Score Logic: score_timeline contains {len(score_timeline)} entries")
        
        # Assertion 5: Ghost Game - is_ghost_game is False
        metadata = result.get('metadata', {})
        is_ghost_game = metadata.get('is_ghost_game', None)
        if is_ghost_game is None:
            print("⚠️  WARNING: is_ghost_game is None (not explicitly set)")
        elif is_ghost_game is True:
            print("❌ FAILED: is_ghost_game is True (expected False for this match)")
            return False
        else:
            print("✓ Ghost Game: is_ghost_game = False (match had fans)")
        
        print("-" * 70)
        print()
        
        # Reporting: Success
        print("=" * 70)
        print("✅ SMOKE TEST PASSED")
        print("=" * 70)
        
        # Extract values for report
        announced_time_90 = targets.get('announced_time_90', 'N/A')
        actual_played_90 = targets.get('actual_played_90', 'N/A')
        attendance = metadata.get('attendance', 'N/A')
        home_team = metadata.get('home_team', 'N/A')
        away_team = metadata.get('away_team', 'N/A')
        final_score = metadata.get('final_score', 'N/A')
        
        print(f"Match: {home_team} vs {away_team}")
        print(f"Final Score: {final_score}")
        print(f"Announced Time (90'): {announced_time_90} minutes")
        print(f"Actual Played Time (90'): {actual_played_90} minutes")
        print(f"Attendance: {attendance:,}" if isinstance(attendance, int) else f"Attendance: {attendance}")
        print(f"Ticker Events: {len(ticker_data)}")
        print(f"Score Timeline Entries: {len(score_timeline)}")
        print("=" * 70)
        
        return True
        
    except RuntimeError as e:
        # Re-raise RuntimeError for crawler failures (expected failure mode)
        print()
        print("=" * 70)
        print("❌ SMOKE TEST FAILED")
        print("=" * 70)
        print(f"Runtime Error: {e}")
        print("=" * 70)
        raise
        
    except AssertionError as e:
        print()
        print("=" * 70)
        print("❌ SMOKE TEST FAILED")
        print("=" * 70)
        print(f"Assertion Error: {e}")
        print("=" * 70)
        return False
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ SMOKE TEST FAILED")
        print("=" * 70)
        print(f"Unexpected Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke test for KickerScraper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--test-known",
        action="store_true",
        help="Test known good and problematic matches (Bayern vs Hoffenheim, Bremen vs Bochum)"
    )
    parser.add_argument(
        "--test-bayern",
        action="store_true",
        help="Test only Bayern vs Hoffenheim (known good match)"
    )
    parser.add_argument(
        "--test-bremen",
        action="store_true",
        help="Test only Bremen vs Bochum (problematic match)"
    )
    
    args = parser.parse_args()
    
    # Initialize scraper once
    scraper = KickerScraper(save_dir=project_root / "data" / "raw")
    
    try:
        if args.test_known:
            # Test both known matches
            success = test_known_matches(scraper)
            sys.exit(0 if success else 1)
        elif args.test_bayern:
            # Test only Bayern match
            bayern_url = "https://www.kicker.de/bayern-gegen-hoffenheim-2024-bundesliga-4862110/ticker"
            success = run_smoke_test(scraper=scraper, test_url=bayern_url, season="2023-24", matchday=17)
            sys.exit(0 if success else 1)
        elif args.test_bremen:
            # Test only Bremen match
            bremen_url = "https://www.kicker.de/bremen-gegen-bochum-2024-bundesliga-4862269/ticker"
            success = run_smoke_test(scraper=scraper, test_url=bremen_url, season="2023-24", matchday=34)
            sys.exit(0 if success else 1)
        else:
            # Default: Dynamic discovery
            success = run_smoke_test(scraper=scraper)
            sys.exit(0 if success else 1)
    finally:
        # Ensure driver is closed
        try:
            scraper.driver.quit()
        except:
            pass

