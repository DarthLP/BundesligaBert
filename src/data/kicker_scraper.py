"""
KickerScraper: Multi-tab scraper for Bundesliga match data from Kicker.de

This module implements a robust scraper using Selenium (undetected-chromedriver) to handle
cookie walls and JavaScript-rendered content. It visits two tabs (Spielinfo, Ticker)
sequentially to extract match metadata and ticker text with precise score states at every minute.
It strictly separates input features from target labels by extracting announced and actual played
time before removing leakage from text.

Usage:
    # Option 1: Run as a script to scrape all seasons (2017-18 to 2024-25)
    python src/data/kicker_scraper.py
    
    # Option 2: Use as a module for individual matches
    from src.data.kicker_scraper import KickerScraper
    from pathlib import Path
    
    scraper = KickerScraper(save_dir=Path("data/raw"))
    result = scraper.scrape_full_match(
        match_url="https://www.kicker.de/...",
        season="2023-24",
        matchday=1
    )
    
    # Option 3: Get match URLs for a specific matchday
    match_urls = scraper.get_match_urls(season="2023-24", matchday=1)
    for url in match_urls:
        scraper.scrape_full_match(url, season="2023-24", matchday=1)

Output:
    Saves JSON files to data/raw/season_{season}/match_{match_id}.json with structure:
    {
        "match_id": "...",
        "metadata": {...},
        "targets": {
            "announced_time_45": 2,
            "actual_played_45": 3,
            "announced_time_90": 4,
            "actual_played_90": 6
        },
        "score_timeline": {...},
        "ticker_data": [...]
    }

Author: BundesligaBERT Project
"""

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)


class KickerScraper:
    """
    Scraper for Bundesliga match data from Kicker.de using Selenium.
    
    Uses undetected-chromedriver to bypass bot detection and handle JavaScript-rendered
    content. Implements a multi-tab scraping strategy:
    1. Spielinfo tab (/spielinfo): Extract metadata (stadium, attendance, referee, stats)
    2. Ticker tab (/ticker): Extract ticker events from server-side rendered HTML
    
    Attributes:
        save_dir: Directory path to save scraped JSON files
        driver: Selenium WebDriver instance (undetected Chrome)
        wait: WebDriverWait instance for explicit waits
    """
    
    def __init__(self, save_dir: Union[str, Path], headless: bool = False) -> None:
        """
        Initialize KickerScraper with Selenium WebDriver.
        
        Args:
            save_dir: Directory to save scraped match JSON files
            headless: If True, run browser in headless mode (default: False for better detection avoidance)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Selenium WebDriver (undetected Chrome)
        logger.info("Initializing Selenium WebDriver...")
        options = uc.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')  # Force Desktop view (Mobile often hides ticker data)
        
        # Initialize driver with retry logic for network timeouts
        max_retries = 3
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                # Try to initialize with version_main=None to use cached driver if available
                # This avoids network calls if a driver is already cached
                self.driver = uc.Chrome(options=options, version_main=None)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    error_msg = str(e)
                    if "timeout" in error_msg.lower() or "URLError" in str(type(e)):
                        logger.warning(f"Network timeout during driver initialization (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        # For other errors, re-raise immediately
                        raise
                else:
                    # Last attempt failed
                    logger.error(f"Failed to initialize WebDriver after {max_retries} attempts: {e}")
                    logger.error("This may be due to network connectivity issues. Please check your internet connection.")
                    raise RuntimeError(f"Failed to initialize Chrome WebDriver: {e}") from e
        
        self.wait = WebDriverWait(self.driver, timeout=10)
        
        # Track if we've handled cookie consent (optimization)
        self._cookie_consent_handled = False
        
        logger.info("WebDriver initialized successfully")
    
    def __del__(self) -> None:
        """Cleanup: Ensure driver is closed on destruction."""
        if hasattr(self, 'driver'):
            try:
                self.driver.quit()
            except Exception:
                pass  # Ignore errors during cleanup
    
    def _random_delay(self, min_sec: float = 2, max_sec: float = 4) -> None:
        """
        Sleep for a random duration between min_sec and max_sec.
        
        Args:
            min_sec: Minimum delay in seconds
            max_sec: Maximum delay in seconds
        """
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)
    
    def _scroll_page(self) -> None:
        """
        Scroll the page slowly to trigger lazy loaders.
        
        The Kicker ticker uses lazy loading, so elements don't exist in the DOM
        until they are scrolled into view. This function performs a slow scroll
        in 4 chunks (25%, 50%, 75%, 100% of document height) with 1 second
        delays between each scroll to allow ticker items to render.
        """
        logger.info("Scrolling page to trigger lazy loaders...")
        try:
            # Get total scroll height
            total_height = self.driver.execute_script("return document.body.scrollHeight")
            
            # Scroll in 4 chunks: 25%, 50%, 75%, 100%
            scroll_positions = [
                int(total_height * 0.25),
                int(total_height * 0.50),
                int(total_height * 0.75),
                total_height
            ]
            
            for i, position in enumerate(scroll_positions, 1):
                self.driver.execute_script(f"window.scrollTo(0, {position});")
                logger.debug(f"Scrolled to {i*25}% of page ({position}px)")
                time.sleep(1)  # Wait 1 second between each scroll to allow rendering
            
            # Final scroll to very bottom to ensure all content is loaded
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            logger.debug("Finished scrolling to bottom")
        except Exception as e:
            logger.warning(f"Error during page scroll: {e}")
    
    def _handle_cookie_consent(self) -> None:
        """
        Handle Usercentrics cookie consent overlay using direct JavaScript execution.
        
        Primary strategy: Execute `ov.cmp.acceptAllConsents()` directly via JavaScript.
        Fallback: Click the "Zustimmen" button if JS execution fails.
        Waits for content to appear after accepting consent.
        Only runs once per session (optimization).
        """
        if self._cookie_consent_handled:
            return
        
        try:
            # Wait longer for the overlay to appear (it may load later)
            time.sleep(2)
            
            # Strategy 1: Direct JavaScript Execution (Primary) - Retry if needed
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    js_result = self.driver.execute_script(
                        "if (typeof ov !== 'undefined' && ov.cmp) { "
                        "ov.cmp.acceptAllConsents(); "
                        "return true; "
                        "} "
                        "return false;"
                    )
                    
                    if js_result:
                        logger.info("Cookies accepted via JS execution.")
                        # Wait for page to reload/render the real content
                        time.sleep(5)
                        
                        # Wait for content to appear (unique match elements)
                        try:
                            # Try to find match-specific content
                            self.wait.until(
                                lambda driver: any([
                                    len(driver.find_elements(By.CSS_SELECTOR, ".kick__scoreboard")) > 0,
                                    len(driver.find_elements(By.CSS_SELECTOR, ".kick__main-head")) > 0,
                                    len(driver.find_elements(By.CSS_SELECTOR, ".kick__match-header")) > 0,
                                    len(driver.find_elements(By.CSS_SELECTOR, "[class*='match']")) > 0,
                                ])
                            )
                            logger.debug("Content appeared after consent acceptance")
                        except Exception:
                            # If waiting fails, just continue (content may already be there)
                            logger.debug("Content wait timeout (may already be loaded)")
                        
                        self._cookie_consent_handled = True
                        logger.info("Cookie consent handled successfully via JS")
                        return
                    else:
                        # JS function not available yet, wait and retry
                        if attempt < max_retries - 1:
                            logger.debug(f"JS consent function not available yet, retrying... (attempt {attempt + 1}/{max_retries})")
                            time.sleep(2)
                            continue
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.debug(f"JS execution failed: {e}, retrying... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2)
                        continue
                    else:
                        logger.debug(f"JS execution failed after {max_retries} attempts: {e}, trying fallback")
            
            # Strategy 2: Fallback - Button Click (using explicit wait with longer timeout)
            try:
                logger.info("JS Consent failed, looking for button...")
                # Use a longer wait for the button to appear (consent may load later)
                button_wait = WebDriverWait(self.driver, timeout=10)
                button = button_wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button.uc-accept-all-button, .kick__cmp__accept, button[class*='accept'], a[class*='accept']"))
                )
                button.click()
                logger.info("Clicked Consent Button.")
                time.sleep(3)
                self._cookie_consent_handled = True
                logger.info("Cookie consent handled successfully via button click")
                return
            except Exception as e:
                logger.debug(f"Button click fallback failed: {e}")
                pass
            
            logger.debug("No cookie consent overlay found (may already be accepted)")
            self._cookie_consent_handled = True  # Mark as handled to avoid repeated checks
            
        except Exception as e:
            logger.debug(f"Error handling cookie consent: {e}")
            # Don't fail the whole scrape if consent handling fails
            self._cookie_consent_handled = True  # Mark as handled to avoid infinite loops
    
    def _extract_announced_time(self, text: str) -> Optional[int]:
        """
        Extract announced added time from text.
        
        Searches for patterns like "4 Minuten Nachspielzeit", "Eine Minute", "Drei Minuten".
        Handles German number words (Eine, Zwei, Drei, etc.) and numeric patterns.
        
        Args:
            text: Event text that may contain time announcements
            
        Returns:
            Extracted minutes as integer, or None if not found
        """
        if not text:
            return None
        
        # German number words to integers
        german_numbers = {
            'eine': 1, 'eins': 1, 'ein': 1,
            'zwei': 2,
            'drei': 3,
            'vier': 4,
            'fünf': 5,
            'sechs': 6,
            'sieben': 7,
            'acht': 8,
            'neun': 9,
            'zehn': 10
        }
        
        text_lower = text.lower()
        
        # Pattern 1: German number words (e.g., "Eine Minute wird nachgespielt", "Drei Minuten gibt es oben drauf")
        # Check for full phrases first to avoid partial matches
        # Sort by length (longest first) to avoid matching "zwei" in "zwölf" or "eine" in "seine"
        sorted_numbers = sorted(german_numbers.items(), key=lambda x: len(x[0]), reverse=True)
        for word, num in sorted_numbers:
            if word in text_lower and ('minute' in text_lower or 'minuten' in text_lower):
                # Check context: should be about added time
                # Look for phrases like "Drei Minuten gibt es oben drauf" or "Eine Minute wird nachgespielt"
                if any(phrase in text_lower for phrase in ['nachgespielt', 'obendrauf', 'oben drauf', 'nachspielzeit', 'gibt es oben drauf', 'wird noch nachgespielt', 'werden noch nachgespielt']):
                    # Check if it's a complete phrase - must have the number word followed by "minute" or "minuten"
                    # Use word boundaries to ensure we match the full word
                    pattern = rf'\b{word}\s+(?:minute|minuten)\b'
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        return num
        
        # Pattern 2: Numeric patterns (e.g., "+4", "+ 4", "4 Minuten Nachspielzeit")
        patterns = [
            r'(\d+)\s*Minuten?\s*(?:wird|werden|gibt|gibt es|obendrauf|oben drauf|nachgespielt|Nachspielzeit)',  # "4 Minuten wird nachgespielt"
            r'(\d+)\s*Min\.?\s*(?:Nachspielzeit|obendrauf)',  # "4 Min. Nachspielzeit"
            r'Nachspielzeit[:\s]+(\d+)',  # "Nachspielzeit: 4"
            r'(\+ ?\d+)',  # "+4" or "+ 4" (fallback, less reliable)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    # Extract the number (could be from different groups)
                    num_str = match.group(1) if match.lastindex >= 1 else match.group(0)
                    # Remove + sign if present
                    num_str = num_str.replace('+', '').strip()
                    return int(num_str)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_actual_played_time(self, minute: str, text: str) -> Optional[int]:
        """
        Extract actual played time from event timestamp.
        
        Parses minute strings like "90+5" or "45+2" for Abpfiff/Halbzeit events.
        Also handles "45+1" and "90+1" format.
        
        Args:
            minute: Minute string (may include "+" notation, e.g., "45+1", "90+1")
            text: Event text to check if it's Abpfiff or Halbzeit
            
        Returns:
            Extracted added time as integer, or None if not found
        """
        if not minute:
            return None
        
        # Only extract for Abpfiff/Halbzeit events
        text_lower = (text or "").lower()
        is_halbzeit = "halbzeitpfiff" in text_lower or "halbzeit" in text_lower
        is_abpfiff = "abpfiff" in text_lower or "spielende" in text_lower
        
        if not (is_halbzeit or is_abpfiff):
            return None
        
        # Match pattern like "90+5" or "45+2" or "45+1"
        match = re.search(r'(\d+)\+(\d+)', minute)
        if match:
            try:
                base_minute = int(match.group(1))
                added_time = int(match.group(2))
                
                # Verify it matches the event type
                if is_halbzeit and base_minute == 45:
                    return added_time
                elif is_abpfiff and base_minute >= 90:
                    return added_time
            except (ValueError, IndexError):
                return None
        
        return None
    
    def _extract_score_from_text(self, text: str, is_goal_event: bool = False) -> Optional[Tuple[int, int]]:
        """
        Extract score from ticker text using regex patterns.
        
        Only extracts scores from actual goal events to avoid false positives from commentary.
        
        Args:
            text: Event text that may contain score information
            is_goal_event: Whether this is actually a goal event (not just mentioning score)
            
        Returns:
            Tuple of (home_score, away_score) or None if not found
        """
        if not text:
            return None
        
        # Only extract score if it's actually a goal event
        # Otherwise, scores mentioned in commentary (e.g., "Mit Ausnahme des 0:1") are not real goals
        if not is_goal_event:
            return None
        
        # Pattern to match score: "1:0", "2:1", "10:2", etc.
        # Look for patterns like "1:0" or "1 - 0" or "(1:0)"
        # Prefer patterns that are more likely to be actual goal scores
        score_patterns = [
            r'\((\d+)\s*:\s*(\d+)\)',  # (1:0) - often used for goal scores
            r'(\d+)\s*:\s*(\d+)',  # 1:0
            r'(\d+)\s*-\s*(\d+)',  # 1 - 0
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    home_score = int(match.group(1))
                    away_score = int(match.group(2))
                    # Sanity check: scores should be reasonable (0-20 for football)
                    if 0 <= home_score <= 20 and 0 <= away_score <= 20:
                        return (home_score, away_score)
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _process_ticker_event(
        self, text: str, minute: str, is_goal_event: bool = False
    ) -> Tuple[str, Optional[int], Optional[int], Optional[Tuple[int, int]]]:
        """
        Process a ticker event to extract targets and remove leakage.
        
        Extracts:
        1. Actual played time from event timestamp (for Abpfiff/Halbzeit)
        2. Announced time from text announcements
        3. Score from text (for goal events only)
        
        Then removes announcement sentences from text to prevent leakage.
        
        Args:
            text: Raw event text
            minute: Minute string (may include "+" notation)
            is_goal_event: Whether this is a goal event
            
        Returns:
            Tuple of (clean_text, announced_time, actual_played_time, extracted_score)
            - extracted_score: (home_score, away_score) or None
        """
        # Extract targets first (before removing leakage)
        announced_time = self._extract_announced_time(text)
        actual_played_time = self._extract_actual_played_time(minute, text)
        extracted_score = self._extract_score_from_text(text, is_goal_event=is_goal_event)
        
        # Remove leakage: entire sentences containing time announcements
        clean_text = text
        
        # Pattern to match sentences with time announcements
        # Match sentence boundaries and remove sentences containing time info
        patterns_to_remove = [
            r'[^.!?]*(?:Nachspielzeit|Minuten obendrauf|Minuten wird nachgespielt|Minuten gibt es oben drauf)[^.!?]*[.!?]?',
            r'[^.!?]*(\d+)\s*Minuten?\s*(?:Nachspielzeit|obendrauf|wird nachgespielt|gibt es oben drauf)[^.!?]*[.!?]?',
            r'[^.!?]*Nachspielzeit[:\s]+(\d+)[^.!?]*[.!?]?',
        ]
        
        for pattern in patterns_to_remove:
            clean_text = re.sub(pattern, '', clean_text, flags=re.IGNORECASE)
        
        # Remove minute text from beginning of text ONLY (e.g., "90' +2 Kanes..." -> "+2 Kanes...")
        # Pattern: minute with apostrophe at start of text, optionally followed by space and +number
        clean_text = re.sub(r'^\d+\'?\s*', '', clean_text)
        
        # Remove "+X" patterns ONLY at the beginning of text (after minute removal)
        # This handles cases like "+2 Kanes..." but doesn't remove "+" patterns in the middle of sentences
        clean_text = re.sub(r'^\+\s*\d+\s+', '', clean_text)
        
        # Remove time patterns from beginning of text (e.g., ":21 Uhr" or "11:21 Uhr")
        clean_text = re.sub(r'^:\d+\s*Uhr\s*', '', clean_text)  # ":21 Uhr"
        clean_text = re.sub(r'^\d+:\d+\s*Uhr\s*', '', clean_text)  # "11:21 Uhr"
        
        # Clean up extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return (clean_text, announced_time, actual_played_time, extracted_score)
    
    def _extract_player_name_from_text(self, text: str, is_card: bool = False) -> Optional[str]:
        """
        Extract player name from ticker text for card/substitution events.
        
        Looks for patterns like "Müller sieht gelb" or "Wechsel: Müller für Müller".
        Player names are typically capitalized and appear before or after event keywords.
        
        Args:
            text: Event text
            is_card: Whether this is a card event (vs substitution)
            
        Returns:
            Player name string or None if not found
        """
        if not text:
            return None
        
        # Common patterns for player names in German ticker text
        # Pattern 1: "Müller sieht gelb" or "Müller sieht rot"
        # Pattern 2: "Gelb für Müller" or "Rot für Müller"
        # Pattern 3: "Müller (Team) sieht gelb"
        
        # Try to find capitalized words (likely player names) near card/sub keywords
        patterns = []
        if is_card:
            # Look for pattern: "Player sieht gelb/rot" or "Gelb/Rot für Player"
            patterns = [
                r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+(?:sieht|bekommt|erhält)\s+(?:gelb|rot|gelb-rot)',
                r'(?:gelb|rot|gelb-rot|gelbe karte|rote karte)\s+(?:für|an)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)',
                r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s*\([^)]+\)\s*(?:sieht|bekommt)',
            ]
        else:
            # Look for pattern: "Wechsel: Player für Player" or "Player kommt für Player"
            patterns = [
                r'[Ww]echsel[:\s]+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+(?:für|kommt für)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)',
                r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+(?:kommt|tritt)\s+(?:für|an die Stelle von)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)',
            ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return first captured group (player name)
                if match.lastindex >= 1:
                    return match.group(1).strip()
        
            return None
    
    def _extract_team_from_text(self, text: str) -> Optional[str]:
        """
        Extract team name from ticker text.
        
        Looks for team names in parentheses or after player names.
        Common pattern: "Müller (Bayern) sieht gelb"
        
        Args:
            text: Event text
            
        Returns:
            Team name string or None if not found
        """
        if not text:
            return None
        
        # Pattern: "Player (Team)" or "Team: Player"
        team_patterns = [
            r'\(([^)]+)\)',  # Text in parentheses (often team name)
            r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*):\s*[A-ZÄÖÜ]',  # "Team: Player"
        ]
        
        for pattern in team_patterns:
            match = re.search(pattern, text)
            if match:
                team = match.group(1).strip()
                # Filter out common false positives
                if team.lower() not in ['aus', 'ein', 'für', 'sieht', 'bekommt']:
                    return team
        
        return None
    
    def _extract_substitution_from_text(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract substitution information from ticker text.
        
        Looks for patterns like "Spielerwechsel: Müller für Müller" or "Wechsel: Player für Player".
        Handles multiple substitutions at the same time (e.g., "Wechsel: Player1 für Player2, Player3 für Player4").
        
        Args:
            text: Event text containing substitution information
            
        Returns:
            List of dictionaries with 'player_in', 'player_out', 'team' or None if not found
            Returns list to handle multiple subs at the same minute
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Check if text contains substitution keywords
        if 'wechsel' not in text_lower and 'spielerwechsel' not in text_lower:
            return None
        
        substitutions = []
        
        # Pattern 1: "Spielerwechsel: PlayerIn für PlayerOut" or "Wechsel: PlayerIn für PlayerOut"
        # Pattern 2: "PlayerIn kommt für PlayerOut"
        # Pattern 3: "PlayerIn (Team) für PlayerOut"
        # Pattern 4: Multiple subs: "Wechsel: Player1 für Player2, Player3 für Player4"
        
        # First, try to find all "für" patterns (handles multiple subs)
        # Look for pattern: "PlayerIn für PlayerOut" (can appear multiple times)
        multi_sub_pattern = r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+(?:für|kommt für)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)'
        
        all_matches = re.finditer(multi_sub_pattern, text)
        for match in all_matches:
            if match.lastindex >= 2:
                player_in = match.group(1).strip()
                player_out = match.group(2).strip()
                
                # Extract team if present (check context around this match)
                # Look for team in parentheses near this substitution
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text), match.end() + 50)
                context = text[start_pos:end_pos]
                team = self._extract_team_from_text(context)
                
                substitutions.append({
                    'player_in': player_in,
                    'player_out': player_out,
                    'team': team
                })
        
        # If we found substitutions, return them
        if substitutions:
            return substitutions
        
        # Fallback: Try single substitution patterns (for cases where finditer doesn't work)
        single_sub_patterns = [
            r'(?:[Ss]pieler)?[Ww]echsel[:\s]+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+(?:für|kommt für)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)',
            r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+(?:kommt|tritt)\s+(?:für|an die Stelle von)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)',
            r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s*\([^)]+\)\s+(?:für|kommt für)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)',
        ]
        
        for pattern in single_sub_patterns:
            match = re.search(pattern, text)
            if match:
                if match.lastindex >= 2:
                    player_in = match.group(1).strip()
                    player_out = match.group(2).strip()
                    
                    # Extract team if present
                    team = self._extract_team_from_text(text)
                    
                    return [{
                        'player_in': player_in,
                        'player_out': player_out,
                        'team': team
                    }]
        
        return None
    
    def _parse_german_number(self, text: str) -> Optional[int]:
        """
        Parse German number format (e.g., "22.012" -> 22012).
        
        Handles non-numeric inputs gracefully by returning None.
        
        Args:
            text: German formatted number string with dots as thousand separators
            
        Returns:
            Parsed integer or None if parsing fails
        """
        if not text or text.strip() == "-" or text.strip() == "":
            return None
        
        try:
            # Remove dots (thousand separators) and whitespace
            cleaned = text.replace('.', '').strip()
            return int(cleaned)
        except (ValueError, AttributeError):
            logger.debug(f"Failed to parse German number: {text}")
            return None
    
    def _parse_german_float(self, text: str) -> Optional[float]:
        """
        Parse German float format (e.g., "3,5" -> 3.5).
        
        Args:
            text: German formatted float string with comma as decimal separator
            
        Returns:
            Parsed float or None if parsing fails
        """
        if not text or text.strip() == "-" or text.strip() == "":
            return None
        
        try:
            # Replace comma with dot and convert
            cleaned = text.replace(',', '.').strip()
            return float(cleaned)
        except (ValueError, AttributeError):
            logger.debug(f"Failed to parse German float: {text}")
            return None
    
    def _detect_ghost_game(self, attendance: Optional[int]) -> bool:
        """
        Detect if a match is a "ghost game" (Corona era with no/low attendance).
        
        Args:
            attendance: Number of attendees
            
        Returns:
            True if attendance < 1000 (ghost game), False otherwise
        """
        if attendance is None:
            return False
        return attendance < 1000
    
    def _parse_minute_to_int(self, minute_str: str) -> Optional[int]:
        """
        Parse minute string to integer (handle "45+2" -> 45).
        
        Args:
            minute_str: Minute string
            
        Returns:
            Integer minute or None
        """
        if not minute_str:
            return None
        
        match = re.search(r'(\d+)', minute_str)
        if match:
            return int(match.group(1))
            return None
    
    def get_match_urls(self, season: str, matchday: int) -> List[str]:
        """
        Crawl Spieltag overview page to extract all match URLs using Selenium.
        
        Parses the matchday overview page and extracts valid match URLs using strict filtering.
        A valid match URL must contain "-gegen-" and exclude analysis pages.
        
        Args:
            season: Season string (e.g., "2023-24")
            matchday: Matchday number (1-34)
            
        Returns:
            List of full match URLs (deduplicated, sorted)
        """
        # 1. Construct URL
        url = f"https://www.kicker.de/bundesliga/spieltag/{season}/{matchday}"
        logger.info(f"Fetching match URLs for {season}, matchday {matchday}")
        
        # 2. Navigate and get HTML
        try:
            # Check if window is still open
            try:
                self.driver.current_window_handle
            except Exception:
                logger.error("Browser window was closed. Reinitializing driver...")
                # Reinitialize driver if window was closed
                options = uc.ChromeOptions()
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-blink-features=AutomationControlled')
                self.driver = uc.Chrome(options=options)
                self.wait = WebDriverWait(self.driver, timeout=10)
                self._cookie_consent_handled = False  # Reset consent flag
            
            self.driver.get(url)
            self._random_delay(2, 3)
            self._handle_cookie_consent()
            
            # Scroll down to ensure all content is loaded (lazy loading)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            self._random_delay(1, 2)
            self.driver.execute_script("window.scrollTo(0, 0);")  # Scroll back to top
            self._random_delay(1, 2)
            
            html = self.driver.page_source
        except Exception as e:
            logger.error(f"Failed to fetch matchday page: {url} - {e}")
            return []
        
        if not html or len(html) < 5000:
            logger.warning(f"Page seems short ({len(html)} chars)")
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        match_urls_set = set()  # Use set for deduplication
        
        # 3. Link Extraction: Find all <a> tags
        links = soup.find_all('a', href=True)
        logger.info(f"Found {len(links)} total links on matchday page")
        
        # Keywords to exclude (only media galleries and women's league - we want analyse, liveticker)
        exclude_keywords = ['fotos', 'video', 'tabelle', 'frauen']
        
        for link in links:
            href = link.get('href', '').strip()
            
            # Filter: A link is valid if and only if:
            # 1. Contains "-gegen-" (German for "-vs-")
            if '-gegen-' not in href:
                continue
            
            # 2. Must contain "bundesliga" (not "2-bundesliga" or other variations)
            # But exclude "frauen-bundesliga" (women's league)
            if 'bundesliga' not in href.lower():
                continue
            if 'frauen' in href.lower() or '2-bundesliga' in href.lower():
                continue
            
            # 3. Does NOT contain excluded keywords
            if any(keyword in href.lower() for keyword in exclude_keywords):
                continue
            
            # 4. Starts with '/' (relative) or 'https://www.kicker.de' (absolute)
            if not (href.startswith('/') or href.startswith('https://www.kicker.de')):
                continue
            
            # 4. Cleaning: Convert relative links to absolute
            if href.startswith('/'):
                    full_url = f"https://www.kicker.de{href}"
            elif href.startswith('https://www.kicker.de'):
                    full_url = href
            else:
                continue
            
            # Strip known suffixes to get clean base match URL
            clean_url = re.sub(r'/(analyse|liveticker|ticker|spielinfo|spielbericht)$', '', full_url)
            clean_url = clean_url.rstrip('/')
            
            # Add to set (automatic deduplication)
            match_urls_set.add(clean_url)
        
        # Convert set to sorted list for consistent output
        match_urls = sorted(list(match_urls_set))
        
        logger.info(f"Found {len(match_urls)} match URLs for {season}, matchday {matchday}")
        return match_urls
    
    def _validate_match_page(self, html: str, page_type: str = "match") -> bool:
        """
        Validate that HTML is actually a match page (not an error page).
        
        Args:
            html: HTML content to validate
            page_type: Type of page ("match", "info", "ticker")
            
        Returns:
            True if page appears valid, False otherwise
        """
        if not html or len(html) < 1000:
            return False
        
        html_lower = html.lower()
        
        # Check for match-specific content indicators
        match_indicators = [
            'bundesliga', 'spielinfo', 'liveticker',
            'schiedsrichter', 'zuschauer'
        ]
        
        # If none of the match indicators are found, likely not a match page
        if not any(indicator in html_lower for indicator in match_indicators):
            logger.debug(f"Page validation failed: no match indicators found")
            return False
        
        return True
    
    def parse_metadata(self, html: str) -> Dict[str, Any]:
        """
        Parse metadata from /spielinfo page.
        
        Extracts: stadium, attendance, is_sold_out, referee, referee_note.
        Also applies ghost game detection and sets has_var flag.
        
        Args:
            html: HTML content of /spielinfo page
            
        Returns:
            Dictionary with metadata fields
        """
        # Validate that this is actually a match info page
        if not self._validate_match_page(html, "info"):
            logger.warning("HTML does not appear to be a valid match info page")
            return {
                'stadium': None,
                'attendance': None,
                'is_sold_out': False,
                'referee': None,
                'referee_note': None,
                'is_ghost_game': False,
                'has_var': True
            }
        
        soup = BeautifulSoup(html, 'html.parser')
        metadata = {
            'stadium': None,
            'attendance': None,
            'is_sold_out': False,
            'referee': None,
            'referee_note': None,
            'is_ghost_game': False,
            'has_var': True  # VAR introduced in 2017-18
        }
        
        try:
            # Extract stadium - look for common patterns
            stadium_elem = soup.find(string=re.compile(r'Stadion|Arena', re.I))
            if stadium_elem:
                parent = stadium_elem.find_parent()
                if parent:
                    metadata['stadium'] = parent.get_text(strip=True)
            
            # Extract attendance - look for numbers with dots (German format)
            attendance_text = soup.find(string=re.compile(r'\d+\.\d+', re.I))
            if attendance_text:
                parent = attendance_text.find_parent()
                if parent:
                    text = parent.get_text()
                    match = re.search(r'(\d+\.?\d*)', text)
                    if match:
                        metadata['attendance'] = self._parse_german_number(match.group(1))
            
            # Check for sold out
            if soup.find(string=re.compile(r'ausverkauft', re.I)):
                metadata['is_sold_out'] = True
            
            # Extract referee
            referee_elem = soup.find(string=re.compile(r'Schiedsrichter|Referee', re.I))
            if referee_elem:
                parent = referee_elem.find_parent()
                if parent:
                    next_sibling = parent.find_next_sibling()
                    if next_sibling:
                        metadata['referee'] = next_sibling.get_text(strip=True)
            
            # Extract referee note (usually a float like "3,5")
            note_elem = soup.find(string=re.compile(r'\d+,\d+', re.I))
            if note_elem:
                parent = note_elem.find_parent()
                if parent:
                    text = parent.get_text()
                    match = re.search(r'(\d+,\d+)', text)
                    if match:
                        metadata['referee_note'] = self._parse_german_float(match.group(1))
            
            # Apply ghost game detection
            metadata['is_ghost_game'] = self._detect_ghost_game(metadata['attendance'])
            
        except Exception as e:
            logger.warning(f"Error parsing metadata: {e}")
        
        return metadata
    
    def parse_ticker(self, html: str) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[int]], Dict[int, Tuple[int, int]]]:
        """
        Parse ticker events from /ticker page using correct HTML selectors.
        
        Uses the actual HTML structure:
        - Container: `.kick__ticker-match-stream`
        - Items: `.kick__ticker-item`
        - Minute: `.kick__ticker-item_min` (may contain "90'<span>+3</span>")
        - Text: `<p>` tag within the item
        
        Also extracts scores from goal events to build score timeline.
        
        Args:
            html: HTML content of /ticker page
            
        Returns:
            Tuple of (ticker_events, targets_dict, score_timeline)
            - ticker_events: List of event dictionaries with cleaned text
            - targets_dict: Dictionary with announced_time_45/90 and actual_played_45/90
            - score_timeline: Dictionary mapping minute -> (home_score, away_score)
        """
        ticker_events = []
        targets = {
            'announced_time_45': None,
            'actual_played_45': None,
            'announced_time_90': None,
            'actual_played_90': None
        }
        score_timeline = {0: (0, 0)}  # Initialize with 0-0 at minute 0
        current_home = 0  # Track score state as we process (like Gemini)
        current_away = 0
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Select ticker items directly from soup (no parent container requirement)
            ticker_items = soup.select('.kick__ticker-item')
            logger.info(f"Found {len(ticker_items)} ticker items")
            
            if not ticker_items:
                logger.warning("No ticker items found with selector '.kick__ticker-item'")
                # Debug dump
                debug_dir = Path(__file__).parent.parent.parent / "tests" / "artifacts"
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_file = debug_dir / "debug_failed_ticker.html"
                try:
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(html)
                    logger.warning(f"Debug HTML saved to {debug_file}")
                except Exception as e:
                    logger.error(f"Failed to save debug HTML: {e}")
                return (ticker_events, targets, score_timeline)
            
            # Track max minutes for actual_played calculation
            max_minute_45 = None
            max_minute_90 = None
            
            # Process items in reverse order (like Gemini - ensures chronological processing)
            for item in reversed(ticker_items):
                # Extract minute from .kick__ticker-item_min
                minute_elem = item.select_one('.kick__ticker-item_min')
                minute = ""
                is_pre_match = False
                is_halftime = False
                is_post_match = False
                
                if minute_elem:
                    # Get all text including nested spans (e.g., "90'<span>+3</span>")
                    minute_text = minute_elem.get_text(strip=True)
                    # Remove apostrophe and clean up
                    minute = minute_text.replace("'", "").strip()
                    # Handle format like "90+3" - already correct
                    # Handle format like "90 +3" - normalize spaces
                    minute = re.sub(r'\s*\+\s*', '+', minute)
                    
                    # Track max minutes for actual_played calculation
                    minute_int = self._parse_minute_to_int(minute)
                    if minute_int is not None:
                        # Check if it's in first half (<= 45) or second half (> 45)
                        if minute_int <= 45:
                            # Check for overtime (e.g., "45+1" -> 1 minute overtime)
                            if '+' in minute:
                                match = re.search(r'45\+(\d+)', minute)
                                if match:
                                    overtime = int(match.group(1))
                                    if max_minute_45 is None or overtime > max_minute_45:
                                        max_minute_45 = overtime
                        elif minute_int >= 90:
                            # Check for overtime (e.g., "90+3" -> 3 minutes overtime)
                            if '+' in minute:
                                match = re.search(r'90\+(\d+)', minute)
                                if match:
                                    overtime = int(match.group(1))
                                    if max_minute_90 is None or overtime > max_minute_90:
                                        max_minute_90 = overtime
                else:
                    # No minute element - could be pre-match, halftime, or post-match
                    # Check text content to determine
                    text_lower = item.get_text(strip=True).lower()
                    if any(word in text_lower for word in ['anpfiff', 'spielbeginn', 'vor dem spiel', 'vor dem anpfiff']):
                        minute = ""  # Empty for pre-match
                        is_pre_match = True
                    elif 'halbzeit' in text_lower:
                        minute = "45"
                        is_halftime = True
                    elif any(word in text_lower for word in ['abpfiff', 'spielende', 'nach dem spiel', 'nach dem abpfiff']):
                        minute = ""  # Empty for post-match
                        is_post_match = True
                    else:
                        minute = ""  # Unknown - will be skipped if no text
                
                # Extract text from ALL <p> tags - get complete text content
                text_elems = item.find_all('p')
                text = ""
                if text_elems:
                    # Get full text from ALL paragraphs (preserve full sentences)
                    # Use separator=' ' to preserve sentence structure
                    text_parts = []
                    for p in text_elems:
                        # Get all text including nested elements, preserving structure
                        p_text = p.get_text(separator=' ', strip=False)
                        text_parts.append(p_text)
                    text = ' '.join(text_parts)
                    # Clean up excessive whitespace but preserve sentence structure
                    text = re.sub(r'\s+', ' ', text).strip()
                else:
                    # Fallback: get text from the item directly (some events are just metadata)
                    # But exclude the minute element text
                    item_text = item.get_text(separator=' ', strip=False)
                    if minute_elem:
                        minute_text_in_item = minute_elem.get_text(strip=True)
                        # Remove minute text from item text (only exact match)
                        text = item_text.replace(minute_text_in_item, '', 1).strip()  # Only replace first occurrence
                    else:
                        text = item_text
                    # Clean up excessive whitespace
                    text = re.sub(r'\s+', ' ', text).strip()
                
                # Extract announced_time BEFORE skipping (so we capture it even if we skip the event)
                # We need to extract from the raw text before any processing
                # Only skip if this SPECIFIC event is a nachspielzeit announcement
                # Other events at the same minute should NOT be skipped
                # Make detection VERY specific to avoid false positives
                # Only match actual overtime announcement phrases
                text_lower_check = text.lower() if text else ""
                is_nachspielzeit_announcement = any(phrase in text_lower_check for phrase in [
                    # Standard Phrasing
                    'minuten wird nachgespielt', 
                    'minuten werden nachgespielt',
                    'minute wird nachgespielt',
                    'minute wird noch nachgespielt',
                    'minuten wird noch nachgespielt',
                    'minuten werden noch nachgespielt',
                    'nachspielzeit:',
                    'nachspielzeit wird',
                    'nachspielzeit beträgt',
                    'angezeigte nachspielzeit',
                    
                    # "On Top" variations
                    'minuten gibt es oben drauf',
                    'minuten obendrauf',
                    'gibt es oben drauf',
                    'minuten drauf',
                    
                    # Colloquial / Synonyms
                    'minuten nachschlag',
                    'minuten zugabe',
                    'minuten bonus',
                    'minuten extra',
                    'zeigerumdrehungen', # Cliché for "minutes"
                    
                    # The Action/Official
                    'die tafel zeigt',
                    'auf der tafel',
                    'vierter offizielle', # Covers "Der vierte Offizielle zeigt..."
                ])
                
                # Extract announced_time from this event if it's a nachspielzeit announcement
                # Only skip THIS specific event, not other events at the same minute
                if is_nachspielzeit_announcement:
                    # Extract announced time before skipping
                    temp_announced_time = self._extract_announced_time(text)
                    if temp_announced_time is not None:
                        minute_num = self._parse_minute_to_int(minute)
                        if minute_num and minute_num <= 45:
                            # We process events in reverse order (newest first)
                            # We want the LAST announcement chronologically (closest to halftime)
                            # So we always update to keep the latest one we've seen
                            targets['announced_time_45'] = temp_announced_time
                        elif minute_num and minute_num > 45:
                            # We want the LAST announcement chronologically (closest to fulltime)
                            targets['announced_time_90'] = temp_announced_time
                    # Now skip ONLY this specific event (leakage prevention)
                    # Other events at the same minute will be processed normally
                    continue
                
                # Skip if no meaningful text (unless it's a special event)
                if not text and not (is_pre_match or is_halftime or is_post_match):
                    continue
                
                # Extract score from scoreboard container (CRITICAL for fallback)
                # BUT: Only update score state for goal events to avoid wrong scores on regular events
                score_container = item.select_one('.kick__scoreboard-ticker_container')
                extracted_score = None
                if score_container:
                    score_text = score_container.get_text(strip=True)
                    # Extract score pattern like "2 : 0" or "1:0"
                    score_match = re.search(r'(\d+)\s*:\s*(\d+)', score_text)
                    if score_match:
                        home_score = int(score_match.group(1))
                        away_score = int(score_match.group(2))
                        # Sanity check: scores should be reasonable
                        if 0 <= home_score <= 20 and 0 <= away_score <= 20:
                            extracted_score = (home_score, away_score)
                
                # Detect event type from classes and icons (use boolean flags internally)
                item_classes = item.get('class', [])
                text_lower = text.lower() if text else ""
                
                # Detect goal
                is_goal = 'kick__ticker-item--highlight' in item_classes
                
                # Detect card
                is_card = (
                    'kick__ticker-item--card' in item_classes or
                    item.select_one('.kick__icon-Gelb') or
                    item.select_one('.kick__icon-Rot') or
                    any(phrase in text_lower for phrase in ['gelbe karte', 'gelb-rote karte', 'rote karte', 'gelb-rot', 'gelb rot'])
                )
                
                # Detect substitution
                is_substitution = (
                    'kick__ticker-item--substitution' in item_classes or
                    item.select_one('.kick__icon-Wechsel') or
                    'wechsel' in text_lower or
                    'spielerwechsel' in text_lower
                )
                
                # Extract card type from text if it's a card event
                card_type = None
                card_player = None
                card_team = None
                if is_card:
                    if 'gelb-rote karte' in text_lower or 'gelb-rot' in text_lower or 'gelb rot' in text_lower:
                        card_type = 'Yellow-Red'
                    elif 'rote karte' in text_lower or 'rot' in text_lower:
                        card_type = 'Red'
                    elif 'gelbe karte' in text_lower or 'gelb' in text_lower:
                        card_type = 'Yellow'
                    
                    # Extract player name and team for card events
                    card_player = self._extract_player_name_from_text(text, is_card=True)
                    card_team = self._extract_team_from_text(text)
                
                # Extract substitution information from text
                substitution_info_list = None
                if is_substitution:
                    substitution_info_list = self._extract_substitution_from_text(text)
                
                # Process event to extract targets and clean text
                clean_text, announced_time, actual_played_time, text_extracted_score = self._process_ticker_event(
                    text, minute, is_goal_event=is_goal
                )
                
                # Use text-extracted score ONLY if:
                # 1. We didn't get one from scoreboard container
                # 2. It's actually a goal event (not just mentioning a score in commentary)
                if not extracted_score and text_extracted_score and is_goal:
                    extracted_score = text_extracted_score
                
                # Update score state and timeline ONLY for goal events
                # This ensures that regular events at the same minute don't get the wrong score
                if is_goal and extracted_score:
                    current_home, current_away = extracted_score
                    minute_int = self._parse_minute_to_int(minute)
                    if minute_int is not None:
                        score_timeline[minute_int] = extracted_score
                        logger.debug(f"Updated score to {current_home}:{current_away} at minute {minute_int} from goal event")
                
                # Store targets (will be updated after processing all events with max minutes)
                
                # NOTE: announced_time extraction from regular events is disabled
                # We only extract announced_time from actual nachspielzeit announcements (handled above)
                # This prevents false positives from regular events that might mention numbers
                
                # Store event (no event_type field)
                event_dict = {
                    'minute': minute,
                    'text': clean_text
                }
                
                # Add card information if it's a card event
                if is_card and card_type:
                    event_dict['card_type'] = card_type
                    if card_player:
                        event_dict['player'] = card_player
                    if card_team:
                        event_dict['team'] = card_team
                
                # Add substitution information if it's a substitution event
                # Handle multiple substitutions at the same minute by creating separate events
                if is_substitution and substitution_info_list:
                    # If multiple substitutions, create separate events for each
                    if len(substitution_info_list) > 1:
                        # Create additional events for each substitution after the first
                        for sub_info in substitution_info_list[1:]:
                            sub_event = event_dict.copy()
                            sub_event['player_in'] = sub_info.get('player_in')
                            sub_event['player_out'] = sub_info.get('player_out')
                            if sub_info.get('team'):
                                sub_event['team'] = sub_info.get('team')
                            ticker_events.append(sub_event)
                    
                    # Add first substitution to main event
                    first_sub = substitution_info_list[0]
                    event_dict['player_in'] = first_sub.get('player_in')
                    event_dict['player_out'] = first_sub.get('player_out')
                    if first_sub.get('team'):
                        event_dict['team'] = first_sub.get('team')
                
                # Add extracted score if available (for timeline building)
                if extracted_score:
                    event_dict['extracted_score'] = extracted_score
                
                # Add current score state for goals (like Gemini's score_at_event)
                if is_goal and (current_home > 0 or current_away > 0):
                    event_dict['score_at_event'] = (current_home, current_away)
                
                ticker_events.append(event_dict)
            
            # After processing all events, set actual_played from max minutes
            if max_minute_45 is not None:
                targets['actual_played_45'] = max_minute_45
            if max_minute_90 is not None:
                targets['actual_played_90'] = max_minute_90
            
            # Set minute to empty for pre/post match events and classify match phases
            # Track positions of key events (exact text matches only)
            anpfiff_index = None
            halbzeitpfiff_index = None
            anpfiff_2nd_index = None
            abpfiff_index = None
            
            # First pass: find exact positions of key events
            for idx, event in enumerate(ticker_events):
                event_text = event.get('text', '').strip()
                event_text_lower = event_text.lower().strip()
                
                # Exact matches only (case-insensitive)
                if event_text_lower == "anpfiff":
                    anpfiff_index = idx
                elif event_text_lower == "halbzeitpfiff":
                    halbzeitpfiff_index = idx
                elif event_text_lower == "anpfiff 2. hälfte" or event_text_lower == "anpfiff 2. halbzeit":
                    anpfiff_2nd_index = idx
                elif event_text_lower == "abpfiff" or event_text_lower == "spielende":
                    abpfiff_index = idx
                    break  # Stop after finding Abpfiff
            
            # Second pass: set minute to empty for pre/post match events
            # No event_type field - we just need to clean up the minute field
            for idx, event in enumerate(ticker_events):
                # Pre-match: everything until and including "Anpfiff" (exact match)
                if anpfiff_index is not None and idx <= anpfiff_index:
                    # Set minute to empty for pre-match events
                    event['minute'] = ""
                
                # Post-match: after "Abpfiff" (exact match)
                elif abpfiff_index is not None and idx > abpfiff_index:
                    # Set minute to empty for post-match events
                    event['minute'] = ""
        
        except Exception as e:
            logger.error(f"Error parsing ticker from HTML: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Debug: If no ticker events found, dump HTML
        if not ticker_events:
            logger.warning("Parsing failed. Saving debug HTML...")
            debug_dir = Path(__file__).parent.parent.parent / "tests" / "artifacts"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / "debug_failed_ticker.html"
            try:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                logger.warning(f"Debug HTML saved to {debug_file}")
            except Exception as e:
                logger.error(f"Failed to save debug HTML: {e}")
        
        logger.info(f"Parsed {len(ticker_events)} ticker events with {len(score_timeline)} score entries")
        return (ticker_events, targets, score_timeline)
    
    def parse_spielinfo(self, html: str) -> Dict[str, Dict[str, Any]]:
        """
        Parse metadata and match stats from /spielinfo page.
        
        Extracts:
        1. Metadata: Kick-off date/time, Stadium, Spectators, Referee
        2. Match Stats: Bar charts with stats like xGoals, Shots, Possession, Pass accuracy
        
        Args:
            html: HTML content of /spielinfo page
            
        Returns:
            Dictionary with keys 'info' (metadata) and 'stats' (match statistics)
        """
        soup = BeautifulSoup(html, 'html.parser')
        metadata = {}
        match_stats = {}
        
        try:
            # 1. Extract Metadata
            
            # Kick-off: Find div.kick__gameinfo-block containing span.kick__icon-Anstoss
            try:
                anstoss_icon = soup.find('span', class_=re.compile(r'kick__icon-Anstoss', re.I))
                if anstoss_icon:
                    gameinfo_block = anstoss_icon.find_parent('div', class_=re.compile(r'kick__gameinfo-block', re.I))
                    if gameinfo_block:
                        # Extract text following weekday badge (e.g., '12.01.2024, 20:30')
                        text = gameinfo_block.get_text(strip=True)
                        # Remove weekday and clean up
                        date_match = re.search(r'(\d{2}\.\d{2}\.\d{4}),?\s*(\d{2}:\d{2})', text)
                        if date_match:
                            metadata['kickoff_date'] = date_match.group(1)
                            metadata['kickoff_time'] = date_match.group(2)
                            metadata['kickoff'] = f"{date_match.group(1)}, {date_match.group(2)}"
            except Exception as e:
                logger.debug(f"Error extracting kick-off: {e}")
            
            # Stadium: Find span.kick__icon-Spielstaette
            try:
                stadium_icon = soup.find('span', class_=re.compile(r'kick__icon-Spielstaette', re.I))
                if stadium_icon:
                    gameinfo_block = stadium_icon.find_parent('div', class_=re.compile(r'kick__gameinfo-block', re.I))
                    if gameinfo_block:
                        # Extract stadium name from <a> tag
                        stadium_link = gameinfo_block.find('a')
                        if stadium_link:
                            metadata['stadium'] = stadium_link.get_text(strip=True)
                            # Extract city (text following the link)
                            full_text = gameinfo_block.get_text(strip=True)
                            stadium_name = stadium_link.get_text(strip=True)
                            city = full_text.replace(stadium_name, '').strip()
                            if city:
                                metadata['stadium_city'] = city
            except Exception as e:
                logger.debug(f"Error extracting stadium: {e}")
            
            # Spectators: Find span.kick__icon-Zuschauer
            try:
                zuschauer_icon = soup.find('span', class_=re.compile(r'kick__icon-Zuschauer', re.I))
                if zuschauer_icon:
                    gameinfo_block = zuschauer_icon.find_parent('div', class_=re.compile(r'kick__gameinfo-block', re.I))
                    if gameinfo_block:
                        text = gameinfo_block.get_text(strip=True)
                        # Remove '(ausverkauft)' if present
                        text = re.sub(r'\(ausverkauft\)', '', text, flags=re.I).strip()
                        # Extract number (e.g., '75.000')
                        number_match = re.search(r'(\d+\.?\d*)', text)
                        if number_match:
                            metadata['spectators'] = self._parse_german_number(number_match.group(1))
                            metadata['attendance'] = metadata['spectators']  # Backward compatibility
                        # Check for sold out
                        if 'ausverkauft' in gameinfo_block.get_text().lower():
                            metadata['is_sold_out'] = True
            except Exception as e:
                logger.debug(f"Error extracting spectators: {e}")
            
            # Referee: Look for section with header "Schiedsrichter-Team"
            try:
                referee_header = soup.find(string=re.compile(r'Schiedsrichter-Team', re.I))
                if referee_header:
                    referee_section = referee_header.find_parent()
                    if referee_section:
                        # Extract name from strong.kick__gameinfo__person
                        referee_name_elem = referee_section.find('strong', class_=re.compile(r'kick__gameinfo__person', re.I))
                        if referee_name_elem:
                            metadata['referee'] = referee_name_elem.get_text(strip=True)
            except Exception as e:
                logger.debug(f"Error extracting referee: {e}")
            
            # 2. Extract Match Stats (Bar Charts)
            try:
                stats_bars = soup.find_all(class_=re.compile(r'kick__stats-bar', re.I))
                for bar in stats_bars:
                    try:
                        # Extract title
                        title_elem = bar.find(class_=re.compile(r'kick__stats-bar__title', re.I))
                        if not title_elem:
                            continue
                        
                        title = title_elem.get_text(strip=True)
                        
                        # Extract home value
                        home_elem = bar.find(class_=re.compile(r'kick__stats-bar__value--opponent1', re.I))
                        home_value = home_elem.get_text(strip=True) if home_elem else None
                        
                        # Extract away value
                        away_elem = bar.find(class_=re.compile(r'kick__stats-bar__value--opponent2', re.I))
                        away_value = away_elem.get_text(strip=True) if away_elem else None
                        
                        if title and (home_value or away_value):
                            # Convert German titles to snake_case where possible
                            title_key = self._normalize_stat_title(title)
                            match_stats[title_key] = {
                                'title': title,
                                'home': self._parse_stat_value(home_value),
                                'away': self._parse_stat_value(away_value)
                            }
                    except Exception as e:
                        logger.debug(f"Error parsing stat bar: {e}")
                        continue
            except Exception as e:
                logger.debug(f"Error extracting match stats: {e}")
        
        except Exception as e:
            logger.warning(f"Error parsing spielinfo: {e}")
        
        logger.info(f"Parsed spielinfo: {len(metadata)} metadata fields, {len(match_stats)} stats")
        return {'info': metadata, 'stats': match_stats}
    
    def _normalize_stat_title(self, title: str) -> str:
        """
        Convert German stat titles to normalized keys.
        
        Args:
            title: German stat title (e.g., 'Ballbesitz', 'Torschüsse')
            
        Returns:
            Normalized key (e.g., 'possession', 'shots')
        """
        # Mapping of German titles to normalized keys
        title_mapping = {
            'ballbesitz': 'possession',
            'torschüsse': 'shots',
            'angekommene pässe': 'pass_accuracy',
            'xgoals': 'xgoals',
            'xgoals': 'xgoals',
            'torschüsse aufs tor': 'shots_on_target',
            'fouls': 'fouls',
            'ecken': 'corners',
            'abseits': 'offside',
        }
        
        title_lower = title.lower().strip()
        
        # Check if we have a direct mapping
        if title_lower in title_mapping:
            return title_mapping[title_lower]
        
        # Otherwise, convert to snake_case and return
        # Replace spaces and special chars with underscores
        normalized = re.sub(r'[^\w\s]', '', title_lower)
        normalized = re.sub(r'\s+', '_', normalized)
        return normalized
    
    def _parse_stat_value(self, value: str) -> Optional[Union[int, float]]:
        """
        Parse a stat value string to number.
        
        Args:
            value: Stat value string (e.g., '61%', '27', '2.77')
            
        Returns:
            Parsed number (int or float) or None
        """
        if not value:
            return None
        
        # Remove % sign and whitespace
        value = value.replace('%', '').strip()
        
        # Try to parse as float first (handles decimals)
        try:
            # Replace comma with dot for German format
            value = value.replace(',', '.')
            num_value = float(value)
            # Return as int if it's a whole number
            if num_value.is_integer():
                return int(num_value)
            return num_value
        except (ValueError, AttributeError):
            return None
    
    def merge_score_state(
        self, 
        ticker_events: List[Dict[str, Any]], 
        goal_lookup: Dict[int, Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """
        Merge score states into ticker events using goal lookup from ticker-extracted scores.
        
        Processes events chronologically and only updates score state when encountering a goal.
        This ensures events at the same minute that occur BEFORE a goal don't get the goal's score.
        
        Args:
            ticker_events: List of ticker event dictionaries
            goal_lookup: Dictionary mapping minute -> (home_score, away_score)
            
        Returns:
            List of ticker events with added score fields
        """
        # Sort events chronologically (by minute, then by original order)
        # Add index to preserve original order for events at the same minute
        indexed_events = [(i, event) for i, event in enumerate(ticker_events)]
        
        def sort_key(item):
            idx, event = item
            minute_str = event.get('minute', '')
            minute_int = self._parse_minute_to_int(minute_str)
            # Empty minutes (pre/post match) sort to beginning
            if minute_int is None:
                return (-1, idx)
            return (minute_int, idx)
        
        sorted_events = sorted(indexed_events, key=sort_key)
        
        # Track current score state as we process events chronologically
        current_home = 0
        current_away = 0
        enriched_events = []
        
        for idx, event in sorted_events:
            minute_str = event.get('minute', '')
            minute_int = self._parse_minute_to_int(minute_str)
            
            # Check if this event is a goal
            # Goals have score_at_event (set when we process them in parse_ticker)
            # This is the definitive marker for goal events
            is_goal = 'score_at_event' in event
            
            # If this is a goal event, update the current score state
            if is_goal:
                # Priority: score_at_event > extracted_score > goal_lookup
                if 'score_at_event' in event:
                    current_home, current_away = event['score_at_event']
                elif 'extracted_score' in event:
                    current_home, current_away = event['extracted_score']
                elif minute_int is not None and minute_int in goal_lookup:
                    current_home, current_away = goal_lookup[minute_int]
            
            # For non-goal events, use the most recent score from goal_lookup
            # but only if it's from a minute BEFORE this event's minute
            if not is_goal and minute_int is not None and goal_lookup:
                # Find goals that occurred BEFORE this minute (not at the same minute)
                relevant_minutes = [m for m in goal_lookup.keys() if m < minute_int]
                if relevant_minutes:
                    latest_minute = max(relevant_minutes)
                    current_home, current_away = goal_lookup[latest_minute]
                elif minute_int in goal_lookup:
                    # If there's a goal at this exact minute, don't use it for non-goal events
                    # They should have the score from before this minute
                    pass
            
            event_copy = event.copy()
            # Remove extracted_score from output (it was just for building timeline)
            event_copy.pop('extracted_score', None)
            
            # Assign current score state to this event
            if minute_int is not None:  # Only assign scores to events with valid minutes
                event_copy['home_score'] = current_home
                event_copy['away_score'] = current_away
                event_copy['goal_difference'] = current_home - current_away
                event_copy['is_home_leading'] = current_home > current_away
            else:
                # Pre/post match events don't have scores
                event_copy['home_score'] = None
                event_copy['away_score'] = None
                event_copy['goal_difference'] = None
                event_copy['is_home_leading'] = None
            
            enriched_events.append((idx, event_copy))
        
        # Restore original order
        enriched_events.sort(key=lambda x: x[0])
        return [event for _, event in enriched_events]
    
    def scrape_full_match(
        self, 
        match_url: str, 
        match_id: Optional[str] = None,
        season: Optional[str] = None,
        matchday: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Scrape a full match by visiting Spielinfo and Ticker tabs sequentially using Selenium.
        
        Extracts ticker events directly from server-side rendered HTML.
        
        Args:
            match_url: Base match URL
            match_id: Optional match ID (extracted from URL if not provided)
            season: Optional season string
            matchday: Optional matchday number
            
        Returns:
            Dictionary with match data or None if scraping fails
        """
        # Extract match_id from URL if not provided
        if not match_id:
            clean_url = match_url.rstrip('/')
            clean_url = re.sub(r'/(ticker|liveticker|spielinfo)/?$', '', clean_url)
            match = re.search(r'/([^/]+)/?$', clean_url)
            if match:
                match_id = match.group(1)
            else:
                match_id = "unknown"
        
        logger.info(f"Scraping match {match_id} from {match_url}")
        
        # Get base URL
        base_url = match_url.split('/ticker')[0].split('/liveticker')[0].split('/spielinfo')[0]
        
        # Step 1: Spielinfo Tab (Stats/Metadata)
        info_url = f"{base_url}/spielinfo"
        logger.debug(f"Fetching spielinfo tab: {info_url}")
        info_html = None
        match_info = {'info': {}, 'stats': {}}
        
        try:
            # Validate window is still open and reinitialize if needed
            try:
                _ = self.driver.current_window_handle
            except Exception as e:
                logger.warning(f"Browser window check failed: {e}. Reinitializing driver...")
                try:
                    if hasattr(self, 'driver') and self.driver:
                        self.driver.quit()
                except Exception as quit_error:
                    logger.debug(f"Error quitting old driver: {quit_error}")
                # Reinitialize driver
                try:
                    options = uc.ChromeOptions()
                    options.add_argument('--no-sandbox')
                    options.add_argument('--disable-dev-shm-usage')
                    options.add_argument('--disable-blink-features=AutomationControlled')
                    options.add_argument('--window-size=1920,1080')
                    self.driver = uc.Chrome(options=options)
                    self.wait = WebDriverWait(self.driver, timeout=10)
                    self._cookie_consent_handled = False  # Reset consent flag
                    logger.info("Driver reinitialized successfully")
                except Exception as init_error:
                    logger.error(f"Failed to reinitialize driver: {init_error}")
                    return None
            
            self.driver.get(info_url)
            self._random_delay(2, 3)
            self._handle_cookie_consent()
            
            # Scroll to ensure stats bars load (they often animate/load on scroll)
            self._scroll_page()
            
            info_html = self.driver.page_source
        except Exception as e:
            logger.error(f"Failed to fetch spielinfo tab: {e}")
            info_html = None
        
        # Parse spielinfo data (OPTIONAL - don't fail if it's missing)
        match_info = {'info': {}, 'stats': {}}
        metadata = {}  # Initialize metadata dict
        
        if info_html and self._validate_match_page(info_html, "info"):
            match_info = self.parse_spielinfo(info_html)
            logger.info(f"Successfully parsed spielinfo: {len(match_info['info'])} metadata fields, {len(match_info['stats'])} stats")
            # Initialize metadata from match_info
            metadata = match_info.get('info', {}).copy()
        else:
            logger.warning(f"Spielinfo tab unavailable for {match_id}. Using fallback metadata parsing.")
            # Fallback to old parse_metadata method
            if info_html:
                metadata = self.parse_metadata(info_html)
                match_info['info'] = metadata
        
        self._random_delay(2, 4)
        
        # Step 2: Ticker Tab (CRITICAL - must succeed)
        ticker_url = f"{base_url}/ticker"  # Use /ticker instead of /liveticker
        logger.debug(f"Fetching ticker tab: {ticker_url}")
        try:
            # Validate window is still open and reinitialize if needed
            try:
                _ = self.driver.current_window_handle
            except Exception as e:
                logger.warning(f"Browser window check failed: {e}. Reinitializing driver...")
                try:
                    if hasattr(self, 'driver') and self.driver:
                        self.driver.quit()
                except Exception as quit_error:
                    logger.debug(f"Error quitting old driver: {quit_error}")
                # Reinitialize driver
                try:
                    options = uc.ChromeOptions()
                    options.add_argument('--no-sandbox')
                    options.add_argument('--disable-dev-shm-usage')
                    options.add_argument('--disable-blink-features=AutomationControlled')
                    options.add_argument('--window-size=1920,1080')
                    self.driver = uc.Chrome(options=options)
                    self.wait = WebDriverWait(self.driver, timeout=10)
                    self._cookie_consent_handled = False  # Reset consent flag
                    logger.info("Driver reinitialized successfully")
                except Exception as init_error:
                    logger.error(f"Failed to reinitialize driver: {init_error}")
                    return None
            
            self.driver.get(ticker_url)
            self._random_delay(2, 3)
            self._handle_cookie_consent()
            
            # Wait for any ticker item to load (using CLASS_NAME like Gemini - more reliable)
            try:
                self.wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "kick__ticker-item"))
                )
                logger.debug("Ticker items found")
            except Exception as e:
                logger.warning(f"Timeout waiting for ticker items: {e}. Continuing anyway...")
            
            # Scroll page to trigger lazy-loading (using dedicated method)
            self._scroll_page()
            
            ticker_html = self.driver.page_source
        except Exception as e:
            logger.error(f"Failed to fetch ticker tab: {e}")
            return None
        
        if not self._validate_match_page(ticker_html, "ticker"):
            logger.error(f"Ticker page validation failed for {match_id}")
            return None
        
        # Parse ticker from HTML (server-side rendered, no API needed)
        ticker_events, targets, ticker_score_timeline = self.parse_ticker(ticker_html)
        
        # If ticker parsing fails, return None (no fallback)
        if not ticker_events:
            logger.error(f"No ticker events found for {match_id}. Cannot proceed without ticker data.")
            return None
        
        # Use ticker-extracted score timeline (primary and only source)
        if len(ticker_score_timeline) > 1:  # Ticker has scores
            logger.info(f"Using ticker-extracted score timeline with {len(ticker_score_timeline)} entries")
            final_goal_lookup = ticker_score_timeline
        else:
            logger.warning("No score timeline available from ticker")
            final_goal_lookup = {0: (0, 0)}
        
        # Merge score states
        ticker_data = self.merge_score_state(ticker_events, final_goal_lookup)
        
        # Extract additional metadata
        home_team = None
        away_team = None
        final_score = None
        
        # Try to extract team names from HTML title/meta tags first
        if ticker_html:
            soup = BeautifulSoup(ticker_html, 'html.parser')
            
            # Method 1: Extract from title tag (e.g., "Bayern M&#252;nchen - TSG Hoffenheim")
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text()
                # Pattern: "Team1 - Team2" or "Team1 vs Team2"
                title_match = re.search(r'([^-|]+?)\s*[-|]\s*([^-|]+?)\s*\d+:\d+', title_text)
                if title_match:
                    home_team = title_match.group(1).strip()
                    away_team = title_match.group(2).strip()
                    # Decode HTML entities (e.g., &#252; -> ü)
                    home_team = BeautifulSoup(home_team, 'html.parser').get_text()
                    away_team = BeautifulSoup(away_team, 'html.parser').get_text()
            
            # Method 2: Try to find team names from scoreboard or match header (fallback)
            if not home_team or not away_team:
                scoreboard = soup.select_one('.kick__scoreboard, .kick__v100-scoreBoard')
                if scoreboard:
                    # Find team name elements in scoreboard
                    team_links = scoreboard.find_all('a', href=re.compile(r'/info/bundesliga'))
                    if len(team_links) >= 2:
                        home_team = team_links[0].get_text(strip=True)
                        away_team = team_links[1].get_text(strip=True)
                    else:
                        # Fallback: look for team name classes
                        team_elems = scoreboard.find_all(class_=re.compile(r'kick__.*team.*name|kick__team', re.I))
            if len(team_elems) >= 2:
                            home_team = team_elems[0].get_text(strip=True).split('\n')[0].strip()
                            away_team = team_elems[1].get_text(strip=True).split('\n')[0].strip()
                
            # If still not found, try match header
            if not home_team or not away_team:
                header = soup.select_one('.kick__main-head, .kick__match-header')
                if header:
                    team_links = header.find_all('a', href=re.compile(r'/info/bundesliga'))
                    if len(team_links) >= 2:
                        home_team = team_links[0].get_text(strip=True)
                        away_team = team_links[1].get_text(strip=True)
        
        if final_goal_lookup and len(final_goal_lookup) > 1:
            last_minute = max(final_goal_lookup.keys())
            home_final, away_final = final_goal_lookup[last_minute]
            final_score = f"{home_final}:{away_final}"
        
        # Add season/matchday to metadata
        if season:
            metadata['season'] = season
        if matchday:
            metadata['matchday'] = matchday
        if home_team:
            metadata['home_team'] = home_team
        if away_team:
            metadata['away_team'] = away_team
        if final_score:
            metadata['final_score'] = final_score
        
        # Merge metadata from match_info with existing metadata
        if match_info.get('info'):
            # Update metadata with spielinfo data (spielinfo takes precedence)
            for key, value in match_info['info'].items():
                if value is not None:  # Only update if value exists
                    metadata[key] = value
        
        # Remove minute 0 from score_timeline (always 0:0, not needed)
        score_timeline_output = {str(k): list(v) for k, v in final_goal_lookup.items() if k != 0}
        
        # Keep both spectators and attendance fields (always)
        # They may differ if stadium is not sold out (attendance vs capacity)
        
        # Construct output with new structure
        output = {
            'match_id': match_id,
            'metadata': metadata,
            'targets': targets,
            'score_timeline': score_timeline_output,  # Excludes minute 0
            'match_info': match_info,  # Spielinfo: metadata + stats
            'ticker_data': ticker_data  # Ticker events (commentary)
        }
        
        # Save to file
        if season:
            season_dir = self.save_dir / f"season_{season}"
            season_dir.mkdir(parents=True, exist_ok=True)
            output_file = season_dir / f"match_{match_id}.json"
        else:
            output_file = self.save_dir / f"match_{match_id}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved match data to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save match data: {e}")
            raise
        
        return output


if __name__ == "__main__":
    """
    Main execution block: Scrape all Bundesliga matches from 2017-18 to 2024-25.
    
    For testing, use tests/test_scraper_live.py instead:
        python -m tests.test_scraper_live --test-known
        python -m tests.test_scraper_live --test-bayern
        python -m tests.test_scraper_live --test-bremen
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize scraper
    save_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    scraper = KickerScraper(save_dir=save_dir, headless=False)
    
    # Target seasons (VAR floor: 2017-18 onwards)
    seasons = [
        "2017-18", "2018-19",  # Pre-Corona, Clean
        "2019-20", "2020-21", "2021-22",  # Corona/Ghost Games (flagged but kept)
        "2022-23", "2023-24",  # Post-Corona, Clean
        "2024-25"  # Placebo Test Season
    ]
    
    # Statistics
    total_matches = 0
    successful_matches = 0
    failed_matches = 0
    
    try:
        # Iterate through seasons
        for season in seasons:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing season: {season}")
            logger.info(f"{'='*60}")
            
            # Iterate through matchdays (1-34 for Bundesliga)
            for matchday in range(1, 35):
                logger.info(f"\nSeason {season}, Matchday {matchday}")
                
                # Get match URLs
                match_urls = scraper.get_match_urls(season, matchday)
                
                if not match_urls:
                    logger.warning(f"No matches found for {season}, matchday {matchday}")
                    continue
                
                # Scrape each match
                for match_url in match_urls:
                    total_matches += 1
                    try:
                        result = scraper.scrape_full_match(
                            match_url, 
                            season=season, 
                            matchday=matchday
                        )
                        
                        if result:
                            successful_matches += 1
                            logger.info(f"✓ Successfully scraped match {result['match_id']}")
                        else:
                            failed_matches += 1
                            logger.warning(f"✗ Failed to scrape match from {match_url}")
                        
                        # Delay between matches
                        scraper._random_delay(2, 4)
                        
                    except Exception as e:
                        failed_matches += 1
                        logger.error(f"✗ Error scraping match from {match_url}: {e}")
                        continue
    
    except Exception as e:
        logger.error(f"Error in scraping loop: {e}")
    finally:
        # Ensure driver is closed
        if scraper and scraper.driver:
            scraper.driver.quit()
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SCRAPING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total matches attempted: {total_matches}")
    logger.info(f"Successful: {successful_matches}")
    logger.info(f"Failed: {failed_matches}")
    logger.info(f"Success rate: {successful_matches/total_matches*100:.1f}%" if total_matches > 0 else "N/A")
