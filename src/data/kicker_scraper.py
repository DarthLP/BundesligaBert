"""
KickerScraper: Multi-tab scraper for Bundesliga match data from Kicker.de

This module implements a robust scraper using standard Selenium with webdriver_manager to handle
cookie walls and JavaScript-rendered content. It visits two tabs (Spielinfo, Ticker)
sequentially to extract match metadata and ticker text with precise score states at every minute.
It strictly separates input features from target labels by extracting announced and actual played
time before removing leakage from text.

The scraper uses a Context Manager pattern for guaranteed driver cleanup, preventing zombie processes.
It includes anti-ban measures:
- Runs Chrome in incognito mode and background (headless)
- Disables image loading to save memory
- Clears cookies between requests
- Uses random user agents
- Implements longer delays between requests
- Skips already downloaded matches to avoid re-scraping

Usage:
    # Option 1: Run simultaneous season download (RECOMMENDED)
    # Scrapes all seasons (2024-25 to 2017-18, latest first) in parallel using multiprocessing
    # Each season runs in a separate process with its own Chrome instance
    # This is the fastest and safest way to download all data
    python src/data/kicker_scraper.py
    
    # Option 2: Use as a module for individual matches (Context Manager pattern)
    from src.data.kicker_scraper import KickerScraper
    from pathlib import Path
    
    with KickerScraper(save_dir=Path("data/raw")) as scraper:
        result = scraper.scrape_full_match(
            match_url="https://www.kicker.de/...",
            season="2023-24",
            matchday=1
        )
    
    # Option 3: Get match URLs for a specific matchday
    with KickerScraper(save_dir=Path("data/raw")) as scraper:
        match_urls = scraper.get_match_urls(season="2023-24", matchday=1)
        for url in match_urls:
            scraper.scrape_full_match(url, season="2023-24", matchday=1)
    
    # Option 4: Force re-scraping even if match already downloaded (for testing)
    with KickerScraper(save_dir=Path("data/raw")) as scraper:
        result = scraper.scrape_full_match(
            match_url="https://www.kicker.de/...",
            season="2023-24",
            matchday=1,
            force_rescrape=True
        )

Multiprocessing Details:
    When running as a script (Option 1), the scraper uses multiprocessing to run
    2 seasons per process (4 processes total for 8 seasons). Each process:
    - Has its own Chrome driver instance in incognito mode
    - Runs in background (headless mode)
    - Processes all matchdays (1-34) for its assigned seasons sequentially
    - Automatically skips matches that are already downloaded
    - Logs progress independently with process-specific prefixes
    - Continues even if individual matches or seasons fail
    
    This approach:
    - Reduces concurrent driver initializations (4 instead of 8)
    - Significantly speeds up scraping while maintaining stability
    - Reduces ban risk (each process has separate session/cookies)
    - Allows resuming interrupted downloads (skips existing files)
    - Provides per-process and per-season progress tracking
    - Better error recovery (one failure doesn't stop everything)

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
import multiprocessing
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


class KickerScraper:
    """
    Scraper for Bundesliga match data from Kicker.de using standard Selenium.
    
    Uses standard Selenium with webdriver_manager for automatic ChromeDriver management.
    Implements a multi-tab scraping strategy:
    1. Spielinfo tab (/spielinfo): Extract metadata (stadium, attendance, referee, stats)
    2. Ticker tab (/ticker): Extract ticker events from server-side rendered HTML
    
    This class implements the Context Manager pattern for guaranteed driver cleanup.
    Always use it with a 'with' statement to ensure proper resource management.
    
    Attributes:
        save_dir: Directory path to save scraped JSON files
        driver: Selenium WebDriver instance (standard Chrome, initialized in __enter__)
        wait: WebDriverWait instance for explicit waits (initialized in __enter__)
    """
    
    def __init__(self, save_dir: Union[str, Path], headless: bool = True) -> None:
        """
        Initialize KickerScraper (driver initialization happens in __enter__).
        
        Use this class as a context manager:
            with KickerScraper(save_dir="data/raw") as scraper:
                # scraping logic
        
        Args:
            save_dir: Directory to save scraped match JSON files
            headless: If True, run browser in headless mode (default: True for background operation)
        """
        # Reduce process priority on macOS to keep system responsive during scraping
        if os.name == 'posix':  # Unix-like (macOS, Linux)
            try:
                import psutil
                p = psutil.Process()
                p.nice(10)  # Lower priority (higher nice value) - keeps system responsive
                logger.debug("Reduced process priority for better system responsiveness")
            except (ImportError, AttributeError, PermissionError):
                # Ignore if psutil not available, nice() not supported, or permission denied
                pass
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        
        # Driver will be initialized in __enter__
        self.driver = None
        self.wait = None
        self._cookie_consent_handled = False
    
    def __enter__(self):
        """
        Context manager entry: Initialize Selenium WebDriver with file lock for multiprocessing safety.
        
        Returns:
            self: The KickerScraper instance
        """
        logger.info("Initializing Selenium WebDriver...")
        
        # File-based lock to prevent race conditions in multiprocessing
        # This ensures only one process initializes ChromeDriver at a time
        lock_file = Path.home() / ".chromedriver_init.lock"
        max_lock_wait = 180  # Maximum seconds to wait for lock
        lock_wait_interval = 0.5  # Check lock every 0.5 seconds
        stale_lock_threshold = 300  # Remove lock file if older than 5 minutes (stale)
        
        # Wait for lock to be released (if another process is initializing)
        lock_acquired = False
        start_time = time.time()
        while not lock_acquired and (time.time() - start_time) < max_lock_wait:
            try:
                # Check if lock file exists and is stale
                if lock_file.exists():
                    lock_age = time.time() - lock_file.stat().st_mtime
                    if lock_age > stale_lock_threshold:
                        # Lock is stale, remove it
                        try:
                            lock_file.unlink()
                            logger.debug("Removed stale lock file")
                        except Exception:
                            pass
                
                # Try to create lock file (exclusive)
                if not lock_file.exists():
                    try:
                        lock_file.touch(exist_ok=False)
                        lock_acquired = True
                        logger.debug("Acquired driver initialization lock")
                    except (FileExistsError, OSError):
                        # Another process created it just now, wait
                        time.sleep(lock_wait_interval)
                else:
                    # Lock file exists, wait for it to be released
                    time.sleep(lock_wait_interval)
            except Exception as e:
                logger.debug(f"Error checking lock: {e}")
                time.sleep(lock_wait_interval)
        
        if not lock_acquired:
            logger.error(f"Could not acquire driver initialization lock after {max_lock_wait}s")
            logger.error("This may indicate too many processes trying to initialize simultaneously")
            raise RuntimeError("Failed to acquire driver initialization lock - too many concurrent initializations")
        
        try:
            # Add small random delay even with lock to further stagger initialization
            time.sleep(random.uniform(0.1, 0.5))
            
            # Initialize driver with retry logic for network timeouts and conflicts
            max_retries = 5
            retry_delay = 3  # Start with longer delay
            for attempt in range(max_retries):
                try:
                    # Create Chrome options
                    options = self._create_chrome_options(headless=self.headless)
                    
                    # Use webdriver_manager to automatically handle ChromeDriver binary
                    service = Service(ChromeDriverManager().install())
                    
                    # Initialize standard Selenium WebDriver
                    self.driver = webdriver.Chrome(service=service, options=options)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        error_msg = str(e)
                        error_type = str(type(e))
                        
                        # Check for various error types
                        if "timeout" in error_msg.lower() or "URLError" in error_type:
                            logger.warning(f"Network timeout during driver initialization (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay = min(retry_delay * 1.5, 10)  # Exponential backoff, max 10s
                        elif "FileNotFoundError" in error_type and "chromedriver" in error_msg.lower():
                            # Race condition with chromedriver download - wait and retry
                            logger.warning(f"ChromeDriver file conflict (attempt {attempt + 1}/{max_retries}). Waiting {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay = min(retry_delay * 1.5, 10)
                        elif "WebDriverException" in error_type and "Can not connect to the Service" in error_msg:
                            # Driver service conflict - wait longer
                            logger.warning(f"ChromeDriver service conflict (attempt {attempt + 1}/{max_retries}). Waiting {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay = min(retry_delay * 1.5, 10)
                        else:
                            # For other errors, wait a bit and retry
                            logger.warning(f"Driver initialization error (attempt {attempt + 1}/{max_retries}): {error_type}. Waiting {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay = min(retry_delay * 1.5, 10)
                    else:
                        # Last attempt failed
                        logger.error(f"Failed to initialize WebDriver after {max_retries} attempts: {e}")
                        logger.error("This may be due to network connectivity issues or driver conflicts.")
                        raise RuntimeError(f"Failed to initialize Chrome WebDriver: {e}") from e
        finally:
            # Release lock
            if lock_acquired and lock_file.exists():
                try:
                    lock_file.unlink()
                    logger.debug("Released driver initialization lock")
                except Exception:
                    pass  # Ignore errors when removing lock file
        
        self.wait = WebDriverWait(self.driver, timeout=5)
        
        # Track if we've handled cookie consent (optimization)
        self._cookie_consent_handled = False
        
        # Clear cookies on initialization (incognito should handle this, but ensure it)
        self._clear_cookies()
        
        logger.info("WebDriver initialized successfully (incognito mode, background)")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit: Always cleanup driver, even on exceptions.
        
        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
            
        Returns:
            False: Don't suppress exceptions
        """
        if self.driver:
            try:
                self.driver.quit()
                logger.debug("WebDriver closed successfully")
            except Exception as e:
                logger.debug(f"Error closing WebDriver: {e}")
                # Ignore errors during cleanup
        return False  # Don't suppress exceptions
    
    def _create_chrome_options(self, headless: bool = True) -> ChromeOptions:
        """
        Create Chrome options with anti-ban measures and memory optimization.
        
        Args:
            headless: If True, run browser in headless mode
            
        Returns:
            Configured ChromeOptions object
        """
        options = ChromeOptions()
        
        # Anti-ban measures: Run in background (headless) and incognito mode
        if headless:
            options.add_argument('--headless=new')  # Use new headless mode
        options.add_argument('--incognito')  # Incognito mode to avoid cookie tracking
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--window-size=1920,1080')  # Force Desktop view
        
        # Additional anti-detection measures
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-software-rasterizer')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        
        # Memory optimization: Disable image loading to save RAM
        prefs = {
            "profile.managed_default_content_settings.images": 2  # 2 = block images
        }
        options.add_experimental_option("prefs", prefs)
        
        # Additional memory and CPU optimizations
        options.add_argument('--blink-settings=imagesEnabled=false')  # Disable images
        options.add_argument('--disable-javascript-harmony-shipping')  # Reduce JS overhead
        
        # Additional CPU reduction flags (safe to disable in headless mode)
        options.add_argument('--disable-background-timer-throttling')  # Reduce background processing
        options.add_argument('--disable-backgrounding-occluded-windows')  # Don't process hidden windows
        options.add_argument('--disable-renderer-backgrounding')  # Don't background renderer
        options.add_argument('--disable-background-networking')  # Disable background network requests
        options.add_argument('--disable-sync')  # Disable sync features
        options.add_argument('--disable-default-apps')  # Disable default apps
        options.add_argument('--disable-component-update')  # Disable component updates
        options.add_argument('--disable-background-downloads')  # Disable background downloads
        options.add_argument('--disable-domain-reliability')  # Disable domain reliability monitoring
        options.add_argument('--disable-breakpad')  # Disable crash reporting
        options.add_argument('--disable-features=TranslateUI')  # Disable translation features
        options.add_argument('--disable-features=AudioServiceOutOfProcess')  # Disable audio service
        options.add_argument('--disable-features=MediaRouter')  # Disable media router
        options.add_argument('--js-flags=--max-old-space-size=256')  # Limit JS heap size
        
        # Exclude enable-automation switch (anti-detection)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        
        # Random user agent for better anti-detection
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        ]
        options.add_argument(f'--user-agent={random.choice(user_agents)}')
        
        return options
    
    def _clear_cookies(self) -> None:
        """
        Clear all cookies from the browser to avoid tracking.
        
        This is called on initialization and can be called between requests
        to reset the session state.
        """
        try:
            self.driver.delete_all_cookies()
            logger.debug("Cookies cleared")
        except Exception as e:
            logger.warning(f"Failed to clear cookies: {e}")
    
    def _is_driver_alive(self) -> bool:
        """
        Check if the WebDriver is still responsive.
        
        Returns:
            True if driver is alive and responsive, False otherwise
        """
        try:
            # Try a simple operation to check if driver is responsive
            _ = self.driver.current_url
            return True
        except Exception:
            return False
    
    def _get_page_source_safe(self, timeout: int = 30) -> Optional[str]:
        """
        Safely get page source with timeout handling and error recovery.
        
        This method wraps driver.page_source with better error handling for:
        - urllib3 ReadTimeoutError (Chrome hung/unresponsive)
        - Selenium WebDriverException (connection issues)
        - Generic timeout errors
        
        Args:
            timeout: Maximum seconds to wait for page source (default: 30)
            
        Returns:
            Page source HTML string, or None if retrieval failed
        """
        try:
            # Check if driver is still alive before attempting to get page source
            if not self._is_driver_alive():
                logger.error("WebDriver is not responsive. Chrome may have crashed.")
                return None
            
            # Import urllib3 exceptions for specific handling
            from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError
            from requests.exceptions import ReadTimeout as RequestsReadTimeout
            from selenium.common.exceptions import TimeoutException, WebDriverException
            
            # Try to get page source with timeout protection
            html = self.driver.page_source
            return html
            
        except (Urllib3ReadTimeoutError, RequestsReadTimeout) as e:
            logger.error(f"ReadTimeoutError getting page source: {e}")
            logger.error("Chrome appears to be hung or unresponsive. This may require driver restart.")
            return None
            
        except (TimeoutException, WebDriverException) as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "read timed out" in error_msg:
                logger.error(f"WebDriver timeout getting page source: {e}")
                logger.error("Chrome connection timed out. This may require driver restart.")
            else:
                logger.error(f"WebDriverException getting page source: {e}")
            return None
            
        except Exception as e:
            error_msg = str(e).lower()
            # Check for timeout-related errors in generic exception
            if "timeout" in error_msg or "read timed out" in error_msg or "timed out" in error_msg:
                logger.error(f"Timeout error getting page source: {e}")
                logger.error("Chrome appears to be hung or unresponsive.")
            else:
                logger.error(f"Unexpected error getting page source: {e}")
            return None
    
    def _is_match_downloaded(self, match_id: str, season: Optional[str] = None) -> bool:
        """
        Check if a match has already been downloaded.
        
        Args:
            match_id: Match ID to check
            season: Optional season string (if provided, checks in season-specific directory)
            
        Returns:
            True if match file exists, False otherwise
        """
        if season:
            match_file = self.save_dir / f"season_{season}" / f"match_{match_id}.json"
        else:
            match_file = self.save_dir / f"match_{match_id}.json"
        
        return match_file.exists()
    
    def _extract_match_id_from_url(self, match_url: str) -> str:
        """
        Extract match ID from a match URL.
        
        Args:
            match_url: Match URL (e.g., "https://www.kicker.de/bayern-gegen-hoffenheim-2024-bundesliga-4862110/ticker")
            
        Returns:
            Match ID (e.g., "bayern-gegen-hoffenheim-2024-bundesliga-4862110")
        """
        clean_url = match_url.rstrip('/')
        clean_url = re.sub(r'/(ticker|liveticker|spielinfo|analyse|spielbericht)/?$', '', clean_url)
        match = re.search(r'/([^/]+)/?$', clean_url)
        if match:
            return match.group(1)
        # Fallback: try splitting
        parts = clean_url.split('/')
        if parts:
            return parts[-1].split('?')[0]
        return "unknown"
    
    def _get_matchday_metadata_path(self, season: str, matchday: int) -> Path:
        """
        Get path to matchday metadata file.
        
        Args:
            season: Season string (e.g., "2023-24")
            matchday: Matchday number (1-34)
            
        Returns:
            Path to metadata file
        """
        season_dir = self.save_dir / f"season_{season}"
        return season_dir / f"matchday_{matchday}_metadata.json"
    
    def _is_matchday_complete(self, season: str, matchday: int) -> bool:
        """
        Check if a matchday is already completely scraped.
        
        Checks if metadata file exists and all match IDs in it have corresponding JSON files.
        
        Args:
            season: Season string (e.g., "2023-24")
            matchday: Matchday number (1-34)
            
        Returns:
            True if matchday is complete, False otherwise
        """
        metadata_path = self._get_matchday_metadata_path(season, matchday)
        
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            match_ids = metadata.get('match_ids', [])
            if not match_ids:
                return False
            
            # Check if all matches exist
            for match_id in match_ids:
                if not self._is_match_downloaded(match_id, season):
                    return False
            
            return True
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.debug(f"Error reading matchday metadata for {season}, matchday {matchday}: {e}")
            return False
    
    def _save_matchday_metadata(self, season: str, matchday: int, match_urls: List[str]) -> None:
        """
        Save metadata for a matchday after successful scraping.
        
        Args:
            season: Season string (e.g., "2023-24")
            matchday: Matchday number (1-34)
            match_urls: List of match URLs that were scraped
        """
        metadata_path = self._get_matchday_metadata_path(season, matchday)
        season_dir = metadata_path.parent
        season_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract match IDs from URLs
        match_ids = [self._extract_match_id_from_url(url) for url in match_urls]
        
        metadata = {
            'season': season,
            'matchday': matchday,
            'match_ids': match_ids,
            'match_urls': match_urls,
            'total_matches': len(match_ids),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved matchday metadata for {season}, matchday {matchday}")
        except IOError as e:
            logger.warning(f"Failed to save matchday metadata for {season}, matchday {matchday}: {e}")
    
    def _random_delay(self, min_sec: float = 1, max_sec: float = 2) -> None:
        """
        Sleep for a random duration between min_sec and max_sec.
        
        Optimized defaults for M1 Mac: reduced from 2-4s to 1-2s for better performance.
        Can still be overridden with custom values when needed.
        
        Args:
            min_sec: Minimum delay in seconds (default: 1)
            max_sec: Maximum delay in seconds (default: 2)
        """
        delay = random.uniform(min_sec, max_sec)
        time.sleep(delay)
    
    def _scroll_page(self) -> None:
        """
        Scroll the page to trigger lazy loaders (CPU-optimized version).
        
        The Kicker ticker uses lazy loading, so elements don't exist in the DOM
        until they are scrolled into view. This function performs a minimal scroll
        directly to bottom (single scroll) to reduce CPU usage.
        
        Optimized for M1 Mac: single scroll to bottom instead of multiple scrolls.
        """
        logger.debug("Scrolling page to trigger lazy loaders...")
        try:
            # Single scroll to bottom - most efficient for CPU
            # Most content loads when scrolling to bottom, so we skip intermediate positions
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.3)  # Minimal wait for lazy loading (reduced from 0.5s)
            
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
                button_wait = WebDriverWait(self.driver, timeout=5)  # Reduced from 10s to 5s
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
        
        Simple and robust approach:
        1. Find all numbers (digits and German words) in the text
        2. If multiple numbers found, prefer the one directly before "minuten"
        3. If only one number found, use that one
        4. Always prefer "zwei" and above over "eins", "eine", etc. (to avoid false positives)
        
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
        
        # Check if this is a stoppage time announcement (must contain context phrases)
        # Simple check: context phrase + "minuten"/"minute" + number
        context_phrases = [
            'nachspielzeit', 'nachgespielt', 'nachgelegt', 'obendrauf', 
            'oben drauf', 'draufgepackt', 'drauf gepackt', 'es gibt', 'gibt es', 
            'nachschlag', 'zugabe', 'bonus', 'extra', 'tafel zeigt', 
            'auf der tafel', 'vierter offizielle', 'werden noch', 'wird noch'
        ]
        has_context = any(phrase in text_lower for phrase in context_phrases)
        has_minuten = 'minuten' in text_lower or 'minute' in text_lower
        
        if not (has_context and has_minuten):
            return None
        
        # Find all numbers in the text (both digits and German words)
        found_numbers = []
        
        # Find German number words
        for word, value in german_numbers.items():
            # Use word boundaries to avoid partial matches
            pattern = rf'\b{word}\b'
            matches = list(re.finditer(pattern, text_lower))
            for match in matches:
                found_numbers.append({
                    'value': value,
                    'position': match.start(),
                    'text': match.group(),
                    'is_german_word': True
                })
        
        # Find digit numbers
        digit_pattern = r'\b(\d+)\b'
        digit_matches = list(re.finditer(digit_pattern, text_lower))
        for match in digit_matches:
            try:
                value = int(match.group(1))
                # Only consider reasonable values (1-15 minutes is typical)
                if 1 <= value <= 15:
                    found_numbers.append({
                        'value': value,
                        'position': match.start(),
                        'text': match.group(1),
                        'is_german_word': False
                    })
            except ValueError:
                continue
        
        if not found_numbers:
            return None
        
        # If multiple numbers, prefer the one EXACTLY before "minuten" or "min."
        if len(found_numbers) > 1:
            # Look for "minuten" or "min." in the text
            minuten_pattern = r'\bminuten?\b|\bmin\.?\b'
            minuten_matches = list(re.finditer(minuten_pattern, text_lower))
            
            if not minuten_matches:
                # No "minuten" found, fall through to single number or higher value logic
                pass
            else:
                # For each "minuten" match, find the number that is CLOSEST to it (immediately before)
                numbers_before_minuten = []
                for min_match in minuten_matches:
                    min_pos = min_match.start()
                    # Find the number that appears immediately before this "minuten"
                    # (closest number that is before "minuten")
                    closest_num = None
                    closest_distance = float('inf')
                    
                    for num_info in found_numbers:
                        num_pos = num_info['position']
                        num_end = num_pos + len(num_info['text'])
                        
                        # Number must be BEFORE "minuten" (not after)
                        # This ensures we don't pick numbers that appear after "minuten"
                        # e.g., "drei minuten und zwei sekunden" -> should get 3, not 2
                        if num_pos < min_pos:
                            # Exclude numbers that are part of scores (like "3:2" or "2:1")
                            if num_end < len(text_lower):
                                next_chars = text_lower[num_end:num_end + 3].strip()
                                # If it starts with ':' or has ':' nearby, it's likely a score
                                if next_chars.startswith(':') or (':' in text_lower[max(0, num_pos - 2):num_end + 3]):
                                    continue  # Skip this number, it's part of a score
                            
                            # Calculate distance (how far before "minuten")
                            distance = min_pos - num_end
                            # Only consider numbers within reasonable distance (20 chars)
                            if distance <= 20 and distance < closest_distance:
                                closest_num = num_info
                                closest_distance = distance
                    
                    if closest_num:
                        numbers_before_minuten.append(closest_num)
                
                # If we found numbers exactly before "minuten", prefer higher values (>= 2) over "eins"/"eine"
                if numbers_before_minuten:
                    higher_values = [n for n in numbers_before_minuten if n['value'] >= 2]
                    if higher_values:
                        # Return the leftmost (first) higher value
                        return min(higher_values, key=lambda n: n['position'])['value']
                    else:
                        # Only "eins"/"eine" before "minuten", return it
                        return numbers_before_minuten[0]['value']
        
        # If only one number found, use it
        if len(found_numbers) == 1:
            return found_numbers[0]['value']
        
        # Multiple numbers but none before "minuten" - prefer higher values (>= 2)
        # Filter out "eins", "eine", "ein" (value 1) if there are higher values
        # This avoids false positives from "eine Minute" when there's a real time announcement
        higher_values = [n for n in found_numbers if n['value'] >= 2]
        if higher_values:
            # Return the leftmost (first) higher value
            return min(higher_values, key=lambda n: n['position'])['value']
        else:
            # Only "eins"/"eine" found, return it (but this is less reliable)
            return found_numbers[0]['value']
    
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
        
        # For card events, also try to extract player name from patterns like "der Bochumer sieht gelb"
        # where the team is mentioned before the player
        if is_card:
            # Pattern: "der [Team]er sieht gelb" -> extract player name that follows
            team_player_pattern = r'(?:der|die|den)\s+[A-ZÄÖÜ][a-zäöüß]+(?:er|ern|em|es|e)?\s+(?:wird|sieht|bekommt)\s+(?:gelb|rot|gelb-rot|von\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*))'
            team_player_match = re.search(team_player_pattern, text, re.IGNORECASE)
            if team_player_match and team_player_match.lastindex >= 1:
                return team_player_match.group(1).strip()
            
            # Pattern: "wird ... von [Player] umgeräumt" or "dafür sieht der [Team]er [Player]"
            # Example: "wird ... von Wittek umgeräumt. Dafür sieht der Bochumer Gelb"
            von_pattern = r'von\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+(?:umgeräumt|gefoult|getroffen)'
            von_match = re.search(von_pattern, text, re.IGNORECASE)
            if von_match:
                return von_match.group(1).strip()
        
        return None
    
    def _extract_team_from_text(self, text: str, home_team: Optional[str] = None, away_team: Optional[str] = None) -> Optional[str]:
        """
        Extract team name from ticker text.
        
        Looks for team names in various patterns:
        - "der Bochumer", "den Bremern" (adjective forms)
        - "Player (Team)" or "Team: Player"
        - Team names mentioned in context
        
        Args:
            text: Event text
            home_team: Optional home team name for validation (e.g., "Werder Bremen")
            away_team: Optional away team name for validation (e.g., "VfL Bochum")
            
        Returns:
            Team name string or None if not found
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Common false positives to exclude
        false_positives = {
            'aus', 'ein', 'für', 'sieht', 'bekommt', 'erhält', 'foul', 'karte', 
            'gelb', 'rot', 'gelb-rot', 'minute', 'minuten', 'spielerwechsel', 'wechsel',
            'gästen', 'gast', 'gäste', 'heimern', 'heim', 'heime'  # "bei den Gästen" means away team, "bei den Heimern" means home team
        }
        
        # Check for "bei den Gästen" (away team) or "bei den Heimern" (home team) first
        if 'bei den gästen' in text_lower or 'bei den gast' in text_lower:
            return away_team if away_team else None
        if 'bei den heimern' in text_lower or 'bei den heim' in text_lower:
            return home_team if home_team else None
        
        # Pattern 1: "der [Team]er", "den [Team]ern", "die [Team]er" (adjective forms)
        # Examples: "der Bochumer" (the Bochum player), "den Bremern" (the Bremen players)
        adjective_patterns = [
            r'(?:der|die|den|dem|des)\s+([A-ZÄÖÜ][a-zäöüß]+(?:er|ern|em|es|e)?)',  # der Bochumer, den Bremern
            r'([A-ZÄÖÜ][a-zäöüß]+(?:er|ern|em|es|e)?)\s+(?:Konter|Angriff|Spieler|Mannschaft)',  # Bochumer Konter
        ]
        
        for pattern in adjective_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                candidate = match.group(1).strip()
                candidate_lower = candidate.lower()
                
                # Skip if it's a false positive
                if candidate_lower in false_positives:
                    continue
                
                # Try to match against known team names if provided
                # Remove adjective endings to get the base (e.g., "Bremern" -> "Brem", "Bochumer" -> "Bochum")
                candidate_base = re.sub(r'(?:er|ern|em|es|e)$', '', candidate_lower)
                
                if home_team:
                    home_lower = home_team.lower()
                    # Check if candidate or its base matches home team
                    # "Bremern" base "Brem" should match "Werder Bremen" or "Bremen"
                    if (candidate_lower in home_lower or 
                        candidate_base in home_lower or
                        any(candidate_lower in part or candidate_base in part for part in home_lower.split()) or
                        any(part in candidate_lower or part in candidate_base for part in home_lower.split())):
                        return home_team
                
                if away_team:
                    away_lower = away_team.lower()
                    # Check if candidate or its base matches away team
                    if (candidate_lower in away_lower or 
                        candidate_base in away_lower or
                        any(candidate_lower in part or candidate_base in part for part in away_lower.split()) or
                        any(part in candidate_lower or part in candidate_base for part in away_lower.split())):
                        return away_team
                
                # If no team names provided, try to reconstruct team name from adjective
                # "Bochumer" -> "Bochum", "Bremer" -> "Bremen" (but this is less reliable)
                # For now, return the full team name if we have it, otherwise the candidate
                # But we need to be careful - let's prefer explicit mentions
        
        # Pattern 2: Team names in parentheses "Player (Team)"
        paren_pattern = r'\(([^)]+)\)'
        paren_matches = re.finditer(paren_pattern, text)
        for match in paren_matches:
            candidate = match.group(1).strip()
            candidate_lower = candidate.lower()
            
            # Skip false positives
            if candidate_lower in false_positives:
                continue
            
            # Check against known teams
            if home_team and candidate_lower in home_team.lower():
                return home_team
            if away_team and candidate_lower in away_team.lower():
                return away_team
            
            # If it looks like a team name (capitalized, reasonable length), return it
            if len(candidate) > 3 and candidate[0].isupper():
                return candidate
        
        # Pattern 3: "Team: Player" format
        colon_pattern = r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*):\s*[A-ZÄÖÜ]'
        colon_match = re.search(colon_pattern, text)
        if colon_match:
            candidate = colon_match.group(1).strip()
            candidate_lower = candidate.lower()
            
            if candidate_lower not in false_positives:
                # Check against known teams
                if home_team and candidate_lower in home_team.lower():
                    return home_team
                if away_team and candidate_lower in away_team.lower():
                    return away_team
                
                if len(candidate) > 3:
                    return candidate
        
        # Pattern 4: Look for explicit team name mentions in text
        # This is a fallback that looks for capitalized words that might be team names
        # But we're more conservative here to avoid false positives
        
        return None
    
    def _extract_substitution_from_text(self, text: str, home_team: Optional[str] = None, away_team: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Extract substitution information from ticker text.
        
        Looks for patterns like "Spielerwechsel: Müller für Müller" or "Wechsel: Player für Player".
        Handles multiple substitutions at the same time (e.g., "Wechsel: Player1 für Player2, Player3 für Player4").
        Handles special patterns like "ist [Player] für [Player] dabei" and "bei [Team] ersetzt [Player] [Player]".
        
        Args:
            text: Event text containing substitution information
            home_team: Optional home team name for validation
            away_team: Optional away team name for validation
            
        Returns:
            List of dictionaries with 'player_in', 'player_out', 'team' or None if not found
            Returns list to handle multiple subs at the same minute
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Check if text contains substitution keywords or narrative substitution descriptions
        has_substitution_keywords = (
            'wechsel' in text_lower or 
            'spielerwechsel' in text_lower or
            'geht vom feld' in text_lower or
            'kommt in die partie' in text_lower or
            re.search(r'\bwechselt\b', text_lower) is not None
        )
        
        if not has_substitution_keywords:
            return None
        
        # Exclude if this looks like a goal (contains "Tor für" or similar goal patterns)
        if any(phrase in text_lower for phrase in ['tor für', 'tor!', 'tore für']):
            return None
        
        substitutions = []
        
        # Track matched positions to avoid duplicate matches
        matched_positions = set()
        
        # Common team name prefixes that should be excluded from player matching
        # These patterns indicate a team name, not a player name
        team_prefixes = r'\b(?:bei|den|der|die|dem|des)\s+'
        
        # Pattern 1: "ist [Player] für [Player] dabei" (e.g., "ist Stage für Lynen dabei")
        # This pattern handles: "Bei den Bremern ist Stage für den bereits verwarnten Lynen dabei"
        ist_fuer_pattern = r'ist\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+für\s+(?:den\s+)?(?:bereits\s+verwarnten\s+)?([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+dabei'
        ist_fuer_matches = re.finditer(ist_fuer_pattern, text)
        for match in ist_fuer_matches:
            if match.lastindex >= 2:
                player_in = match.group(1).strip()
                player_out = match.group(2).strip()
                
                # Extract team from context (look for "Bei den Bremern", "den Bremern", "bei den Gästen", etc.)
                start_pos = max(0, match.start() - 200)
                context = text[start_pos:match.start()].lower()
                
                # Check for "bei den Gästen" (away team) or "bei den Heimern" (home team)
                if 'bei den gästen' in context or 'bei den gast' in context:
                    team = away_team
                elif 'bei den heimern' in context or 'bei den heim' in context:
                    team = home_team
                else:
                    # Look for team pattern like "Bei den Bremern" or "den Bremern"
                    # Use team extraction function with home/away team validation for better accuracy
                    team = self._extract_team_from_text(text[start_pos:match.start()], home_team=home_team, away_team=away_team)
                
                substitutions.append({
                    'player_in': player_in,
                    'player_out': player_out,
                    'team': team
                })
                # Track this match position
                matched_positions.add((match.start(), match.end()))
        
        # Pattern 2: "bei [Team] ersetzt [Player] [Player]" (e.g., "bei Bochum ersetzt Bero Osterhage")
        # We need to skip the team name and extract "Bero ersetzt Osterhage"
        # Capture the team name as well for validation
        bei_ersetzt_pattern = r'bei\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+ersetzt\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)'
        bei_ersetzt_matches = re.finditer(bei_ersetzt_pattern, text)
        for match in bei_ersetzt_matches:
            if match.lastindex >= 3:
                team_candidate = match.group(1).strip()
                player_in = match.group(2).strip()
                player_out = match.group(3).strip()
                
                # Validate team name using team extraction function
                context_for_team = text[max(0, match.start()-100):match.end()+50]
                team = self._extract_team_from_text(context_for_team, home_team=home_team, away_team=away_team)
                # If extraction didn't find a match but candidate looks valid, use it
                # But prefer the validated team name
                if not team and team_candidate and len(team_candidate) > 2:
                    # Check if candidate matches home or away team
                    if home_team and team_candidate.lower() in home_team.lower():
                        team = home_team
                    elif away_team and team_candidate.lower() in away_team.lower():
                        team = away_team
                    else:
                        team = team_candidate
                
                substitutions.append({
                    'player_in': player_in,
                    'player_out': player_out,
                    'team': team
                })
                # Track this match position
                matched_positions.add((match.start(), match.end()))
        
        # Pattern 3: General "PlayerIn für PlayerOut" or "PlayerIn ersetzt PlayerOut"
        # BUT exclude matches that come after team prefixes (like "bei Bochum", "den Bremern")
        # AND exclude matches that overlap with Pattern 1 or Pattern 2 matches
        multi_sub_pattern = r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+(?:für|kommt für|ersetzt)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)'
        
        all_matches = re.finditer(multi_sub_pattern, text)
        for match in all_matches:
            # Check if this match overlaps with any already matched position
            match_start, match_end = match.start(), match.end()
            overlaps = any(
                not (match_end <= prev_start or match_start >= prev_end)
                for prev_start, prev_end in matched_positions
            )
            if overlaps:
                continue
            
            # Check if this match comes after a team prefix (exclude it if so)
            start_pos = max(0, match.start() - 30)  # Increased lookback
            context_before = text[start_pos:match.start()]
            
            # Skip if preceded by team prefixes like "bei", "den", etc.
            if re.search(team_prefixes + r'[A-ZÄÖÜ]', context_before, re.IGNORECASE):
                continue
            
            # Additional check: skip if immediately after "bei [Team]" (Pattern 2 should have caught this)
            if re.search(r'bei\s+[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*\s+ersetzt\s*$', context_before, re.IGNORECASE):
                continue
            
            # Also skip if the matched text itself looks like a team name pattern
            # Common team patterns: single capitalized word before "ersetzt" or "für"
            player_in_candidate = match.group(1).strip()
            
            # Skip if it's likely a team name (very short and appears in context with "bei", "den", etc.)
            # But allow it if we've already skipped team prefixes above
            if match.lastindex >= 2:
                player_in = player_in_candidate
                player_out = match.group(2).strip()
                
                # Additional validation: player names typically have 2+ words or are longer
                # Team names sometimes are single words, but let's be more permissive
                # The key is to check context - if we've made it here and skipped team prefixes, it's likely valid
                
                # Extract team if present (check context around this match)
                # Use wider context to ensure we capture "bei den Gästen" patterns
                start_pos = max(0, match.start() - 200)
                end_pos = min(len(text), match.end() + 50)
                context = text[start_pos:end_pos]
                context_lower = context.lower()
                
                # Check for "bei den Gästen" (away team) or "bei den Heimern" (home team) FIRST
                # This must come before team extraction to avoid extracting "Gästen" as a team name
                if 'bei den gästen' in context_lower or 'bei den gast' in context_lower:
                    team = away_team if away_team else None
                elif 'bei den heimern' in context_lower or 'bei den heim' in context_lower:
                    team = home_team if home_team else None
                else:
                    # Use team extraction function
                    team = self._extract_team_from_text(context, home_team=home_team, away_team=away_team)
                
                substitutions.append({
                    'player_in': player_in,
                    'player_out': player_out,
                    'team': team
                })
                # Track this match position
                matched_positions.add((match.start(), match.end()))
        
        # Deduplicate substitutions: remove duplicates based on player_in and player_out
        # Keep the first occurrence with the best team information
        seen = {}
        deduplicated = []
        for sub in substitutions:
            key = (sub.get('player_in', '').lower().strip(), sub.get('player_out', '').lower().strip())
            if key not in seen:
                seen[key] = sub
                deduplicated.append(sub)
            else:
                # If we have a better team match (not None, validated against home/away), prefer it
                existing = seen[key]
                existing_team = existing.get('team')
                new_team = sub.get('team')
                # Prefer new team if it's validated (matches home or away team) and existing isn't
                replace_existing = False
                if new_team and existing_team:
                    if home_team and new_team == home_team and existing_team != home_team:
                        replace_existing = True
                    elif away_team and new_team == away_team and existing_team != away_team:
                        replace_existing = True
                elif new_team and not existing_team:
                    replace_existing = True
                
                if replace_existing:
                    seen[key] = sub
                    # Replace in deduplicated list
                    idx = next((i for i, s in enumerate(deduplicated) if s == existing), None)
                    if idx is not None:
                        deduplicated[idx] = sub
        
        # Pattern 4: Narrative substitution descriptions (only if no substitutions found yet)
        # "Ein scheidender Bremer geht vom Feld, ein weiterer Bremer...kommt in die Partie"
        # "Während Bochum wechselt"
        if not deduplicated:
            # Try to extract team from narrative substitution text
            narrative_team = None
            
            # Pattern: "während [Team] wechselt" - direct team mention
            während_match = re.search(r'während\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+wechselt', text, re.IGNORECASE)
            if während_match:
                team_candidate = während_match.group(1).strip()
                # Validate against known teams
                if home_team and team_candidate.lower() in home_team.lower():
                    narrative_team = home_team
                elif away_team and team_candidate.lower() in away_team.lower():
                    narrative_team = away_team
                else:
                    narrative_team = team_candidate
            
            # Pattern: "Bremer geht vom Feld" or mentions of team players leaving/entering
            # Also check for "bei den Gästen" pattern first
            if not narrative_team:
                text_lower_check = text.lower()
                if 'bei den gästen' in text_lower_check or 'bei den gast' in text_lower_check:
                    narrative_team = away_team if away_team else None
                elif 'bei den heimern' in text_lower_check or 'bei den heim' in text_lower_check:
                    narrative_team = home_team if home_team else None
                else:
                    narrative_team = self._extract_team_from_text(text, home_team=home_team, away_team=away_team)
            
            # If we found a narrative substitution with a team, add it
            if narrative_team:
                deduplicated.append({
                    'player_in': None,
                    'player_out': None,
                    'team': narrative_team
                })
        
        # If we found substitutions, return them
        if deduplicated:
            return deduplicated
        
        # Fallback: Try single substitution patterns (for cases where finditer doesn't work)
        single_sub_patterns = [
            r'(?:[Ss]pieler)?[Ww]echsel[:\s]+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)\s+(?:für|kommt für|ersetzt)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)',
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
                    team = self._extract_team_from_text(text, home_team=home_team, away_team=away_team)
                    
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
            True if attendance < 1500 (ghost game), False otherwise
        """
        if attendance is None:
            return False
        return attendance < 1500
    
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
        logger.info(f"Fetching match URLs for {season}, matchday {matchday} from {url}")
        
        # 2. Navigate and get HTML
        try:
            # Clear cookies before request
            self._clear_cookies()
            self.driver.get(url)
            # Longer delay for anti-ban (2-3 seconds, reduced from 3-6 for M1 Mac optimization)
            self._random_delay(2, 3)
            self._handle_cookie_consent()
            
            # Wait for match links to load (they might be loaded via JavaScript)
            try:
                # Wait for at least one match link to appear
                match_wait = WebDriverWait(self.driver, timeout=10)
                match_wait.until(
                    EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '-gegen-') and contains(@href, 'bundesliga')]"))
                )
                logger.debug("Match links detected on page")
            except Exception:
                # If no links found immediately, try scrolling to trigger lazy loading
                logger.debug("No match links found immediately, scrolling to trigger lazy loading...")
                self._scroll_page()
                self._random_delay(1, 2)  # Wait a bit after scrolling
            
            html = self._get_page_source_safe(timeout=30)
            if html is None:
                raise RuntimeError("Failed to get page source (timeout or Chrome hung)")
        except Exception as e:
            error_msg = str(e)
            # Check if it's a network disconnection error
            if "ERR_INTERNET_DISCONNECTED" in error_msg:
                logger.error(f"Network disconnection error fetching matchday page: {url}")
                logger.error("This is a network connectivity issue. The scraper will retry this matchday.")
            else:
                logger.error(f"Failed to fetch matchday page: {url} - {e}")
            # Re-raise so caller can handle retry logic
            raise
        
        # Check if page is valid (not 404 or error page)
        if not html or len(html) < 5000:
            logger.warning(f"Page seems short ({len(html)} chars if html else 0) for {url}")
            # Check for common error indicators
            html_lower = html.lower() if html else ""
            if "404" in html_lower or "nicht gefunden" in html_lower or "seite nicht gefunden" in html_lower:
                logger.warning(f"Page appears to be a 404 error page for {season}, matchday {matchday}")
                logger.warning(f"This matchday might not exist or the URL format might be incorrect for this season")
            return []
        
        # Check if page contains expected matchday content
        html_lower = html.lower()
        if "spieltag" not in html_lower and "match" not in html_lower:
            logger.warning(f"Page doesn't appear to contain matchday content for {season}, matchday {matchday}")
            logger.warning(f"URL: {url}")
            logger.warning(f"Page length: {len(html)} chars")
            # Log a snippet of the page to help debug
            if len(html) > 500:
                logger.debug(f"Page content snippet (first 500 chars): {html[:500]}")
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        match_urls_set = set()  # Use set for deduplication
        
        # 3. Link Extraction: Find all <a> tags
        links = soup.find_all('a', href=True)
        logger.info(f"Found {len(links)} total links on matchday page for {season}, matchday {matchday}")
        
        if len(links) == 0:
            logger.warning(f"No links found on page for {season}, matchday {matchday}")
            logger.warning(f"This might indicate the page structure has changed or the page is empty")
            return []
        
        # Keywords to exclude (only media galleries and women's league - we want analyse, liveticker)
        exclude_keywords = ['fotos', 'video', 'tabelle', 'frauen']
        
        links_with_gegen = 0
        links_with_bundesliga = 0
        excluded_links = 0
        failed_format_check = 0
        successfully_added = 0
        excluded_samples = []  # Store first few excluded links for debugging
        
        for link in links:
            href = link.get('href', '').strip()
            
            # Filter: A link is valid if and only if:
            # 1. Contains "-gegen-" (German for "-vs-")
            if '-gegen-' not in href:
                continue
            links_with_gegen += 1
            
            # 2. Must contain "bundesliga" (not "2-bundesliga" or other variations)
            # But exclude "frauen-bundesliga" (women's league)
            if 'bundesliga' not in href.lower():
                continue
            links_with_bundesliga += 1
            # Check for 'frauen' (women's league) or '2-bundesliga' (2nd division)
            # Use word boundary checks to avoid false positives (e.g., '2022-bundesliga' should not match '2-bundesliga')
            has_frauen = 'frauen' in href.lower()
            # Check for '2-bundesliga' but not as part of a year (e.g., '2022-bundesliga')
            # Pattern: '-2-bundesliga' or '2-bundesliga-' or start/end of string
            has_2_bundesliga = bool(re.search(r'(^|[^0-9])2-bundesliga([^0-9]|$)', href.lower()))
            if has_frauen or has_2_bundesliga:
                excluded_links += 1
                reason = 'frauen' if has_frauen else '2-bundesliga'
                if len(excluded_samples) < 3:
                    excluded_samples.append(f"{reason}: {href}")
                logger.warning(f"Excluding link ({reason}): {href}")
                continue
            
            # 3. Does NOT contain excluded keywords
            matching_keyword = None
            for keyword in exclude_keywords:
                if keyword in href.lower():
                    matching_keyword = keyword
                    break
            if matching_keyword:
                excluded_links += 1
                if len(excluded_samples) < 3:
                    excluded_samples.append(f"keyword '{matching_keyword}': {href}")
                logger.warning(f"Excluding link (keyword '{matching_keyword}'): {href}")
                continue
            
            # 4. Starts with '/' (relative) or 'https://www.kicker.de' (absolute)
            if not (href.startswith('/') or href.startswith('https://www.kicker.de')):
                failed_format_check += 1
                continue
            
            # 5. Cleaning: Convert relative links to absolute
            if href.startswith('/'):
                full_url = f"https://www.kicker.de{href}"
            elif href.startswith('https://www.kicker.de'):
                full_url = href
            else:
                failed_format_check += 1
                continue
            
            # Strip known suffixes to get clean base match URL
            clean_url = re.sub(r'/(analyse|liveticker|ticker|spielinfo|spielbericht)$', '', full_url)
            clean_url = clean_url.rstrip('/')
            
            # Validate clean URL before adding
            if not clean_url or len(clean_url) < 10:
                logger.debug(f"Skipping invalid clean URL: '{clean_url}' (from {href})")
                failed_format_check += 1
                continue
            
            # Add to set (automatic deduplication)
            match_urls_set.add(clean_url)
            successfully_added += 1
            logger.debug(f"Added match URL: {clean_url}")
        
        # If no matches found, try scrolling and reloading the page
        if len(match_urls_set) == 0 and len(links) > 0:
            logger.warning(f"No match URLs found on first attempt for {season}, matchday {matchday}. Trying scroll and reload...")
            try:
                # Scroll to trigger lazy loading
                self._scroll_page()
                self._random_delay(2, 3)
                # Reload HTML after scrolling
                html = self._get_page_source_safe(timeout=30)
                if html:
                    soup = BeautifulSoup(html, 'html.parser')
                    links = soup.find_all('a', href=True)
                    logger.info(f"After scrolling: Found {len(links)} total links")
                    
                    # Try extracting again
                    for link in links:
                        href = link.get('href', '').strip()
                        if '-gegen-' not in href:
                            continue
                        if 'bundesliga' not in href.lower():
                            continue
                        # Check for 'frauen' (women's league) or '2-bundesliga' (2nd division)
                        # Use word boundary checks to avoid false positives
                        has_frauen = 'frauen' in href.lower()
                        has_2_bundesliga = bool(re.search(r'(^|[^0-9])2-bundesliga([^0-9]|$)', href.lower()))
                        if has_frauen or has_2_bundesliga:
                            continue
                        if any(keyword in href.lower() for keyword in exclude_keywords):
                            continue
                        if not (href.startswith('/') or href.startswith('https://www.kicker.de')):
                            continue
                        
                        if href.startswith('/'):
                            full_url = f"https://www.kicker.de{href}"
                        elif href.startswith('https://www.kicker.de'):
                            full_url = href
                        else:
                            continue
                        
                        clean_url = re.sub(r'/(analyse|liveticker|ticker|spielinfo|spielbericht)$', '', full_url)
                        clean_url = clean_url.rstrip('/')
                        match_urls_set.add(clean_url)
                    
                    if len(match_urls_set) > 0:
                        logger.info(f"✓ Found {len(match_urls_set)} match URLs after scrolling for {season}, matchday {matchday}")
            except Exception as e:
                logger.debug(f"Error during scroll retry: {e}")
        
        # Log diagnostic information if still no matches found
        if len(match_urls_set) == 0:
            logger.warning(f"No match URLs found for {season}, matchday {matchday}")
            logger.warning(f"Diagnostics: {len(links)} total links, {links_with_gegen} with '-gegen-', {links_with_bundesliga} with 'bundesliga', {excluded_links} excluded, {failed_format_check} failed format check, {successfully_added} successfully added to set")
            if excluded_samples:
                logger.warning(f"Sample excluded links: {excluded_samples}")
            logger.warning(f"URL tried: {url}")
            # Log example links to help debug - show links with 'gegen' even if they don't pass other filters
            if len(links) > 0:
                example_links = [link.get('href', '') for link in links[:20]]
                logger.warning(f"Example links found on page (first 20): {example_links}")
                # Also show links that contain 'gegen' to see the format
                gegen_links = [link.get('href', '') for link in links if '-gegen-' in link.get('href', '').lower()][:10]
                if gegen_links:
                    logger.warning(f"Example links with '-gegen-' found: {gegen_links}")
                else:
                    logger.warning(f"No links with '-gegen-' found. Checking for alternative formats...")
                    # Check for other possible formats
                    vs_links = [link.get('href', '') for link in links if ' vs ' in link.get('href', '').lower() or '/vs/' in link.get('href', '').lower()][:10]
                    bundesliga_links = [link.get('href', '') for link in links if 'bundesliga' in link.get('href', '').lower()][:10]
                    if vs_links:
                        logger.warning(f"Found links with 'vs' format: {vs_links[:5]}")
                    if bundesliga_links:
                        logger.warning(f"Found bundesliga links (first 5): {bundesliga_links[:5]}")
        
        # Validation: Check if we got expected number of matches (9 per matchday for Bundesliga)
        match_urls_list = sorted(list(match_urls_set))
        if len(match_urls_list) != 9:
            logger.warning(f"Expected 9 matches per matchday, but found {len(match_urls_list)} matches for {season}, matchday {matchday}")
            if len(match_urls_list) == 0:
                logger.warning(f"This might indicate:")
                logger.warning(f"  1. The matchday hasn't been played yet")
                logger.warning(f"  2. The URL format is incorrect for this season")
                logger.warning(f"  3. The page structure has changed")
                logger.warning(f"  4. Network/page loading issues")
            else:
                logger.warning(f"This might indicate incomplete page loading. Consider retrying with scrolling enabled.")
        else:
            logger.info(f"✓ Found all 9 matches for {season}, matchday {matchday}")
        
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
    
    def parse_ticker(self, html: str) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[int]], Dict[int, Tuple[int, int]], Dict[str, Dict[str, int]]]:
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
            Tuple of (ticker_events, targets_dict, score_timeline, counts)
            - ticker_events: List of event dictionaries with cleaned text
            - targets_dict: Dictionary with announced_time_45/90 and actual_played_45/90
            - score_timeline: Dictionary mapping minute -> (home_score, away_score)
            - counts: Dictionary with yellow_cards, red_cards, yellow_red_cards, substitutes (minute -> count)
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
        
        # Count cards and substitutions by minute
        yellow_cards = {}  # minute -> count
        red_cards = {}  # minute -> count
        yellow_red_cards = {}  # minute -> count
        substitutes = {}  # minute -> count
        
        # Extract team names early for better team extraction in events
        home_team = None
        away_team = None
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract team names from HTML for use in team extraction
            # Method 1: Extract from title tag
            title_tag = soup.find('title')
            if title_tag:
                title_text = title_tag.get_text()
                title_match = re.search(r'([^-|]+?)\s*[-|]\s*([^-|]+?)\s*\d+:\d+', title_text)
                if title_match:
                    home_team = title_match.group(1).strip()
                    away_team = title_match.group(2).strip()
                    home_team = BeautifulSoup(home_team, 'html.parser').get_text()
                    away_team = BeautifulSoup(away_team, 'html.parser').get_text()
            
            # Method 2: Try scoreboard (fallback)
            if not home_team or not away_team:
                scoreboard = soup.select_one('.kick__scoreboard, .kick__v100-scoreBoard')
                if scoreboard:
                    team_links = scoreboard.find_all('a', href=re.compile(r'/info/bundesliga'))
                    if len(team_links) >= 2:
                        home_team = team_links[0].get_text(strip=True)
                        away_team = team_links[1].get_text(strip=True)
            
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
                
                # Extract text from header elements (h1-h6) and <p> tags
                # Header elements often contain card information (e.g., "Gelbe Karte", player name, team name)
                # and substitution information (e.g., "Spielerwechsel")
                # Also check for div/span elements with header-like classes that might contain structured event info
                header_elems = item.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                # Also check for div/span elements that might contain header-like content
                # (some ticker items structure headers differently)
                header_like_elems = item.find_all(['div', 'span'], class_=re.compile(r'ticker.*header|header.*ticker|event.*header|header.*event', re.I))
                text_elems = item.find_all('p')
                text = ""
                text_parts = []
                header_text_lower = ""  # Store header text separately for explicit detection
                
                # First, extract text from header elements (for structured event detection)
                all_header_elems = header_elems + header_like_elems
                if all_header_elems:
                    header_texts = []
                    for h in all_header_elems:
                        h_text = h.get_text(separator=' ', strip=False)
                        if h_text and h_text.strip():  # Only add non-empty header text
                            header_texts.append(h_text)
                            text_parts.append(h_text)
                    if header_texts:
                        header_text_lower = ' '.join(header_texts).lower()
                
                # Then, extract text from paragraph elements (main content)
                if text_elems:
                    for p in text_elems:
                        p_text = p.get_text(separator=' ', strip=False)
                        text_parts.append(p_text)
                
                if text_parts:
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
                
                # Simple and robust detection: If text contains:
                # 1. A context phrase (nachspielzeit, es gibt, obendrauf, nachgespielt, etc.)
                # 2. "minuten" or "minute"
                # 3. A number (digit or German word)
                # Then it's a stoppage time announcement
                
                # Context phrases that indicate stoppage time announcements
                context_phrases = [
                    'nachspielzeit', 'nachgespielt', 'nachgelegt', 'obendrauf', 
                    'oben drauf', 'draufgepackt', 'drauf gepackt', 'es gibt', 'gibt es', 
                    'nachschlag', 'zugabe', 'bonus', 'extra', 'tafel zeigt', 
                    'auf der tafel', 'vierter offizielle', 'werden noch', 'wird noch'
                ]
                has_context = any(phrase in text_lower_check for phrase in context_phrases)
                
                # Check for "minuten" or "minute"
                has_minuten = 'minuten' in text_lower_check or 'minute' in text_lower_check
                
                # Check for numbers (digits 1-15 or German number words)
                has_number = (
                    re.search(r'\b([1-9]|1[0-5])\b', text_lower_check) or
                    any(word in text_lower_check for word in ['eine', 'eins', 'ein', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun', 'zehn'])
                )
                
                # It's an announcement if it has context + minuten + number
                # BUT: Only check for announcements in messages that are:
                # - At minute 40+ (for first half) OR in overtime (45+)
                # - At minute 85+ (for second half) OR in overtime (90+)
                # This is where announcements typically appear
                minute_num = self._parse_minute_to_int(minute)
                is_overtime = '+' in minute if minute else False
                
                # Check if this message is in a relevant time period for announcements
                is_relevant_for_45 = False
                is_relevant_for_90 = False
                
                if minute_num:
                    if minute_num <= 45:
                        # First half: check if minute >= 40 OR in overtime (45+)
                        is_relevant_for_45 = (minute_num >= 40) or (is_overtime and minute_num == 45)
                    else:
                        # Second half: check if minute >= 85 OR in overtime (90+)
                        is_relevant_for_90 = (minute_num >= 85) or (is_overtime and minute_num >= 90)
                
                # Only treat as announcement if it matches the pattern AND is in relevant time period
                is_nachspielzeit_announcement = (
                    has_context and has_minuten and has_number and 
                    (is_relevant_for_45 or is_relevant_for_90)
                )
                
                # Extract announced_time from this event if it's a nachspielzeit announcement
                # Only skip THIS specific event, not other events at the same minute
                if is_nachspielzeit_announcement:
                    # Extract announced time before skipping
                    temp_announced_time = self._extract_announced_time(text)
                    if temp_announced_time is not None:
                        if is_relevant_for_45:
                            # We process events in reverse order (newest first)
                            # We want the LAST announcement chronologically (closest to halftime)
                            # So we always update to keep the latest one we've seen
                            targets['announced_time_45'] = temp_announced_time
                        elif is_relevant_for_90:
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
                # Detect goal - but only for in-game events (minute must not be empty)
                # Pre-match, halftime, and post-match events should not have goals
                is_goal = False
                if minute and minute.strip():
                    is_goal = 'kick__ticker-item--highlight' in item_classes
                
                # Detect card - explicitly check headers for card types
                # Check for card-related icon classes (various possible class names)
                card_icon = (
                    item.select_one('.kick__icon-Gelb') or
                    item.select_one('.kick__icon-Rot') or
                    item.select_one('[class*="icon-Gelb"]') or
                    item.select_one('[class*="icon-Rot"]') or
                    item.select_one('[class*="Gelb"]') or
                    item.select_one('[class*="Rot"]')
                )
                
                # Check for card-related text in headers and full text
                card_text_in_header = any(phrase in header_text_lower for phrase in [
                    'gelbe karte', 'gelb-rote karte', 'rote karte', 'gelb-rot', 'gelb rot'
                ])
                card_text_in_body = any(phrase in text_lower for phrase in [
                    'gelbe karte', 'gelb-rote karte', 'rote karte', 'gelb-rot', 'gelb rot'
                ])
                
                is_card = (
                    'kick__ticker-item--card' in item_classes or
                    card_icon is not None or
                    card_text_in_header or
                    card_text_in_body
                )
                
                # Detect substitution - explicitly check headers for substitution keywords
                # IMPORTANT: Don't treat goals as substitutions (goals often have "für" which could match)
                is_substitution = (
                    not is_goal and (  # Never treat goals as substitutions
                        'kick__ticker-item--substitution' in item_classes or
                        item.select_one('.kick__icon-Wechsel') or
                        'spielerwechsel' in header_text_lower or
                        ('wechsel' in header_text_lower and 'eingewechselt' not in text_lower) or  # "eingewechselt" is about a substitution that happened before
                        'spielerwechsel' in text_lower or
                        ('wechsel' in text_lower and 'eingewechselt' not in text_lower and 'tor' not in text_lower)
                    )
                )
                
                # Extract card type from text if it's a card event
                # Check both header text and full text for card type indicators
                card_type = None
                card_player = None
                card_team = None
                if is_card:
                    # Check header text first (more structured), then full text as fallback
                    # Combine header and text for card type detection
                    combined_text_for_card = (header_text_lower + ' ' + text_lower).strip()
                    # Check in order of specificity (most specific first)
                    if 'gelb-rote karte' in combined_text_for_card or 'gelb-rot' in combined_text_for_card or 'gelb rot' in combined_text_for_card:
                        card_type = 'Yellow-Red'
                    elif 'rote karte' in combined_text_for_card or ('rot' in combined_text_for_card and 'gelb-rot' not in combined_text_for_card):
                        card_type = 'Red'
                    elif 'gelbe karte' in combined_text_for_card or ('gelb' in combined_text_for_card and 'gelb-rot' not in combined_text_for_card):
                        card_type = 'Yellow'
                    # Note: We check for single words ('gelb', 'rot') as fallback, but exclude if they're part of 'gelb-rot'
                    
                    # Extract player name and team for card events (keep for event dict)
                    card_player = self._extract_player_name_from_text(text, is_card=True)
                    card_team = self._extract_team_from_text(text, home_team=home_team, away_team=away_team)
                    
                    # Count cards by minute (only for in-game events)
                    if minute and minute.strip() and card_type:
                        minute_int = self._parse_minute_to_int(minute)
                        if minute_int is not None:
                            if card_type == 'Yellow':
                                yellow_cards[minute_int] = yellow_cards.get(minute_int, 0) + 1
                            elif card_type == 'Red':
                                red_cards[minute_int] = red_cards.get(minute_int, 0) + 1
                            elif card_type == 'Yellow-Red':
                                yellow_red_cards[minute_int] = yellow_red_cards.get(minute_int, 0) + 1
                
                # Count substitutions by minute (only for in-game events, not goals)
                # IMPORTANT: Only count when "Spielerwechsel" is explicitly in the text/header
                # Don't count events that just describe substitutions (like minute 73 describing minute 72's substitution)
                has_spielerwechsel = (
                    'spielerwechsel' in header_text_lower or
                    'spielerwechsel' in text_lower
                )
                if has_spielerwechsel and not is_goal and minute and minute.strip():
                    minute_int = self._parse_minute_to_int(minute)
                    if minute_int is not None:
                        # Count 1 substitution per "Spielerwechsel" event
                        substitutes[minute_int] = substitutes.get(minute_int, 0) + 1
                
                # Process event to extract targets and clean text
                clean_text, announced_time, actual_played_time, text_extracted_score = self._process_ticker_event(
                    text, minute, is_goal_event=is_goal
                )
                
                # Additional text cleaning: Remove redundant minute and event type information
                # This prevents patterns like "Gelbe Karte 35' Gelbe Karte" or "Tor für Bremen 80' Tor"
                if is_card and minute:
                    # Extract numeric part of minute and make apostrophe optional
                    minute_num = re.sub(r"[^\d]", "", minute)  # Extract just the number
                    if minute_num:
                        minute_pattern = re.escape(minute_num) + r"\'?"
                        # Remove patterns like "Gelbe Karte 35'" or "35' Gelbe Karte" or "Gelbe Karte 35' Gelbe Karte"
                        # Pattern: (card type) + minute + (optional card type)
                        clean_text = re.sub(
                            r'(?:Gelbe Karte|Rote Karte|Gelb-Rote Karte|Gelbe|Rote|Gelb-Rote)\s*' + minute_pattern + r'\s*(?:Gelbe Karte|Rote Karte|Gelb-Rote Karte|Gelbe|Rote|Gelb-Rote)?',
                            '', clean_text, flags=re.IGNORECASE
                        ).strip()
                        # Remove minute at start if it remains
                        clean_text = re.sub(r'^' + minute_pattern + r'\s*', '', clean_text).strip()
                    # Remove redundant card type mentions
                    clean_text = re.sub(r'\b(Gelbe Karte|Rote Karte|Gelb-Rote Karte)\s+\1\b', r'\1', clean_text, flags=re.IGNORECASE).strip()
                
                if is_goal and minute:
                    # Extract numeric part of minute and make apostrophe optional
                    minute_num = re.sub(r"[^\d]", "", minute)
                    if minute_num:
                        minute_pattern = re.escape(minute_num) + r"\'?"
                        # Remove patterns like "Tor für [Team] 80'" or "80' Tor für [Team]"
                        # Match "Tor für [anything] 80'" or "80' Tor für [anything]"
                        clean_text = re.sub(
                            r'(?:Tor\s+für\s+[^\s]+(?:\s+[^\s]+)*\s*' + minute_pattern + r')|(?:' + minute_pattern + r'\s*Tor\s+für)',
                            '', clean_text, flags=re.IGNORECASE
                        ).strip()
                        # Remove minute at start if it remains
                        clean_text = re.sub(r'^' + minute_pattern + r'\s*', '', clean_text).strip()
                
                if is_substitution and minute:
                    # Extract numeric part of minute and make apostrophe optional
                    minute_num = re.sub(r"[^\d]", "", minute)
                    if minute_num:
                        minute_pattern = re.escape(minute_num) + r"\'?"
                        # Remove patterns like "Spielerwechsel 77'" or "77' Spielerwechsel"
                        # Match "Spielerwechsel 77'" or "77' Spielerwechsel" or "Wechsel 77'"
                        clean_text = re.sub(
                            r'(?:Spielerwechsel|Wechsel)\s*' + minute_pattern + r'|' + minute_pattern + r'\s*(?:Spielerwechsel|Wechsel)',
                            '', clean_text, flags=re.IGNORECASE
                        ).strip()
                        # Remove minute at start if it remains
                        clean_text = re.sub(r'^' + minute_pattern + r'\s*', '', clean_text).strip()
                
                # Clean up any remaining excessive whitespace
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                
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
                # Always include player and team fields for card events (even if None for consistency)
                if is_card and card_type:
                    event_dict['card_type'] = card_type
                    event_dict['player'] = card_player  # Include even if None
                    event_dict['team'] = card_team  # Include even if None
                
                # Add extracted score if available (for timeline building)
                # IMPORTANT: Only add goals for in-game events (minute must not be empty)
                if extracted_score and minute and minute.strip():
                    event_dict['extracted_score'] = extracted_score
                
                # Add current score state for goals (like Gemini's score_at_event)
                # IMPORTANT: Only add goals for in-game events (minute must not be empty)
                if is_goal and minute and minute.strip() and (current_home > 0 or current_away > 0):
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
            
            # Validation: Check that we have Anpfiff and Abpfiff (required for complete match)
            if anpfiff_index is None:
                logger.warning(f"⚠️  Anpfiff event not found in ticker for match {match_id}. This might indicate incomplete page loading.")
            if abpfiff_index is None:
                logger.warning(f"⚠️  Abpfiff event not found in ticker for match {match_id}. This might indicate incomplete page loading.")
            if anpfiff_index is not None and abpfiff_index is not None:
                # Count events between Anpfiff and Abpfiff
                in_game_events = abpfiff_index - anpfiff_index - 1
                logger.debug(f"✓ Found {in_game_events} events between Anpfiff (index {anpfiff_index}) and Abpfiff (index {abpfiff_index})")
            elif anpfiff_index is None or abpfiff_index is None:
                logger.warning(f"⚠️  Missing key events (Anpfiff: {anpfiff_index is not None}, Abpfiff: {abpfiff_index is not None}). Consider retrying with scrolling enabled.")
            
            # Second pass: set minute to empty for pre/post match events
            # No event_type field - we just need to clean up the minute field
            # Also remove goal information from pre/post match events (substitutions/cards are counted separately)
            for idx, event in enumerate(ticker_events):
                # Pre-match: everything until and including "Anpfiff" (exact match)
                if anpfiff_index is not None and idx <= anpfiff_index:
                    # Set minute to empty for pre-match events
                    event['minute'] = ""
                    # Remove goal information from pre-match events
                    event.pop('extracted_score', None)
                    event.pop('score_at_event', None)
                
                # Post-match: after "Abpfiff" (exact match)
                elif abpfiff_index is not None and idx > abpfiff_index:
                    # Set minute to empty for post-match events
                    event['minute'] = ""
                    # Remove goal information from post-match events
                    event.pop('extracted_score', None)
                    event.pop('score_at_event', None)
                
                # Also clean up any events that already have empty minute (shouldn't have goals)
                elif not event.get('minute') or not event.get('minute', '').strip():
                    event.pop('extracted_score', None)
                    event.pop('score_at_event', None)
        
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
        # Convert counts to string keys for JSON serialization (same format as score_timeline)
        counts = {
            'yellow_cards': {str(k): v for k, v in yellow_cards.items()},
            'red_cards': {str(k): v for k, v in red_cards.items()},
            'yellow_red_cards': {str(k): v for k, v in yellow_red_cards.items()},
            'substitutes': {str(k): v for k, v in substitutes.items()}
        }
        return (ticker_events, targets, score_timeline, counts)
    
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
        matchday: Optional[int] = None,
        force_rescrape: bool = False,
        is_retry: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Scrape a full match by visiting Spielinfo and Ticker tabs sequentially using Selenium.
        
        Extracts ticker events directly from server-side rendered HTML.
        
        Args:
            match_url: Base match URL
            match_id: Optional match ID (extracted from URL if not provided)
            season: Optional season string
            matchday: Optional matchday number
            force_rescrape: If True, re-scrape even if match is already downloaded (default: False)
            is_retry: If True, uses longer timeout (30s) for slow pages; if False, uses shorter timeout (15s) for speed (default: False)
            
        Returns:
            Dictionary with match data or None if scraping fails or match already downloaded
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
        
        # Check if match is already downloaded - skip if it exists (unless force_rescrape)
        if not force_rescrape and self._is_match_downloaded(match_id, season):
            logger.info(f"Match {match_id} already downloaded, skipping...")
            return None
        
        logger.info(f"Scraping match {match_id} from {match_url}")
        
        # Clear cookies before each match to avoid tracking
        self._clear_cookies()
        
        # Get base URL
        base_url = match_url.split('/ticker')[0].split('/liveticker')[0].split('/spielinfo')[0]
        
        # Step 1: Spielinfo Tab (Stats/Metadata)
        info_url = f"{base_url}/spielinfo"
        logger.debug(f"Fetching spielinfo tab: {info_url}")
        info_html = None
        match_info = {'info': {}, 'stats': {}}
        
        try:
            # Clear cookies before each request
            self._clear_cookies()
            self.driver.get(info_url)
            # Longer delay for anti-ban (2-3 seconds, reduced from 3-6 for M1 Mac optimization)
            self._random_delay(2, 3)
            self._handle_cookie_consent()
            
            # No scrolling for normal operation (CPU optimization)
            # Scrolling only happens on retry if needed
            
            info_html = self._get_page_source_safe(timeout=30)
            if info_html is None:
                logger.error("Failed to get spielinfo page source (timeout or Chrome hung)")
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
            # Clear cookies before each request
            self._clear_cookies()
            self.driver.get(ticker_url)
            # Longer delay for anti-ban (2-3 seconds, reduced from 3-6 for M1 Mac optimization)
            self._random_delay(2, 3)
            self._handle_cookie_consent()
            
            # Wait for any ticker item to load (using CLASS_NAME like Gemini - more reliable)
            # Use adaptive timeout: shorter for first attempt (faster), longer for retries (more patient)
            ticker_timeout = 30 if is_retry else 15  # 15s first attempt, 30s on retry
            try:
                ticker_wait = WebDriverWait(self.driver, timeout=ticker_timeout)
                ticker_wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, "kick__ticker-item"))
                )
                logger.debug("Ticker items found")
            except Exception as e:
                logger.warning(f"Timeout waiting for ticker items ({ticker_timeout}s timeout): {e}")
                if not is_retry:
                    logger.warning("This might indicate a slow page. Will use longer timeout on retry.")
                else:
                    logger.warning("This might indicate a slow page or network issue. Continuing anyway...")
            
            # Only scroll on retry to trigger lazy-loading (CPU optimization)
            # Normal operation: no scrolling - page should load all content without it
            if is_retry:
                logger.debug("Retry detected - scrolling to trigger lazy-loading")
                self._scroll_page()
            
            ticker_html = self._get_page_source_safe(timeout=30)
            if ticker_html is None:
                logger.error("Failed to get ticker page source (timeout or Chrome hung)")
                return None
        except Exception as e:
            logger.error(f"Failed to fetch ticker tab: {e}")
            return None
        
        if not self._validate_match_page(ticker_html, "ticker"):
            logger.error(f"Ticker page validation failed for {match_id}")
            logger.error("This could indicate:")
            logger.error("  - The match page doesn't exist (404)")
            logger.error("  - The page format has changed")
            logger.error("  - Network issues prevented proper page load")
            logger.error("This match will be retried automatically on next run")
            return None
        
        # Parse ticker from HTML (server-side rendered, no API needed)
        ticker_events, targets, ticker_score_timeline, counts = self.parse_ticker(ticker_html)
        
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
            'yellow_cards': counts['yellow_cards'],
            'red_cards': counts['red_cards'],
            'yellow_red_cards': counts['yellow_red_cards'],
            'substitutes': counts['substitutes'],
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


def scrape_seasons(seasons: List[str], save_dir: Path, process_id: int, max_retries: int = 2) -> List[Dict[str, int]]:
    """
    Scrape all matches for multiple seasons (runs in separate process).
    
    This function is designed to run in a multiprocessing pool, with 2 seasons
    per process. Each process has its own Chrome driver instance running in
    incognito mode and background.
    
    Args:
        seasons: List of season strings (e.g., ["2023-24", "2024-25"])
        save_dir: Directory to save scraped match JSON files
        process_id: Unique identifier for this process (for logging)
        max_retries: Maximum number of retries for failed matches (default: 2)
        
    Returns:
        List of dictionaries with statistics per season: [{'season': str, 'total': int, 'successful': int, 'failed': int, 'failed_urls': List[str]}, ...]
    """
    # Configure logging for this process
    season_str = ", ".join(seasons)
    logging.basicConfig(
        level=logging.INFO,
        format=f'[PROCESS {process_id}: {season_str}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add random delay to stagger process initialization and reduce race conditions
    # Each process waits 0-5 seconds before starting
    initial_delay = random.uniform(0, 5)
    time.sleep(initial_delay)
    
    # Initialize scraper for these seasons (each process gets its own driver)
    results = []
    
    try:
        # Use Context Manager for guaranteed cleanup
        with KickerScraper(save_dir=save_dir, headless=True) as scraper:
            # Process each season sequentially in this process
            for season in seasons:
                # Statistics for this season
                total_matches = 0
                successful_matches = 0
                failed_matches = 0
                failed_urls = []  # Track failed match URLs for retry
                
                try:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Processing season: {season}")
                    logger.info(f"{'='*60}")
                    
                    # Iterate through matchdays (1-34 for Bundesliga)
                    for matchday in range(1, 35):
                        logger.info(f"\nSeason {season}, Matchday {matchday}")
                        
                        try:
                            # Check if matchday is already complete (skip visiting page if all matches exist)
                            if scraper._is_matchday_complete(season, matchday):
                                logger.info(f"✓ Matchday {matchday} already complete, skipping...")
                                # Load metadata to get match count for statistics
                                metadata_path = scraper._get_matchday_metadata_path(season, matchday)
                                try:
                                    with open(metadata_path, 'r', encoding='utf-8') as f:
                                        metadata = json.load(f)
                                        match_count = metadata.get('total_matches', 0)
                                        total_matches += match_count
                                        successful_matches += match_count
                                        logger.info(f"  → {match_count} matches already downloaded")
                                except (json.JSONDecodeError, IOError):
                                    pass
                                continue
                            
                            # Get match URLs with retry logic for network failures
                            match_urls = []
                            matchday_retries = 2  # Retry matchday page fetch up to 2 times
                            for matchday_attempt in range(matchday_retries + 1):
                                try:
                                    match_urls = scraper.get_match_urls(season, matchday)
                                    if match_urls:
                                        break  # Success, exit retry loop
                                    elif matchday_attempt < matchday_retries:
                                        logger.warning(f"No matches found for {season}, matchday {matchday} (attempt {matchday_attempt + 1}/{matchday_retries + 1}), retrying...")
                                        scraper._random_delay(3, 8)  # Reduced from 5-10s for M1 Mac optimization
                                    else:
                                        logger.warning(f"No matches found for {season}, matchday {matchday} after {matchday_retries + 1} attempts")
                                except Exception as matchday_error:
                                    if matchday_attempt < matchday_retries:
                                        error_msg = str(matchday_error)
                                        if "ERR_INTERNET_DISCONNECTED" in error_msg or "timeout" in error_msg.lower():
                                            logger.warning(f"Network error fetching matchday page (attempt {matchday_attempt + 1}/{matchday_retries + 1}): {matchday_error}")
                                            logger.info(f"Retrying in 3-8 seconds...")
                                            scraper._random_delay(3, 8)  # Reduced from 10-15s for M1 Mac optimization
                                        else:
                                            # Non-network error, don't retry
                                            logger.error(f"Error fetching matchday page: {matchday_error}")
                                            break
                                    else:
                                        logger.error(f"Failed to fetch matchday page after {matchday_retries + 1} attempts: {matchday_error}")
                            
                            if not match_urls:
                                logger.warning(f"No matches found for {season}, matchday {matchday} - skipping this matchday")
                                # Track failed matchdays for potential retry
                                failed_urls.append({
                                    'url': f"matchday_{season}_{matchday}",
                                    'season': season,
                                    'matchday': matchday,
                                    'error': f"No matches found after {matchday_retries + 1} attempts",
                                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'type': 'matchday_fetch_failed'
                                })
                                continue
                            
                            # Check if all matches are already downloaded (handles case where matches exist but metadata doesn't)
                            all_matches_downloaded = True
                            for match_url in match_urls:
                                match_id = scraper._extract_match_id_from_url(match_url)
                                if not scraper._is_match_downloaded(match_id, season):
                                    all_matches_downloaded = False
                                    break
                            
                            if all_matches_downloaded:
                                # All matches already exist, save metadata and skip scraping
                                logger.info(f"✓ All {len(match_urls)} matches for matchday {matchday} already downloaded")
                                scraper._save_matchday_metadata(season, matchday, match_urls)
                                total_matches += len(match_urls)
                                successful_matches += len(match_urls)
                                logger.info(f"  → Saved metadata, skipping individual match scraping")
                                continue
                            
                            # Scrape each match with retry logic
                            for match_url in match_urls:
                                total_matches += 1
                                success = False
                                last_error = None
                                
                                # Retry loop
                                for attempt in range(max_retries + 1):  # 0 to max_retries (inclusive)
                                    try:
                                        is_retry_attempt = attempt > 0  # True if this is a retry (not first attempt)
                                        result = scraper.scrape_full_match(
                                            match_url, 
                                            season=season, 
                                            matchday=matchday,
                                            is_retry=is_retry_attempt
                                        )
                                        
                                        if result:
                                            successful_matches += 1
                                            logger.info(f"✓ Successfully scraped match {result['match_id']}")
                                            success = True
                                            break  # Success, exit retry loop
                                        else:
                                            # Result is None - could be skipped (already downloaded) or failed
                                            # Check if file exists to distinguish
                                            match_id = scraper._extract_match_id_from_url(match_url)
                                            if scraper._is_match_downloaded(match_id, season):
                                                # Already downloaded, not a failure
                                                success = True
                                                logger.debug(f"Match {match_id} already downloaded, skipping...")
                                                break
                                            else:
                                                # Failed but no exception - treat as failure
                                                if attempt < max_retries:
                                                    logger.warning(f"Match {match_id} returned None (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                                                    scraper._random_delay(3, 8)  # Reduced from 5-10s for M1 Mac optimization
                                                else:
                                                    last_error = "scrape_full_match returned None"
                                    
                                    except Exception as e:
                                        last_error = str(e)
                                        if attempt < max_retries:
                                            logger.warning(f"✗ Error scraping match from {match_url} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                                            logger.info(f"Retrying in 3-8 seconds...")
                                            scraper._random_delay(3, 8)  # Reduced from 5-10s for M1 Mac optimization
                                        else:
                                            logger.error(f"✗ Error scraping match from {match_url} after {max_retries + 1} attempts: {e}")
                                
                                # If all retries failed, record it
                                if not success:
                                    failed_matches += 1
                                    failed_match_info = {
                                        'url': match_url,
                                        'season': season,
                                        'matchday': matchday,
                                        'error': last_error,
                                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                                    }
                                    failed_urls.append(failed_match_info)
                                    logger.error(f"✗ FAILED to scrape {match_url} after {max_retries + 1} attempts")
                                    logger.error(f"  Error: {last_error}")
                                    logger.error(f"  This match will be saved to failed_matches.json for retry")
                                
                                # Longer delay between matches for anti-ban (4-8 seconds)
                                scraper._random_delay(3, 5)  # Reduced from 4-8s for M1 Mac optimization
                            
                            # Save matchday metadata if we successfully got match URLs
                            # This allows us to skip the matchday page on future runs
                            if match_urls:
                                scraper._save_matchday_metadata(season, matchday, match_urls)
                                logger.debug(f"Saved matchday metadata for {season}, matchday {matchday}")
                    
                        except Exception as e:
                            logger.error(f"Error processing matchday {matchday} for season {season}: {e}")
                            # Continue with next matchday even if one fails
                            continue
                    
                    # Print summary for this season
                    logger.info(f"\n{'='*60}")
                    logger.info(f"SEASON {season} SUMMARY")
                    logger.info(f"{'='*60}")
                    logger.info(f"Total matches attempted: {total_matches}")
                    logger.info(f"Successful: {successful_matches}")
                    logger.info(f"Failed: {failed_matches}")
                    logger.info(f"Success rate: {successful_matches/total_matches*100:.1f}%" if total_matches > 0 else "N/A")
                    
                    results.append({
                        'season': season,
                        'total': total_matches,
                        'successful': successful_matches,
                        'failed': failed_matches,
                        'failed_urls': failed_urls
                    })
                
                except Exception as e:
                    logger.error(f"Error in scraping loop for season {season}: {e}")
                    # Add failed season result
                    results.append({
                        'season': season,
                        'total': total_matches,
                        'successful': successful_matches,
                        'failed': failed_matches,
                        'failed_urls': failed_urls
                    })
                    # Continue with next season even if one fails
        # Context Manager automatically closes driver here
    
    except Exception as e:
        logger.error(f"Fatal error in process {process_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return results


if __name__ == "__main__":
    """
    Main execution block: Simultaneous season download using multiprocessing.
    
    Scrapes all Bundesliga matches from 2024-25 to 2017-18 (latest first) in parallel.
    Uses multiprocessing to run 2 seasons per process (5 processes total), each with
    its own Chrome driver instance in incognito mode and background. This reduces
    concurrent driver initializations and improves stability.
    
    By default, runs in single-process mode for testing. Set USE_MULTIPROCESSING = True
    in the main block to enable parallel processing.
    
    How to run:
        # From project root:
        python src/data/kicker_scraper.py
        
        # Or from the src/data directory:
        python kicker_scraper.py
    
    Modes:
        - Single-process mode (default): All seasons run sequentially in one process
          - Good for testing and debugging
          - Easier to see what's happening
          - No lock contention issues
        
        - Multiprocessing mode: 5 processes (2 seasons per process, last with 1)
          - Set USE_MULTIPROCESSING = True in the code
          - Faster execution
          - Each process runs Chrome in incognito mode and background
          - Processes continue even if individual matches or seasons fail
    
    What happens:
        1. Scrapes all matchdays (1-34) for each season sequentially
        2. Runs Chrome in incognito mode and background
        3. Already downloaded matches are automatically skipped
        4. Progress is logged with season-specific prefixes
        5. Continues even if individual matches fail
        6. Final summary shows statistics for all seasons
    
    Performance:
        - Single-process: ~2,448 matches sequentially (slower but stable)
        - Multiprocessing: 4 processes, 2 seasons each
        - Each season processes ~306 matches (9 matches/matchday × 34 matchdays)
        - Total: ~2,448 matches across all seasons
        - Estimated time: Several hours (depends on network and delays)
    
    Safety features:
        - Incognito mode prevents cookie tracking
        - Cookies cleared between requests
        - Random user agents
        - Longer delays (4-8 seconds) between matches
        - Automatic skip of already downloaded matches
    
    For testing individual matches, use tests/test_scraper_live.py instead:
        python -m tests.test_scraper_live --test-known
        python -m tests.test_scraper_live --test-bayern
        python -m tests.test_scraper_live --test-bremen
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Target seasons (VAR floor: 2017-18 onwards)
    # Process from latest to earliest (2024-25 first, then backwards to 2017-18)
    all_seasons = [
        "2024-25",  # Placebo Test Season (latest - processed first)
        "2023-24", "2022-23",  # Post-Corona, Clean
        "2021-22", "2020-21", "2019-20",  # Corona/Ghost Games (flagged but kept)
        "2018-19", "2017-18"  # Pre-Corona, Clean (earliest - processed last)
    ]
    
    save_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    
    # Configuration: Set to False for single-process mode (testing), True for multiprocessing
    USE_MULTIPROCESSING = True  # Set to True when ready for parallel processing
    
    if USE_MULTIPROCESSING:
        # Group seasons for 4 processes (2 seasons per process)
        # This provides good parallelization while maintaining stability
        seasons_per_process = 2
        season_groups = []
        for i in range(0, len(all_seasons), seasons_per_process):
            group = all_seasons[i:i + seasons_per_process]
            season_groups.append(group)
        
        num_processes = len(season_groups)
        
        logger.info(f"Starting multiprocessing scrape: {len(all_seasons)} seasons, {num_processes} processes")
        logger.info(f"Configuration: {seasons_per_process} seasons per process")
        logger.info("Each process runs Chrome in incognito mode and background")
        logger.info("Processes will continue even if individual matches or seasons fail")
        
        # Collect all results
        all_results = []
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Create tasks: one per season group
            tasks = [
                pool.apply_async(scrape_seasons, (group, save_dir, i+1)) 
                for i, group in enumerate(season_groups)
            ]
            
            # Wait for all processes to complete and collect results
            # Use timeout and error handling to prevent one failure from stopping everything
            for i, task in enumerate(tasks):
                try:
                    process_results = task.get(timeout=None)  # Wait indefinitely for each process
                    all_results.extend(process_results)
                except Exception as e:
                    logger.error(f"Process {i+1} failed with error: {e}")
                    # Add placeholder results for failed seasons in this group
                    for season in season_groups[i]:
                        all_results.append({
                            'season': season,
                            'total': 0,
                            'successful': 0,
                            'failed': 0,
                            'failed_urls': [],
                            'error': str(e)
                        })
    else:
        # Single-process mode for testing
        logger.info(f"Starting single-process scrape: {len(all_seasons)} seasons")
        logger.info("Running in single-process mode (testing/debugging)")
        logger.info("Set USE_MULTIPROCESSING = True to enable parallel processing")
        
        # Run all seasons sequentially in a single process
        all_results = scrape_seasons(all_seasons, save_dir, process_id=1)
    
    # Print overall summary
    logger.info(f"\n{'='*60}")
    logger.info("OVERALL SCRAPING SUMMARY")
    logger.info(f"{'='*60}")
    
    total_all = sum(r.get('total', 0) for r in all_results)
    successful_all = sum(r.get('successful', 0) for r in all_results)
    failed_all = sum(r.get('failed', 0) for r in all_results)
    
    logger.info(f"Total matches attempted: {total_all}")
    logger.info(f"Successful: {successful_all}")
    logger.info(f"Failed: {failed_all}")
    logger.info(f"Success rate: {successful_all/total_all*100:.1f}%" if total_all > 0 else "N/A")
    
    # Collect all failed URLs
    all_failed_urls = []
    failed_matchdays = []
    for result in all_results:
        failed_urls = result.get('failed_urls', [])
        all_failed_urls.extend(failed_urls)
        # Separate matchday fetch failures from match failures
        for failed_item in failed_urls:
            if failed_item.get('type') == 'matchday_fetch_failed':
                failed_matchdays.append(failed_item)
    
    # Save failed URLs to a file for later retry
    if all_failed_urls:
        failed_file = save_dir.parent / "failed_matches.json"
        try:
            # Load existing failed matches if file exists (to append, not overwrite)
            existing_failed = []
            if failed_file.exists():
                try:
                    with open(failed_file, 'r', encoding='utf-8') as f:
                        existing_failed = json.load(f)
                    logger.info(f"Found {len(existing_failed)} existing failed matches in {failed_file}")
                except Exception as e:
                    logger.warning(f"Could not read existing failed_matches.json: {e}")
            
            # Merge with new failures (avoid duplicates by URL)
            existing_urls = {item['url'] for item in existing_failed}
            new_failed = [item for item in all_failed_urls if item['url'] not in existing_urls]
            all_failed_merged = existing_failed + new_failed
            
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(all_failed_merged, f, ensure_ascii=False, indent=2)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"⚠️  FAILED MATCHES SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Total failed matches: {len(all_failed_merged)}")
            logger.info(f"  - New failures this run: {len(new_failed)}")
            logger.info(f"  - Previously failed: {len(existing_failed)}")
            
            # Show breakdown by failure type
            matchday_failures = [f for f in all_failed_merged if f.get('type') == 'matchday_fetch_failed']
            match_failures = [f for f in all_failed_merged if f.get('type') != 'matchday_fetch_failed']
            
            if matchday_failures:
                logger.info(f"\n  - Matchday fetch failures: {len(matchday_failures)}")
                logger.info(f"    (These are network issues fetching matchday pages)")
                logger.info(f"    (They will be retried automatically on next run)")
            
            if match_failures:
                logger.info(f"\n  - Individual match failures: {len(match_failures)}")
                logger.info(f"    (These are matches that failed to scrape)")
            
            logger.info(f"\nFailed matches saved to: {failed_file.absolute()}")
            logger.info(f"\nTo retry failed matches, run:")
            logger.info(f"  python src/data/retry_failed_matches.py")
            logger.info(f"\nOr simply re-run the scraper - it will automatically retry")
            logger.info(f"matches that don't have files yet.")
            logger.info(f"{'='*60}")
        except Exception as e:
            logger.error(f"Failed to save failed matches list: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Per-season breakdown
    logger.info(f"\nPer-season breakdown:")
    for result in all_results:
        season = result.get('season', 'Unknown')
        successful = result.get('successful', 0)
        total = result.get('total', 0)
        failed = result.get('failed', 0)
        failed_urls_count = len(result.get('failed_urls', []))
        if 'error' in result:
            logger.info(f"  {season}: ERROR - {result['error']}")
        else:
            logger.info(f"  {season}: {successful}/{total} successful, {failed} failed")
            if failed_urls_count > 0:
                logger.info(f"    → {failed_urls_count} failed matches will be retried on next run")
