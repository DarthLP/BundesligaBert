"""
Match Data Processing Script

This script processes raw match JSON files from data/raw into structured features
for econometric analysis. It extracts targets, separates event phases, engineers
features, and constructs BERT input text while preventing data leakage.

The script supports three econometric models:
1. Baseline Regression: Announced_Time ~ Goals + Subs + Cards + Injuries + VAR + Pressure
2. BERT Model: Announced_Time ~ Deep Textual Context
3. Whistle Regression: Excess_Time ~ Overtime_Events + Pressure

Usage:
    # Process all matches from all seasons
    python src/data/process_match_data.py
    
    # Process matches from a specific season
    python src/data/process_match_data.py --season 2022-23
    
    # Process specific match file
    python src/data/process_match_data.py --input data/raw/season_2022-23/match_xxx.json

Input:
    JSON files from data/raw/season_{season}/match_{match_id}.json

Output:
    Processed JSON files to data/processed/season_{season}/match_{match_id}.json

Author: BundesligaBERT Project
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def parse_minute_to_int(minute_str: str) -> Optional[int]:
    """
    Parse minute string to integer (handle "45+2" -> 45).
    
    Args:
        minute_str: Minute string (may include "+" notation)
        
    Returns:
        Integer minute or None if not parseable
    """
    if not minute_str:
        return None
    
    match = re.search(r'(\d+)', minute_str)
    if match:
        return int(match.group(1))
    return None


def parse_minute_for_sorting(minute_str: str) -> tuple:
    """
    Parse minute string for sorting (handles "90+3" format).
    
    Returns a tuple (base_minute, overtime_offset) for proper sorting.
    "90+3" -> (90, 3), "90" -> (90, 0), "87" -> (87, 0)
    
    Args:
        minute_str: Minute string (may include "+" notation)
        
    Returns:
        Tuple (base_minute, overtime_offset) for sorting
    """
    if not minute_str:
        return (0, 0)
    
    # Match pattern like "90+3" or "45+2" or just "90"
    match = re.match(r'(\d+)(?:\+(\d+))?', minute_str)
    if match:
        base = int(match.group(1))
        overtime = int(match.group(2)) if match.group(2) else 0
        return (base, overtime)
    
    return (0, 0)


def extract_announced_time(text: str) -> Optional[int]:
    """
    Extract announced added time from text.
    
    Simply extracts the number (digit or German word) that appears directly before "Minuten"
    when followed by context phrases indicating stoppage time.
    
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
    
    # Context phrases that indicate stoppage time announcements
    context_pattern = r'(?:nachgespielt|obendrauf|oben drauf|nachspielzeit|nachschlag|gibt es oben drauf|gibt es|wird noch nachgespielt|werden noch nachgespielt|tafel)'
    
    # Extract number (digit or German word) directly before "Minuten" followed by context
    # Examples: "5 Minuten Nachschlag", "Fünf Minuten gibt es", "4 Minuten Nachspielzeit"
    patterns = [
        # German number words: "Fünf Minuten Nachschlag", "Drei Minuten gibt es"
        rf'\b({"|".join(german_numbers.keys())})\s+minuten?\s+{context_pattern}',
        # Digits: "5 Minuten Nachschlag", "4 Minuten Nachspielzeit"
        rf'(\d+)\s+minuten?\s+{context_pattern}',
        # With "Min." abbreviation: "5 Min. Nachschlag"
        rf'(\d+)\s+min\.?\s+{context_pattern}',
        # Pattern 2: Number appears AFTER context phrase (e.g., "mehrere Minuten Nachspielzeit geben, nämlich 5")
        # "mehrere Minuten Nachspielzeit ... nämlich 5" or "Nachspielzeit geben, nämlich 5"
        rf'(?:mehrere|einige)\s+minuten?\s+{context_pattern}.*?(?:nämlich|genau|und zwar|konkret)\s+(\d+)',
        rf'{context_pattern}.*?(?:nämlich|genau|und zwar|konkret)\s+(\d+)',
        # Alternative: "Nachspielzeit: 4" or "Nachspielzeit 5"
        r'nachspielzeit[:\s]+(\d+)',
        # Fallback: "+4" or "+ 4" (less reliable)
        r'\+ ?(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            try:
                # Get the matched group (could be German word or digit)
                matched_text = match.group(1) if match.lastindex >= 1 else match.group(0)
                
                # If it's a German number word, convert it
                if matched_text.lower() in german_numbers:
                    return german_numbers[matched_text.lower()]
                
                # Otherwise, it's a digit
                num_str = matched_text.replace('+', '').strip()
                return int(num_str)
            except (ValueError, IndexError):
                continue
    
    return None


def is_substitution_event(text: str) -> bool:
    """
    Check if text describes a substitution event.
    
    Looks for "Wechsel", "kommt für" (comes on for), "verlässt den Platz" (leaves the pitch).
    Excludes "verletzt" (injured) unless accompanied by a change/substitution keyword
    to avoid confusing injuries with subs.
    
    Args:
        text: Event text to check
        
    Returns:
        True if text describes a substitution, False otherwise
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Substitution keywords - match scraper patterns
    substitution_keywords = [
        'spielerwechsel', 'wechsel', 'kommt für', 'verlässt den platz', 'auswechslung', 
        'ersetzt', 'geht vom feld', 'kommt in die partie', 'wechselt'
    ]
    
    # Check if any substitution keyword is present
    has_substitution_keyword = any(keyword in text_lower for keyword in substitution_keywords)
    
    if has_substitution_keyword:
        # If it has substitution keyword but also mentions injury, only count if it's clearly a substitution
        if 'verletzt' in text_lower:
            # Only count if there's a clear substitution context (player names, "für", etc.)
            if any(phrase in text_lower for phrase in ['kommt für', 'ersetzt', 'auswechslung']):
                return True
            return False
        return True
    
    return False


def extract_score_at_minute(score_timeline: Dict[str, List[int]], minute: int, inclusive: bool = False) -> Tuple[int, int]:
    """
    Extract score at a specific minute from score timeline.
    
    Args:
        score_timeline: Dictionary mapping minute strings (e.g., "1", "45", "90") to [home_score, away_score]
        minute: Target minute to get score for
        inclusive: If True, includes scores at exactly the minute. If False, gets last score before the minute.
        
    Returns:
        Tuple of (home_score, away_score), defaulting to (0, 0) if no score found
    """
    valid_scores = []
    
    for key, score in score_timeline.items():
        try:
            key_minute = int(key)
            if inclusive:
                if key_minute <= minute:
                    valid_scores.append((key_minute, score))
            else:
                if key_minute < minute:
                    valid_scores.append((key_minute, score))
        except (ValueError, TypeError):
            continue
    
    if valid_scores:
        # Get the last score (latest minute)
        latest_minute, latest_score = max(valid_scores, key=lambda x: x[0])
        return tuple(latest_score)
    
    return (0, 0)


def is_critical_event(text: str) -> bool:
    """
    Check if an event is critical and should be preserved during truncation.
    
    Critical events include goals, cards, substitutions, VAR, and injuries.
    
    Args:
        text: Event text to check
        
    Returns:
        True if event is critical, False otherwise
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Goals
    goal_keywords = ['tor', 'treffer', 'eingenetzt', 'torschütze']
    if any(keyword in text_lower for keyword in goal_keywords):
        return True
    
    # Cards
    card_keywords = ['gelb', 'rot', 'karte', 'verwarnung', 'gelbe karte', 'rote karte', 'gelb-rot']
    if any(keyword in text_lower for keyword in card_keywords):
        return True
    
    # Substitutions - match scraper patterns
    sub_keywords = ['spielerwechsel', 'wechsel', 'kommt für', 'verlässt', 'verlässt den platz', 
                    'auswechslung', 'ersetzt', 'geht vom feld', 'kommt in die partie', 'wechselt']
    if any(keyword in text_lower for keyword in sub_keywords):
        return True
    
    # VAR - use word boundaries for "VAR" to avoid matching player names like "Vargas"
    # Excluded: 'check' and 'überprüfung' (too generic, can appear in non-VAR contexts)
    if re.search(r'\bvar\b', text_lower):
        return True
    var_keywords = [
        'video-assistent', 'videoassistent', 'var-prüfung', 'var-check', 'kölner keller',
        'videocheck', 'videoüberprüfung', 'video'
    ]
    if any(keyword in text_lower for keyword in var_keywords):
        return True
    
    # Injuries - keywords to catch treatment patterns
    # Excluded: 'liegt'/'liegen' (can be in score contexts: "liegt 1:0 hinten")
    # Excluded: 'weitergehen' (can be in VAR contexts: "es geht weiter nach dem VAR")
    # Excluded: 'pausiert' (can be VAR-related: "das Spiel wurde aufgrund von einem VAR eingriff pausiert")
    # Excluded: 'unterbrochen' (too generic, can be for various reasons)
    # Excluded: 'zeigt an' (too generic, can be used in various contexts)
    # Excluded: 'ausgefallen' (too generic, can mean "missed" for various reasons, not just injury)
    injury_keywords = [
        'verletzt', 'verletzung', 'behandlung', 'behandelt werden', 'muss behandelt werden',
        'trage', 'steht wieder', 'liegen bleibt', 'eisspray',
        'wehgetan', 'weh getan', 'weh tut', 'sich wehgetan', 'sich weh getan',
        'behandelt werden muss', 'behandlung notwendig',
        'gesundheitliche probleme', 'muskuläre probleme',
        'sanitäter', 'brutales foul', 'verletzungspause'
    ]
    if any(keyword in text_lower for keyword in injury_keywords):
        return True
    
    # Disturbances/Interruptions - pitch invasions, police, security, pyrotechnics
    disturbance_keywords = [
        'platzsturm', 'polizist', 'polizisten', 'polizei', 'ordner', 'sicherheitskräfte',
        'feuerwerkskörper', 'feuerwerk', 'pyrotechnik', 'bengalo', 'rauchbombe',
        'spielabbruch', 'unterbrechung', 'spielunterbrechung', 'rasen stürmen',
        'feld stürmen', 'platz stürmen', 'Flitzer',
        'plätze stürmen', 'fans auf dem platz', 'zuschauer auf dem platz'
    ]
    if any(keyword in text_lower for keyword in disturbance_keywords):
        return True
    
    return False


def construct_bert_input(metadata_str: str, events_list: List[str], max_words: int = 400) -> str:
    """
    Construct BERT input with smart truncation that preserves critical events.
    
    Uses tail-priority truncation: removes earliest non-critical events first,
    preserving the end of the half which is most critical for stoppage time.
    
    Args:
        metadata_str: Metadata string (e.g., "[META] Home: ... [START]")
        events_list: List of formatted event strings (e.g., "[MIN_1] ...")
        max_words: Maximum word count (default 400, gives ~520 tokens, safe under 512 limit)
        
    Returns:
        Final BERT input string within word limit
    """
    # Initial build
    all_text = metadata_str + ' ' + ' '.join(events_list)
    word_count = len(all_text.split())
    
    # If within limit, return immediately
    if word_count <= max_words:
        return all_text.strip()
    
    # Need to truncate - start with events list
    remaining_events = events_list.copy()
    
    # Tail-priority truncation: remove earliest non-critical events
    while word_count > max_words and len(remaining_events) > 0:
        # Find the first non-critical event
        removed = False
        for i, event in enumerate(remaining_events):
            # Extract text from event string (format: "[MIN_X] text")
            # Try to extract the text part after the minute marker
            event_text = event
            if '] ' in event:
                # Extract text after "[MIN_X] "
                event_text = event.split('] ', 1)[1] if '] ' in event else event
            
            if not is_critical_event(event_text):
                # Remove this non-critical event
                remaining_events.pop(i)
                removed = True
                break
        
        # If all remaining events are critical, force remove the oldest one
        if not removed and len(remaining_events) > 0:
            remaining_events.pop(0)
        
        # Recalculate word count
        all_text = metadata_str + ' ' + ' '.join(remaining_events)
        word_count = len(all_text.split())
    
    return all_text.strip()


def count_feature_in_events(events: List[Dict[str, Any]], feature_type: str, half: Optional[str] = None) -> int:
    """
    Count occurrences of a specific feature type in events.
    
    Args:
        events: List of event dictionaries
        feature_type: Type of feature to count ('goals', 'cards', 'subs', 'var', 'injuries', 'disturbances')
        half: Which half ('1st' or '2nd') - used for filtering by minute range. If None, count all events.
        
    Returns:
        Count of feature occurrences
    """
    count = 0
    
    # Define minute ranges based on half (if specified)
    if half == '1st':
        min_minute, max_minute = 0, 45
    elif half == '2nd':
        min_minute, max_minute = 46, 90
    elif half is None:
        min_minute, max_minute = None, None  # No filtering
    else:
        raise ValueError(f"Invalid half: {half}. Must be '1st', '2nd', or None")
    
    # Keywords for each feature type
    # VAR keywords - exclude generic terms that can appear in other contexts
    var_keywords = ['VAR', 'Video-Assistent', 'Videoassistent', 'Kölner Keller', 'var-prüfung', 'var-check',
                    'Videocheck', 'Videoüberprüfung', 'Video']
    # Note: Excluded 'Überprüfung' and 'Check' (without VAR/Video prefix) as they're too generic
    
    # Injury keywords - exclude words that can be part of other events
    # Excluded: 'liegt'/'liegen' (can be in score contexts: "liegt 1:0 hinten")
    # Excluded: 'weitergehen' (can be in VAR contexts: "es geht weiter nach dem VAR")
    # Excluded: 'pausiert' (can be VAR-related: "das Spiel wurde aufgrund von einem VAR eingriff pausiert")
    # Excluded: 'unterbrochen' (too generic, can be for various reasons)
    # Excluded: 'zeigt an' (too generic, can be used in various contexts)
    # Excluded: 'ausgefallen' (too generic, can mean "missed" for various reasons, not just injury)
    injury_keywords = [
        'verletzt', 'verletzung', 'behandlung', 'behandelt werden', 'muss behandelt werden',
        'trage', 'steht wieder', 'liegen bleibt', 'eisspray',
        'wehgetan', 'weh getan', 'weh tut', 'sich wehgetan', 'sich weh getan',
        'behandelt werden muss', 'behandlung notwendig',
        'gesundheitliche probleme', 'muskuläre probleme',
        'sanitäter', 'brutales foul', 'verletzungspause'
    ]
    
    for event in events:
        minute_str = event.get('minute', '')
        minute_int = parse_minute_to_int(minute_str)
        
        # Skip if minute is outside the half range (only if half is specified)
        if half is not None:
            if minute_int is None or minute_int < min_minute or minute_int > max_minute:
                continue
        
        text = event.get('text', '').lower()
        
        if feature_type == 'goals':
            # Goals have score_at_event field
            if 'score_at_event' in event:
                count += 1
        
        elif feature_type == 'cards':
            # Cards have card_type field or mention cards in text (use card_type as primary, text as fallback)
            card_type = event.get('card_type')
            if card_type in ['Yellow', 'Red', 'YellowRed']:
                count += 1
            elif any(keyword in text for keyword in ['gelb', 'gelbe karte', 'rote karte', 'gelb-rot', 'gelb rot', 'gelb-rote karte', 'verwarnung']):
                count += 1
        
        elif feature_type == 'subs':
            # Substitutions detected via helper function
            if is_substitution_event(event.get('text', '')):
                count += 1
        
        elif feature_type == 'var':
            # VAR mentions in text (use word boundaries to avoid matching player names like "Vargas")
            text_lower = text.lower()
            # Check for "VAR" as standalone word first
            if re.search(r'\bvar\b', text_lower):
                count += 1
            else:
                # Check other VAR-specific keywords (exclude generic terms like "check", "überprüfung")
                for keyword in var_keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in text_lower:
                        count += 1
                        break
        
        elif feature_type == 'injuries':
            # Injury keywords in text
            if any(keyword.lower() in text for keyword in injury_keywords):
                count += 1
        
        elif feature_type == 'disturbances':
            # Disturbance/interruption keywords (pitch invasions, police, security, pyrotechnics)
            disturbance_keywords = [
                'platzsturm', 'polizist', 'polizisten', 'polizei', 'ordner', 'sicherheitskräfte',
                'feuerwerkskörper', 'feuerwerk', 'pyrotechnik', 'bengalo', 'rauchbombe',
                'spielabbruch', 'unterbrechung', 'spielunterbrechung', 'rasen stürmen',
                'feld stürmen', 'platz stürmen', 'flitzer',
                'plätze stürmen', 'fans auf dem platz', 'zuschauer auf dem platz'
            ]
            if any(keyword.lower() in text for keyword in disturbance_keywords):
                count += 1
    
    return count


def process_match_data_final(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw match data into structured features for econometric analysis.
    
    This function implements strict phase separation and leakage prevention:
    - Separates regular time events from overtime events
    - Extracts targets with proper fallback logic
    - Engineers features for baseline regression
    - Constructs BERT input text (excluding overtime events)
    
    Args:
        raw_data: Raw match JSON data dictionary
        
    Returns:
        Processed match data dictionary with structured features
    """
    match_id = raw_data.get('match_id', '')
    season = raw_data.get('metadata', {}).get('season', '')
    metadata = raw_data.get('metadata', {})
    targets_raw = raw_data.get('targets', {})
    ticker_data = raw_data.get('ticker_data', [])
    score_timeline_raw = raw_data.get('score_timeline', {})
    match_info = raw_data.get('match_info', {})
    yellow_cards_raw = raw_data.get('yellow_cards', {})
    red_cards_raw = raw_data.get('red_cards', {})
    yellow_red_cards_raw = raw_data.get('yellow_red_cards', {})
    substitutes_raw = raw_data.get('substitutes', {})
    
    # ========== VALIDATION: Check for required cutoff markers ==========
    cutoff_markers = {
        'Anpfiff': False,
        'Halbzeitpfiff': False,
        'Anpfiff 2. Hälfte': False,
        'Abpfiff': False
    }
    
    for event in ticker_data:
        text = event.get('text', '').lower()
        if 'anpfiff' in text and '2' not in text and 'hälfte' not in text:
            cutoff_markers['Anpfiff'] = True
        elif 'halbzeitpfiff' in text:
            cutoff_markers['Halbzeitpfiff'] = True
        elif 'anpfiff 2' in text or ('anpfiff' in text and '2' in text and 'hälfte' in text):
            cutoff_markers['Anpfiff 2. Hälfte'] = True
        elif 'abpfiff' in text:
            cutoff_markers['Abpfiff'] = True
    
    # Add validation flag if any marker is missing
    missing_markers = [marker for marker, found in cutoff_markers.items() if not found]
    if missing_markers:
        logger.warning(f"Match {match_id}: Missing cutoff markers: {missing_markers}")
    
    # ========== VALIDATION: Check score_timeline matches final_score ==========
    final_score_valid = True
    if metadata.get('final_score'):
        try:
            # Parse final_score (e.g., "2:1")
            final_parts = metadata['final_score'].split(':')
            if len(final_parts) == 2:
                final_home = int(final_parts[0])
                final_away = int(final_parts[1])
                
                # Get last entry from score_timeline (including overtime entries like "90+3")
                if score_timeline_raw:
                    # Sort all minute keys properly (handles "90+3" > "90" > "87")
                    sorted_minutes = sorted(score_timeline_raw.keys(), key=parse_minute_for_sorting)
                    last_minute_key = sorted_minutes[-1]
                    last_score = score_timeline_raw[last_minute_key]
                    
                    if tuple(last_score) != (final_home, final_away):
                        logger.warning(f"Match {match_id}: Last score_timeline entry {last_score} (at {last_minute_key}) doesn't match final_score {final_home}:{final_away}")
                        final_score_valid = False
                else:
                    logger.warning(f"Match {match_id}: score_timeline is empty but final_score is {metadata['final_score']}")
                    final_score_valid = False
        except (ValueError, KeyError) as e:
            logger.warning(f"Match {match_id}: Error validating final_score: {e}")
            final_score_valid = False
    
    # ========== FIX OVERTIME MINUTES in yellow_cards, red_cards, substitutes, score_timeline ==========
    # The scraper sometimes records overtime events as "90" or "45" instead of "90+x" or "45+x"
    # Fix this by checking ticker_data for the actual minute strings
    
    def fix_overtime_minutes(event_dict: Dict[str, int], ticker_data: List[Dict], event_type: str) -> Dict[str, int]:
        """
        Fix overtime minutes in event dictionaries by checking ticker_data for actual minute strings.
        
        Args:
            event_dict: Dictionary with minute keys (e.g., {"90": 1})
            ticker_data: List of ticker events
            event_type: Type of event ('card', 'sub', 'score')
            
        Returns:
            Fixed dictionary with correct overtime minute keys
        """
        fixed_dict = {}
        
        # For each entry in the original dict
        for minute_key, count in event_dict.items():
            minute_int = int(minute_key) if minute_key.isdigit() else None
            
            if minute_int is None:
                # Already has overtime format (e.g., "90+2"), keep as is
                fixed_dict[minute_key] = count
                continue
            
            # Check if this is a boundary minute (45 or 90) that might be overtime
            if minute_int in [45, 90]:
                # Look through ticker_data to find events with actual overtime minutes
                found_overtime_minutes = set()
                
                for event in ticker_data:
                    event_minute_str = event.get('minute', '')
                    event_minute_int = parse_minute_to_int(event_minute_str)
                    is_overtime = '+' in event_minute_str if event_minute_str else False
                    
                    # Only consider overtime events at the boundary (45+ or 90+)
                    if event_minute_int == minute_int and is_overtime:
                        if event_type == 'card':
                            # Check if this event has a card
                            if event.get('card_type') in ['Yellow', 'Red', 'YellowRed']:
                                found_overtime_minutes.add(event_minute_str)
                        elif event_type == 'sub':
                            # Check if this is a substitution
                            if is_substitution_event(event.get('text', '')):
                                found_overtime_minutes.add(event_minute_str)
                        elif event_type == 'score':
                            # Check if this is a goal
                            if 'score_at_event' in event:
                                found_overtime_minutes.add(event_minute_str)
                
                # If we found overtime minutes, use them; otherwise keep original
                if found_overtime_minutes:
                    # Distribute the count across found overtime minutes
                    # For simplicity, if count is 1, use the first found minute
                    # If count > 1, we'd need more logic, but typically it's 1
                    for ot_minute in sorted(found_overtime_minutes):
                        fixed_dict[ot_minute] = 1
                else:
                    # No overtime found, keep original
                    fixed_dict[minute_key] = count
            else:
                # Not a boundary minute, keep as is
                fixed_dict[minute_key] = count
        
        return fixed_dict
    
    # Fix each dictionary
    yellow_cards = fix_overtime_minutes(yellow_cards_raw, ticker_data, 'card')
    red_cards = fix_overtime_minutes(red_cards_raw, ticker_data, 'card')
    yellow_red_cards = fix_overtime_minutes(yellow_red_cards_raw, ticker_data, 'card')
    substitutes = fix_overtime_minutes(substitutes_raw, ticker_data, 'sub')
    
    # For score_timeline, we need to be more careful - preserve all entries but fix overtime ones
    score_timeline = {}
    for minute_key, score in score_timeline_raw.items():
        minute_int = int(minute_key) if minute_key.isdigit() else None
        
        if minute_int in [45, 90]:
            # Check ticker_data for goals at overtime minutes
            found_overtime_minutes = []
            for event in ticker_data:
                event_minute_str = event.get('minute', '')
                event_minute_int = parse_minute_to_int(event_minute_str)
                is_overtime = '+' in event_minute_str if event_minute_str else False
                
                if event_minute_int == minute_int and is_overtime and 'score_at_event' in event:
                    if event.get('score_at_event') == score:
                        found_overtime_minutes.append((event_minute_str, event.get('score_at_event')))
            
            if found_overtime_minutes:
                # Use the overtime minute(s) instead
                for ot_minute, ot_score in found_overtime_minutes:
                    score_timeline[ot_minute] = ot_score
            else:
                # Keep original
                score_timeline[minute_key] = score
        else:
            score_timeline[minute_key] = score
    
    # Determine if this is a corona/ghost game season
    corona_seasons = ['2019-20', '2020-21', '2021-22']
    is_corona_season = season in corona_seasons
    
    # Get ghost game flag from metadata (if available), otherwise infer from season
    is_ghost_game = metadata.get('is_ghost_game', False)
    # If not explicitly set, use season-based detection
    if is_ghost_game is False and is_corona_season:
        # In corona seasons, most games were ghost games, but check attendance as well
        attendance = metadata.get('attendance') or metadata.get('spectators')
        # Only mark as ghost game if we have attendance data and it's < 1500
        # If attendance is None/missing, set is_ghost_game to None (unknown)
        if attendance is not None:
            is_ghost_game = attendance < 1500  # Ghost game if attendance < 1500
        else:
            is_ghost_game = None  # Unknown when attendance is missing
    
    # Initialize output structure
    result = {
        'match_id': match_id,
        'season': season,
        'metadata': {
            'home': metadata.get('home_team', ''),
            'away': metadata.get('away_team', ''),
            'attendance': metadata.get('attendance') or metadata.get('spectators'),
            'final_score': metadata.get('final_score', ''),
            'matchday': metadata.get('matchday'),
            'is_sold_out': metadata.get('is_sold_out', False),
            'is_ghost_game': is_ghost_game,
            'is_corona_season': is_corona_season,
            'stats': match_info.get('stats', {})
        },
        'score_timeline': score_timeline,
        'yellow_cards': yellow_cards,
        'red_cards': red_cards,
        'yellow_red_cards': yellow_red_cards,
        'substitutes': substitutes,
        'flags': {
            # Target imputation flags (will be set during target extraction)
            'is_inferred_zero_45': False,
            'is_inferred_zero_90': False,
            'target_missing_45': False,
            'target_missing_90': False,
            'is_imputed_actual_45': False,
            'is_imputed_actual_90': False,
            'is_imputed_announced_45': False,
            'is_imputed_announced_90': False,
            # Validation flags
            'missing_cutoff_markers': missing_markers if missing_markers else None,
            'score_timeline_valid': final_score_valid
        },
        'targets': {},
        'features_regular': {},
        'features_overtime': {},
        'bert_input_45': '',
        'bert_input_90': ''
    }
    
    # ========== 1. TARGET EXTRACTION (Robust Imputation) ==========
    
    # Step A: Preserve Raw Data
    targets_raw_stored = {
        'announced_time_45': targets_raw.get('announced_time_45'),
        'announced_time_90': targets_raw.get('announced_time_90'),
        'actual_played_45': targets_raw.get('actual_played_45'),
        'actual_played_90': targets_raw.get('actual_played_90')
    }
    
    # Extract raw values (preserve None if missing)
    actual_45_raw = targets_raw.get('actual_played_45')
    actual_90_raw = targets_raw.get('actual_played_90')
    announced_45_raw = targets_raw.get('announced_time_45')
    announced_90_raw = targets_raw.get('announced_time_90')
    
    # Step B: Standard Extraction (Announced Time)
    # Try metadata first, then regex extraction from ticker_data
    
    if announced_45_raw is None:
        # Regex extraction from ticker_data (minutes 40-45)
        for event in ticker_data:
            minute_str = event.get('minute', '')
            minute_int = parse_minute_to_int(minute_str)
            if minute_int is not None and 40 <= minute_int <= 45:
                text = event.get('text', '')
                # Ignore lines containing "Wechsel", "Auswechslung", or player names
                text_lower = text.lower()
                if any(exclude in text_lower for exclude in ['wechsel', 'auswechslung']):
                    continue
                # Check for player name patterns (capitalized words, common patterns)
                if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text):
                    continue
                extracted = extract_announced_time(text)
                if extracted is not None:
                    announced_45_raw = extracted
                    break
    
    if announced_90_raw is None:
        # Regex extraction from ticker_data (minutes 85-90)
        for event in ticker_data:
            minute_str = event.get('minute', '')
            minute_int = parse_minute_to_int(minute_str)
            if minute_int is not None and 85 <= minute_int <= 90:
                text = event.get('text', '')
                text_lower = text.lower()
                if any(exclude in text_lower for exclude in ['wechsel', 'auswechslung']):
                    continue
                if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text):
                    continue
                extracted = extract_announced_time(text)
                if extracted is not None:
                    announced_90_raw = extracted
                    break
    
    # Step C: Imputation Logic (The "Safe" Fill) - Process each half separately
    # Helper function to apply imputation logic for one half
    def apply_imputation_logic(announced_raw, actual_raw, half_suffix):
        """Apply imputation logic for one half."""
        announced = announced_raw
        actual = actual_raw
        
        # Convert None to None (keep as None for now, we'll handle imputation)
        if actual_raw is None:
            actual = None
        else:
            actual = int(actual_raw) if actual_raw is not None else None
            
        if announced_raw is not None:
            announced = int(announced_raw)
        
        # Scenario 1: Missing Board, Short Game (Announced is None, Actual <= 1)
        if announced is None and actual is not None and actual <= 1:
            announced = 0
            result['flags'][f'is_inferred_zero_{half_suffix}'] = True
        
        # Scenario 2: Missing Board, Long Game (Announced is None, Actual > 1)
        elif announced is None and actual is not None and actual > 1:
            announced = None  # Keep as None
            result['flags'][f'target_missing_{half_suffix}'] = True
        
        # Scenario 3: Missing Whistle, Valid Board (Actual is None, Announced > 0)
        elif actual is None and announced is not None and announced > 0:
            actual = announced
            result['flags'][f'is_imputed_actual_{half_suffix}'] = True
        
        # Scenario 4: Both Missing (Actual is None, Announced is None)
        elif actual is None and announced is None:
            actual = 0
            announced = 0
            result['flags'][f'is_inferred_zero_{half_suffix}'] = True
        
        # Ensure integers (convert if not None)
        if actual is not None:
            actual = int(actual)
        if announced is not None:
            announced = int(announced)
            
        return announced, actual
    
    # Apply imputation for both halves
    announced_45, actual_45 = apply_imputation_logic(announced_45_raw, actual_45_raw, '45')
    announced_90, actual_90 = apply_imputation_logic(announced_90_raw, actual_90_raw, '90')
    
    # Step D: Handle case where Announced > Actual (Scenario 5: Negative Excess)
    # If announced is bigger than played, we probably didn't capture the actual event,
    # so use the announced value (set actual = announced)
    # Status: VALID for both models (actual is corrected to announced)
    if announced_45 is not None and actual_45 is not None and announced_45 > actual_45:
        actual_45 = announced_45
        result['flags']['is_imputed_announced_45'] = True
    
    if announced_90 is not None and actual_90 is not None and announced_90 > actual_90:
        actual_90 = announced_90
        result['flags']['is_imputed_announced_90'] = True
    
    # Step E: Final Checks and Excess Time Calculation
    # Calculate Excess_Time (Actual - Announced) only if both exist and target is not missing
    excess_45 = None
    excess_90 = None
    
    if not result['flags']['target_missing_45'] and actual_45 is not None and announced_45 is not None:
        excess_45 = actual_45 - announced_45
    elif result['flags']['target_missing_45']:
        excess_45 = None  # Explicitly None if target is missing
    
    if not result['flags']['target_missing_90'] and actual_90 is not None and announced_90 is not None:
        excess_90 = actual_90 - announced_90
    elif result['flags']['target_missing_90']:
        excess_90 = None  # Explicitly None if target is missing
    
    # Store targets
    result['targets'] = {
        'announced_45': announced_45,
        'actual_45': actual_45,
        'excess_45': excess_45,
        'announced_90': announced_90,
        'actual_90': actual_90,
        'excess_90': excess_90,
        'targets_raw': targets_raw_stored  # Preserve raw values
    }
    
    # ========== VALIDATION: Check for required cutoff markers ==========
    cutoff_markers = {
        'Anpfiff': False,
        'Halbzeitpfiff': False,
        'Anpfiff 2. Hälfte': False,
        'Abpfiff': False
    }
    
    for event in ticker_data:
        text = event.get('text', '').lower()
        if 'anpfiff' in text and '2' not in text and 'hälfte' not in text:
            cutoff_markers['Anpfiff'] = True
        elif 'halbzeitpfiff' in text:
            cutoff_markers['Halbzeitpfiff'] = True
        elif 'anpfiff 2' in text or ('anpfiff' in text and '2' in text):
            cutoff_markers['Anpfiff 2. Hälfte'] = True
        elif 'abpfiff' in text:
            cutoff_markers['Abpfiff'] = True
    
    # Add validation flag if any marker is missing
    missing_markers = [marker for marker, found in cutoff_markers.items() if not found]
    if missing_markers:
        logger.warning(f"Match {match_id}: Missing cutoff markers: {missing_markers}")
    
    # ========== VALIDATION: Check score_timeline matches final_score ==========
    final_score_valid = True
    if metadata.get('final_score'):
        try:
            # Parse final_score (e.g., "2:1")
            final_parts = metadata['final_score'].split(':')
            if len(final_parts) == 2:
                final_home = int(final_parts[0])
                final_away = int(final_parts[1])
                
                # Get last entry from score_timeline (including overtime entries like "90+3")
                if score_timeline:
                    # Sort all minute keys properly (handles "90+3" > "90" > "87")
                    sorted_minutes = sorted(score_timeline.keys(), key=parse_minute_for_sorting)
                    last_minute_key = sorted_minutes[-1]
                    last_score = score_timeline[last_minute_key]
                    
                    if tuple(last_score) != (final_home, final_away):
                        logger.warning(f"Match {match_id}: Last score_timeline entry {last_score} (at {last_minute_key}) doesn't match final_score {final_home}:{final_away}")
                        final_score_valid = False
                else:
                    logger.warning(f"Match {match_id}: score_timeline is empty but final_score is {metadata['final_score']}")
                    final_score_valid = False
        except (ValueError, KeyError) as e:
            logger.warning(f"Match {match_id}: Error validating final_score: {e}")
            final_score_valid = False
    
    # ========== FIX OVERTIME MINUTES in yellow_cards, red_cards, substitutes, score_timeline ==========
    # The scraper sometimes records overtime events as "90" or "45" instead of "90+x" or "45+x"
    # Fix this by checking ticker_data for the actual minute strings
    
    def fix_overtime_minutes(event_dict: Dict[str, int], ticker_data: List[Dict], event_type: str) -> Dict[str, int]:
        """
        Fix overtime minutes in event dictionaries by checking ticker_data for actual minute strings.
        
        Args:
            event_dict: Dictionary with minute keys (e.g., {"90": 1})
            ticker_data: List of ticker events
            event_type: Type of event ('card', 'sub', 'score')
            
        Returns:
            Fixed dictionary with correct overtime minute keys
        """
        fixed_dict = {}
        
        # For each entry in the original dict
        for minute_key, count in event_dict.items():
            minute_int = int(minute_key) if minute_key.isdigit() else None
            
            if minute_int is None:
                # Already has overtime format (e.g., "90+2"), keep as is
                fixed_dict[minute_key] = count
                continue
            
            # Check if this is a boundary minute (45 or 90) that might be overtime
            if minute_int in [45, 90]:
                # Look through ticker_data to find events with actual overtime minutes
                found_overtime_minutes = set()
                
                for event in ticker_data:
                    event_minute_str = event.get('minute', '')
                    event_minute_int = parse_minute_to_int(event_minute_str)
                    is_overtime = '+' in event_minute_str if event_minute_str else False
                    
                    # Only consider overtime events at the boundary (45+ or 90+)
                    if event_minute_int == minute_int and is_overtime:
                        if event_type == 'card':
                            # Check if this event has a card
                            if event.get('card_type') in ['Yellow', 'Red', 'YellowRed']:
                                found_overtime_minutes.add(event_minute_str)
                        elif event_type == 'sub':
                            # Check if this is a substitution
                            if is_substitution_event(event.get('text', '')):
                                found_overtime_minutes.add(event_minute_str)
                        elif event_type == 'score':
                            # Check if this is a goal
                            if 'score_at_event' in event:
                                found_overtime_minutes.add(event_minute_str)
                
                # If we found overtime minutes, use them; otherwise keep original
                if found_overtime_minutes:
                    # Distribute the count across found overtime minutes
                    # For simplicity, if count is 1, use the first found minute
                    # If count > 1, we'd need more logic, but typically it's 1
                    for ot_minute in sorted(found_overtime_minutes):
                        fixed_dict[ot_minute] = 1
                else:
                    # No overtime found, keep original
                    fixed_dict[minute_key] = count
            else:
                # Not a boundary minute, keep as is
                fixed_dict[minute_key] = count
        
        return fixed_dict
    
    # Fix each dictionary
    yellow_cards = fix_overtime_minutes(yellow_cards, ticker_data, 'card')
    red_cards = fix_overtime_minutes(red_cards, ticker_data, 'card')
    yellow_red_cards = fix_overtime_minutes(yellow_red_cards, ticker_data, 'card')
    substitutes = fix_overtime_minutes(substitutes, ticker_data, 'sub')
    
    # For score_timeline, we need to be more careful - preserve all entries but fix overtime ones
    fixed_score_timeline = {}
    for minute_key, score in score_timeline.items():
        minute_int = int(minute_key) if minute_key.isdigit() else None
        
        if minute_int in [45, 90]:
            # Check ticker_data for goals at overtime minutes
            found_overtime_minutes = []
            for event in ticker_data:
                event_minute_str = event.get('minute', '')
                event_minute_int = parse_minute_to_int(event_minute_str)
                is_overtime = '+' in event_minute_str if event_minute_str else False
                
                if event_minute_int == minute_int and is_overtime and 'score_at_event' in event:
                    if event.get('score_at_event') == score:
                        found_overtime_minutes.append((event_minute_str, event.get('score_at_event')))
            
            if found_overtime_minutes:
                # Use the overtime minute(s) instead
                for ot_minute, ot_score in found_overtime_minutes:
                    fixed_score_timeline[ot_minute] = ot_score
            else:
                # Keep original
                fixed_score_timeline[minute_key] = score
        else:
            fixed_score_timeline[minute_key] = score
    
    score_timeline = fixed_score_timeline
    
    # ========== 2. EVENT PHASE SEPARATION ==========
    
    phase_1_events = []  # 1st Half Regular (before Halbzeitpfiff, minute <= 45)
    phase_2_events = []  # Halftime Gap (between Halbzeitpfiff and Anpfiff 2. Hälfte)
    phase_3_events = []  # 2nd Half Regular (after Anpfiff 2. Hälfte, minute <= 90)
    phase_4_overtime_1st = []  # Overtime 1st half (minute > 45 before Halftime whistle)
    phase_4_overtime_2nd = []  # Overtime 2nd half (minute > 90 after restart)
    
    halftime_started = False
    second_half_started = False
    
    for event in ticker_data:
        minute_str = event.get('minute', '')
        minute_int = parse_minute_to_int(minute_str)
        text = event.get('text', '')
        
        # Check if minute string indicates overtime (contains "+")
        is_overtime = '+' in minute_str if minute_str else False
        
        # Check for phase markers
        if 'halbzeitpfiff' in text.lower():
            halftime_started = True
            continue
        if 'anpfiff 2. hälfte' in text.lower() or 'anpfiff 2' in text.lower():
            second_half_started = True
            continue
        
        if not halftime_started:
            # Phase 1: 1st Half Regular (minute <= 45, not overtime)
            if minute_int is not None and minute_int <= 45 and not is_overtime:
                phase_1_events.append(event)
            elif (minute_int is not None and minute_int > 45) or is_overtime:
                # Phase 4: Overtime 1st half (minute > 45 or contains "+")
                phase_4_overtime_1st.append(event)
        elif halftime_started and not second_half_started:
            # Phase 2: Halftime Gap
            phase_2_events.append(event)
        elif second_half_started:
            # Phase 3: 2nd Half Regular (minute <= 90, not overtime)
            if minute_int is not None and minute_int <= 90 and not is_overtime:
                phase_3_events.append(event)
            elif (minute_int is not None and minute_int > 90) or is_overtime:
                # Phase 4: Overtime 2nd half (minute > 90 or contains "+")
                phase_4_overtime_2nd.append(event)
    
    # Build halftime context from Phase 2
    halftime_context = ' '.join([event.get('text', '') for event in phase_2_events])
    
    # ========== 3. FEATURE ENGINEERING ==========
    
    # A. Regular Features (minutes 0-45 & 46-90)
    features_regular = {
        'goals_1st': count_feature_in_events(phase_1_events, 'goals', '1st'),
        'goals_2nd': count_feature_in_events(phase_3_events, 'goals', '2nd'),
        'subs_1st': count_feature_in_events(phase_1_events, 'subs', '1st'),
        'subs_2nd': count_feature_in_events(phase_3_events, 'subs', '2nd'),
        'cards_1st': count_feature_in_events(phase_1_events, 'cards', '1st'),
        'cards_2nd': count_feature_in_events(phase_3_events, 'cards', '2nd'),
        'var_1st': count_feature_in_events(phase_1_events, 'var', '1st'),
        'var_2nd': count_feature_in_events(phase_3_events, 'var', '2nd'),
        'injuries_1st': count_feature_in_events(phase_1_events, 'injuries', '1st'),
        'injuries_2nd': count_feature_in_events(phase_3_events, 'injuries', '2nd'),
        'disturbances_1st': count_feature_in_events(phase_1_events, 'disturbances', '1st'),
        'disturbances_2nd': count_feature_in_events(phase_3_events, 'disturbances', '2nd'),
    }
    
    # Get score at minute 90 (exclusive, before overtime)
    home_score_90, away_score_90 = extract_score_at_minute(score_timeline, 90, inclusive=False)
    features_regular['home_score_90'] = home_score_90
    features_regular['away_score_90'] = away_score_90
    
    result['features_regular'] = features_regular
    
    # B. Overtime Features (minutes 45+ & 90+)
    # Count features ONLY in Phase 4 events (no minute filtering needed since they're already filtered)
    features_overtime = {
        'ot_goals_45': count_feature_in_events(phase_4_overtime_1st, 'goals', None),
        'ot_goals_90': count_feature_in_events(phase_4_overtime_2nd, 'goals', None),
        'ot_subs_45': count_feature_in_events(phase_4_overtime_1st, 'subs', None),
        'ot_subs_90': count_feature_in_events(phase_4_overtime_2nd, 'subs', None),
        'ot_cards_45': count_feature_in_events(phase_4_overtime_1st, 'cards', None),
        'ot_cards_90': count_feature_in_events(phase_4_overtime_2nd, 'cards', None),
        'ot_var_45': count_feature_in_events(phase_4_overtime_1st, 'var', None),
        'ot_var_90': count_feature_in_events(phase_4_overtime_2nd, 'var', None),
        'ot_injuries_45': count_feature_in_events(phase_4_overtime_1st, 'injuries', None),
        'ot_injuries_90': count_feature_in_events(phase_4_overtime_2nd, 'injuries', None),
        'ot_disturbances_45': count_feature_in_events(phase_4_overtime_1st, 'disturbances', None),
        'ot_disturbances_90': count_feature_in_events(phase_4_overtime_2nd, 'disturbances', None),
    }
    
    result['features_overtime'] = features_overtime
    
    # ========== 4. TEXT CONSTRUCTION FOR BERT ==========
    
    # CRITICAL: Exclude all Phase 4 (Overtime) events to prevent leakage
    
    # Build prematch text (events with empty minute before first minute >= 1)
    prematch_text_parts = []
    for event in ticker_data:
        minute_str = event.get('minute', '')
        if not minute_str or minute_str.strip() == '':
            text = event.get('text', '')
            if 'anpfiff' not in text.lower():
                prematch_text_parts.append(text)
            else:
                break  # Stop at "Anpfiff"
        else:
            minute_int = parse_minute_to_int(minute_str)
            if minute_int is not None and minute_int >= 1:
                break
    
    prematch_text = ' '.join(prematch_text_parts)
    
    # Build Phase 1 events list (1st Half Regular only) - sorted by minute
    phase_1_events_sorted = sorted(phase_1_events, key=lambda e: (
        parse_minute_to_int(e.get('minute', '')) or 0,
        phase_1_events.index(e)  # Preserve original order for same minute
    ))
    phase_1_events_list = []
    for event in phase_1_events_sorted:
        minute_str = event.get('minute', '')
        text = event.get('text', '')
        if minute_str:
            phase_1_events_list.append(f"[MIN_{minute_str}] {text}")
    
    # Build Phase 3 events list (2nd Half Regular only) - sorted by minute
    phase_3_events_sorted = sorted(phase_3_events, key=lambda e: (
        parse_minute_to_int(e.get('minute', '')) or 0,
        phase_3_events.index(e)  # Preserve original order for same minute
    ))
    phase_3_events_list = []
    for event in phase_3_events_sorted:
        minute_str = event.get('minute', '')
        text = event.get('text', '')
        if minute_str:
            phase_3_events_list.append(f"[MIN_{minute_str}] {text}")
    
    # Get score at 45 for 2nd half input
    home_score_45, away_score_45 = extract_score_at_minute(score_timeline, 45, inclusive=True)
    
    # Construct BERT input strings with smart truncation
    # IMPORTANT: Drop context (prematch/halftime) BEFORE truncating events
    home = result['metadata']['home']
    away = result['metadata']['away']
    att = result['metadata']['attendance']
    corona_flag = "Corona" if is_corona_season else "Normal"
    cards_1st = features_regular['cards_1st']
    subs_1st = features_regular['subs_1st']
    var_1st = features_regular['var_1st']
    injuries_1st = features_regular['injuries_1st']
    
    # 1st Half BERT input
    # Step 1: Build with prematch context and check word count
    metadata_45_with_pre = f"[META] Home: {home} Away: {away} Attendance: {att} Corona: {corona_flag} [PRE] {prematch_text} [START]"
    test_with_pre = metadata_45_with_pre + ' ' + ' '.join(phase_1_events_list)
    word_count_with_pre = len(test_with_pre.split())
    
    # Step 2: If too long, drop prematch context FIRST (before truncating events)
    if word_count_with_pre > 400:
        metadata_45 = f"[META] Home: {home} Away: {away} Attendance: {att} Corona: {corona_flag} [START]"
    else:
        metadata_45 = metadata_45_with_pre
    
    # Step 3: Now truncate events if needed
    bert_input_45_str = construct_bert_input(metadata_45, phase_1_events_list, max_words=400)
    result['bert_input_45'] = bert_input_45_str
    
    # 2nd Half BERT input
    # Step 1: Build with halftime context and check word count
    # Build STATS string (only include stats if > 0 to save tokens)
    stats_parts = [f"Cards_1st: {cards_1st}", f"Subs_1st: {subs_1st}"]
    if var_1st > 0:
        stats_parts.append(f"VAR_1st: {var_1st}")
    if injuries_1st > 0:
        stats_parts.append(f"Injuries_1st: {injuries_1st}")
    stats_str = " ".join(stats_parts)
    
    metadata_90_with_half = (
        f"[META] Home: {home} Away: {away} Score_45: {home_score_45}:{away_score_45} Corona: {corona_flag} "
        f"[STATS_1ST] {stats_str} [HALF] {halftime_context} [START]"
    )
    test_with_half = metadata_90_with_half + ' ' + ' '.join(phase_3_events_list)
    word_count_with_half = len(test_with_half.split())
    
    # Step 2: If too long, drop halftime context FIRST (before truncating events)
    if word_count_with_half > 400:
        metadata_90 = (
            f"[META] Home: {home} Away: {away} Score_45: {home_score_45}:{away_score_45} Corona: {corona_flag} "
            f"[STATS_1ST] {stats_str} [START]"
        )
    else:
        metadata_90 = metadata_90_with_half
    
    # Step 3: Now truncate events if needed
    bert_input_90_str = construct_bert_input(metadata_90, phase_3_events_list, max_words=400)
    result['bert_input_90'] = bert_input_90_str
    
    # Calculate token counts (simple word-based approximation)
    # BERT uses WordPiece tokenization, but word count is a reasonable approximation
    def estimate_tokens(text: str) -> int:
        """Estimate token count using word splitting (rough approximation for BERT)."""
        if not text:
            return 0
        # Split on whitespace and count words (BERT typically has ~1.3 tokens per word)
        words = text.split()
        return len(words)
    
    result['bert_input_45_tokens'] = estimate_tokens(bert_input_45_str)
    result['bert_input_90_tokens'] = estimate_tokens(bert_input_90_str)
    
    # Build overtime ticker event data
    # Overtime 1st half (45+)
    ot_45_events = []
    for event in phase_4_overtime_1st:
        minute_str = event.get('minute', '')
        text = event.get('text', '')
        if minute_str and text:
            ot_45_events.append({
                'minute': minute_str,
                'text': text
            })
    
    # Overtime 2nd half (90+)
    ot_90_events = []
    for event in phase_4_overtime_2nd:
        minute_str = event.get('minute', '')
        text = event.get('text', '')
        if minute_str and text:
            ot_90_events.append({
                'minute': minute_str,
                'text': text
            })
    
    result['overtime_ticker_45'] = ot_45_events
    result['overtime_ticker_90'] = ot_90_events
    
    return result


def process_all_matches(raw_dir: Path, processed_dir: Path, season_filter: Optional[str] = None) -> None:
    """
    Process all match files from raw directory to processed directory.
    
    Args:
        raw_dir: Path to data/raw directory
        processed_dir: Path to data/processed directory
        season_filter: Optional season string to filter (e.g., "2022-23")
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    
    # Find all season directories
    season_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith('season_')]
    
    if season_filter:
        season_dirs = [d for d in season_dirs if season_filter in d.name]
    
    total_matches = 0
    successful = 0
    failed = 0
    
    for season_dir in sorted(season_dirs):
        season = season_dir.name.replace('season_', '')
        logger.info(f"Processing season: {season}")
        
        # Create corresponding processed directory
        processed_season_dir = processed_dir / season_dir.name
        processed_season_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all match JSON files
        match_files = list(season_dir.glob('match_*.json'))
        
        for match_file in sorted(match_files):
            total_matches += 1
            match_id = match_file.stem.replace('match_', '')
            output_file = processed_season_dir / match_file.name
            
            # Skip if already processed (unless we want to reprocess)
            if output_file.exists():
                logger.debug(f"Skipping already processed: {match_id}")
                continue
            
            try:
                # Load raw data
                with open(match_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # Process match
                processed_data = process_match_data_final(raw_data)
                
                # Save processed data
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                successful += 1
                if successful % 50 == 0:
                    logger.info(f"Processed {successful} matches...")
            
            except Exception as e:
                failed += 1
                logger.error(f"Failed to process {match_file.name}: {e}")
                continue
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed out of {total_matches} total")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Process raw match data into structured features')
    parser.add_argument(
        '--input',
        type=str,
        help='Path to a specific input JSON file (if not provided, processes all files)'
    )
    parser.add_argument(
        '--season',
        type=str,
        help='Filter by season (e.g., "2022-23") when processing all files'
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='data/raw',
        help='Path to raw data directory (default: data/raw)'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Path to processed data directory (default: data/processed)'
    )
    
    args = parser.parse_args()
    
    if args.input:
        # Process single file
        input_file = Path(args.input)
        output_file = Path(args.processed_dir) / input_file.parent.name / input_file.name
        
        logger.info(f"Processing single file: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_data = process_match_data_final(raw_data)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved processed data to: {output_file}")
    else:
        # Process all files
        process_all_matches(
            raw_dir=Path(args.raw_dir),
            processed_dir=Path(args.processed_dir),
            season_filter=args.season
        )


if __name__ == '__main__':
    main()

