"""
Truncate BERT inputs that exceed 512 tokens by removing events (earliest to latest).

This script processes matches that have bert_input_45_tokens or bert_input_90_tokens > 512.
It removes events one by one (earliest to latest) until the token count is <= 512.
Only events are removed, not metadata (season token, team names, attendance, score, corona flag).

Usage:
    python src/data/truncate_over_512_tokens.py [--processed-dir data/processed] [--dry-run]
    
    # Process only specific matches from JSON file
    python src/data/truncate_over_512_tokens.py --matches-file data/results/matches_over_512_tokens.json
    
    # Dry run to see what would be changed
    python src/data/truncate_over_512_tokens.py --dry-run

Output:
    Updates processed match files in place, removing events until token count <= 512.
    Logs all changes made.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained('distilbert-base-german-cased')
    TOKENIZER_AVAILABLE = True
except Exception as e:
    logger.warning(f"Could not load tokenizer: {e}. Will use word count approximation.")
    TOKENIZER = None
    TOKENIZER_AVAILABLE = False


def parse_minute_to_int(minute_str: str) -> Optional[int]:
    """
    Parse minute string to integer.
    
    Handles formats like:
    - "20" -> 20
    - "45+2" -> 45
    - "90+4" -> 90
    
    Args:
        minute_str: Minute string
        
    Returns:
        Integer minute or None if cannot parse
    """
    if not minute_str:
        return None
    
    # Remove whitespace
    minute_str = minute_str.strip()
    
    # Handle overtime notation: "45+2" -> 45, "90+4" -> 90
    if '+' in minute_str:
        minute_str = minute_str.split('+')[0]
    
    try:
        return int(minute_str)
    except ValueError:
        return None


def parse_bert_input(bert_input: str, half: int) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Parse BERT input into metadata and events.
    
    Format:
    - First half: "[season] HomeTeam AwayTeam [attendance] [Krank] event1 event2 ..."
    - Second half: "[season] HomeTeam AwayTeam [attendance] Stand 1:1 [Krank] event1 event2 ..."
    
    Events format: "minute: text"
    
    Args:
        bert_input: Full BERT input string
        half: 45 for first half, 90 for second half
        
    Returns:
        Tuple of (metadata_str, events_list) where events_list is [(minute, text), ...]
    """
    if not bert_input:
        return "", []
    
    # Pattern to match events: "minute: text"
    # Events start with a number (minute) followed by colon and space
    # The text continues until the next "minute: " pattern or end of string
    # Use a more robust pattern that handles cases where text might contain numbers
    event_pattern = r'(\d+(?:\+\d+)?):\s+([^0-9]+?)(?=\s+\d+(?:\+\d+)?:\s+|$)'
    
    # Find all event positions
    events = []
    last_end = 0
    metadata_start = 0
    
    # Find where metadata ends (first event starts)
    first_event_match = re.search(r'\d+(?:\+\d+)?:\s+', bert_input)
    if first_event_match:
        metadata_end = first_event_match.start()
        metadata = bert_input[:metadata_end].strip()
    else:
        # No events found, entire input is metadata
        return bert_input.strip(), []
    
    # Extract all events
    for match in re.finditer(event_pattern, bert_input):
        minute_str = match.group(1)
        event_text = match.group(2).strip()
        minute_int = parse_minute_to_int(minute_str)
        if minute_int is not None and event_text:
            events.append((minute_int, event_text))
    
    return metadata, events


def count_tokens(text: str) -> int:
    """
    Count tokens in text including special tokens (CLS + content + SEP).
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Token count including special tokens
    """
    if not text:
        return 0
    
    if TOKENIZER_AVAILABLE:
        try:
            tokens = TOKENIZER.encode(text, add_special_tokens=True, truncation=False, max_length=10000)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Using word count approximation.")
            return len(text.split()) * 1.4  # Approximate multiplier
    else:
        # Fallback to word count approximation
        return int(len(text.split()) * 1.4)


def truncate_bert_input(bert_input: str, half: int, max_tokens: int = 512) -> Tuple[str, int, int]:
    """
    Truncate BERT input by removing events (earliest to latest) until token count <= max_tokens.
    
    Only removes events, preserves all metadata.
    
    Args:
        bert_input: Full BERT input string
        half: 45 for first half, 90 for second half
        max_tokens: Maximum token count (default 512)
        
    Returns:
        Tuple of (truncated_input, original_token_count, new_token_count)
    """
    if not bert_input:
        return bert_input, 0, 0
    
    # Count original tokens
    original_tokens = count_tokens(bert_input)
    
    if original_tokens <= max_tokens:
        return bert_input, original_tokens, original_tokens
    
    # Parse into metadata and events
    metadata, events = parse_bert_input(bert_input, half)
    
    if not events:
        # No events to remove, return as is
        return bert_input, original_tokens, original_tokens
    
    # Sort events by minute (earliest first)
    events_sorted = sorted(events, key=lambda x: x[0])
    
    # Remove events one by one until token count <= max_tokens
    remaining_events = events_sorted.copy()
    current_tokens = original_tokens
    
    removed_count = 0
    while current_tokens > max_tokens and remaining_events:
        # Remove earliest event
        removed_event = remaining_events.pop(0)
        removed_count += 1
        
        # Reconstruct input by building from metadata + remaining events
        # This is more reliable than trying to remove from original string
        event_strings = []
        for event_minute, event_text in remaining_events:
            event_strings.append(f"{event_minute}: {event_text}")
        
        if event_strings:
            current_input = f"{metadata} {' '.join(event_strings)}"
        else:
            current_input = metadata
        
        # Clean up extra whitespace
        current_input = re.sub(r'\s+', ' ', current_input).strip()
        
        # Recalculate token count
        current_tokens = count_tokens(current_input)
    
    if removed_count > 0:
        logger.info(
            f"Removed {removed_count} events (earliest to latest). "
            f"Tokens: {original_tokens} -> {current_tokens}"
        )
    
    return current_input, original_tokens, current_tokens


def process_match_file(match_file: Path, dry_run: bool = False) -> Dict[str, any]:
    """
    Process a single match file, truncating BERT inputs if they exceed 512 tokens.
    
    Args:
        match_file: Path to processed match JSON file
        dry_run: If True, don't save changes, just report what would be changed
        
    Returns:
        Dictionary with processing results
    """
    match_id = match_file.stem.replace('match_', '')
    
    try:
        with open(match_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokens_45 = data.get('bert_input_45_tokens', 0)
        tokens_90 = data.get('bert_input_90_tokens', 0)
        
        changes_made = []
        
        # Process first half if over 512 tokens
        if tokens_45 > 512:
            bert_input_45 = data.get('bert_input_45', '')
            if bert_input_45:
                truncated_45, orig_45, new_45 = truncate_bert_input(bert_input_45, half=45, max_tokens=512)
                
                if new_45 < orig_45:
                    changes_made.append({
                        'half': 45,
                        'original_tokens': orig_45,
                        'new_tokens': new_45,
                        'truncated': True
                    })
                    data['bert_input_45'] = truncated_45
                    data['bert_input_45_tokens'] = new_45
                else:
                    changes_made.append({
                        'half': 45,
                        'original_tokens': orig_45,
                        'new_tokens': new_45,
                        'truncated': False,
                        'note': 'Could not reduce below 512 tokens'
                    })
        
        # Process second half if over 512 tokens
        if tokens_90 > 512:
            bert_input_90 = data.get('bert_input_90', '')
            if bert_input_90:
                truncated_90, orig_90, new_90 = truncate_bert_input(bert_input_90, half=90, max_tokens=512)
                
                if new_90 < orig_90:
                    changes_made.append({
                        'half': 90,
                        'original_tokens': orig_90,
                        'new_tokens': new_90,
                        'truncated': True
                    })
                    data['bert_input_90'] = truncated_90
                    data['bert_input_90_tokens'] = new_90
                else:
                    changes_made.append({
                        'half': 90,
                        'original_tokens': orig_90,
                        'new_tokens': new_90,
                        'truncated': False,
                        'note': 'Could not reduce below 512 tokens'
                    })
        
        # Save changes if not dry run
        if changes_made and not dry_run:
            with open(match_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Updated {match_id}: {changes_made}")
        
        return {
            'match_id': match_id,
            'season': data.get('season', 'Unknown'),
            'changes_made': changes_made,
            'saved': not dry_run and len(changes_made) > 0
        }
    
    except Exception as e:
        logger.error(f"Error processing {match_file.name}: {e}")
        return {
            'match_id': match_id,
            'error': str(e)
        }


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Truncate BERT inputs over 512 tokens')
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Path to processed data directory (default: data/processed)'
    )
    parser.add_argument(
        '--matches-file',
        type=str,
        help='JSON file with matches to process (from find_over_512_tokens.py). If not provided, scans all matches.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode: show what would be changed without saving'
    )
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    
    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return
    
    # Get list of matches to process
    matches_to_process = []
    
    if args.matches_file:
        # Load matches from JSON file
        matches_file = Path(args.matches_file)
        if not matches_file.exists():
            logger.error(f"Matches file not found: {matches_file}")
            return
        
        with open(matches_file, 'r', encoding='utf-8') as f:
            matches_data = json.load(f)
        
        all_matches = matches_data.get('all_matches', [])
        
        for match_info in all_matches:
            season = match_info['season']
            match_id = match_info['match_id']
            file_path = match_info.get('file_path', f"season_{season}/match_{match_id}.json")
            match_file = processed_dir / file_path
            
            if match_file.exists():
                matches_to_process.append(match_file)
            else:
                logger.warning(f"Match file not found: {match_file}")
    else:
        # Scan all matches for ones over 512 tokens
        logger.info("Scanning all matches for ones over 512 tokens...")
        season_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir() and d.name.startswith('season_')])
        
        for season_dir in season_dirs:
            match_files = list(season_dir.glob('match_*.json'))
            
            for match_file in match_files:
                try:
                    with open(match_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    tokens_45 = data.get('bert_input_45_tokens', 0)
                    tokens_90 = data.get('bert_input_90_tokens', 0)
                    
                    if tokens_45 > 512 or tokens_90 > 512:
                        matches_to_process.append(match_file)
                except Exception as e:
                    logger.warning(f"Error checking {match_file.name}: {e}")
    
    if not matches_to_process:
        logger.info("No matches found that exceed 512 tokens.")
        return
    
    logger.info(f"Found {len(matches_to_process)} matches to process")
    if args.dry_run:
        logger.info("DRY RUN MODE: No files will be modified")
    
    # Process each match
    results = []
    for match_file in matches_to_process:
        result = process_match_file(match_file, dry_run=args.dry_run)
        results.append(result)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    
    successful = sum(1 for r in results if r.get('saved', False) or (args.dry_run and r.get('changes_made')))
    failed = sum(1 for r in results if 'error' in r)
    
    logger.info(f"Total matches processed: {len(results)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed: {failed}")
    
    if args.dry_run:
        logger.info("\nDRY RUN: No files were modified. Run without --dry-run to apply changes.")


if __name__ == '__main__':
    main()

