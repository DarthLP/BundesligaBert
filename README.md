# BundesligaBERT: Predicting Stoppage Time

**Repository:** `bundesliga-bert-time`

## Overview
A BERT-based NLP model predicting Bundesliga stoppage time from live ticker text. Investigates referee behavior and home bias by comparing model-predicted norms against actual added time.

## Project Goals
1. **Data Engineering:** Scrape Kicker.de (handling strict anti-bot measures) and parse German ticker text.
2. **NLP Modeling:** Fine-tune BERT for regression (predicting minutes).
3. **Economic Analysis:** Calculate "Residuals" (Actual - Predicted) to measure referee bias and perform a Placebo DID analysis on the 2024/25 season.

## Critical Constraints
1. **NO LEAKAGE:** Aggressively remove explicit time announcements (e.g., "Nachspielzeit", "+3") from input text.
2. **Fine-Tuning:** Do NOT train from scratch. Use `BertForSequenceClassification.from_pretrained()`.
3. **Tech Stack:** PyTorch, Hugging Face `transformers`, `pandas`.

## Setup

### Environment
1. Create conda environment: `conda env create -f environment.yml`
2. Activate: `conda activate bundesliga-bert`

### Dependencies
- Python 3.10
- PyTorch, transformers, tokenizers, datasets, accelerate
- pandas, numpy, scikit-learn
- selenium, undetected-chromedriver (for scraping)
- matplotlib, seaborn (for visualization)
- jupyterlab (for notebooks)

## Project Structure

```
BundesligaBert/
├── data/
│   ├── raw/                    # Scraped match JSON files
│   │   └── season_YYYY-YY/     # Per-season directories
│   ├── processed/              # Processed datasets for training
│   └── results/                # Model outputs and predictions
├── src/
│   ├── data/
│   │   └── kicker_scraper.py  # Web scraper for Kicker.de
│   ├── features/
│   │   └── preprocessing.py   # Text preprocessing and leakage removal (TODO)
│   ├── models/
│   │   └── bert_module.py      # BERT fine-tuning module (TODO)
│   └── visualization/
│       └── plot_results.py     # Visualization utilities (TODO)
├── tests/
│   └── test_scraper_live.py   # Integration tests for scraper
├── notebooks/                  # Jupyter notebooks for analysis
├── reports/
│   ├── figures/                # Generated plots
│   └── tables/                 # Generated tables
├── environment.yml             # Conda environment specification
└── README.md                  # This file
```

## Data Collection: Simultaneous Season Download

The scraper can download all Bundesliga match data from 2017-18 to 2024-25 using multiprocessing for parallel execution.

### Quick Start: Download All Seasons

```bash
# From project root directory
python src/data/kicker_scraper.py
```

This will:
- **Run 4 processes simultaneously** (2 seasons per process)
- **Scrape all matchdays** (1-34) for each season
- **Skip already downloaded matches** automatically
- **Save data** to `data/raw/season_{season}/match_{match_id}.json`
- **Show progress** with per-season logging

### How It Works

1. **Multiprocessing**: Each process handles 2 seasons with its own Chrome instance
2. **Anti-ban measures**:
   - Chrome runs in incognito mode and background (headless)
   - Cookies are cleared between requests
   - Random user agents are used
   - Delays of 4-8 seconds between matches
3. **Resume capability**: Already downloaded matches are automatically skipped
4. **Progress tracking**: Each process logs independently with prefixes like `[PROCESS 1: 2017-18, 2018-19]`

### Data Structure

Each match is saved as `data/raw/season_{season}/match_{match_id}.json`:

```json
{
  "match_id": "bayern-gegen-hoffenheim-2024-bundesliga-4862110",
  "metadata": {
    "home_team": "Bayern München",
    "away_team": "TSG Hoffenheim",
    "final_score": "3:0",
    "attendance": 75000,
    "stadium": "Allianz Arena",
    "referee": "Felix Zwayer",
    "kickoff_date": "2024-01-12",
    "kickoff_time": "20:30",
    "is_ghost_game": false
  },
  "targets": {
    "announced_time_45": 2,
    "actual_played_45": 3,
    "announced_time_90": 4,
    "actual_played_90": 6
  },
  "score_timeline": {
    "23": [1, 0],
    "45": [1, 0],
    "67": [2, 0],
    "90": [3, 0]
  },
  "ticker_data": [
    {
      "minute": "23",
      "text": "Tor für Bayern München durch Robert Lewandowski",
      "extracted_score": [1, 0],
      "score_at_event": [1, 0]
    },
    ...
  ]
}
```

### Testing Individual Matches

For testing or debugging, use the test script:

```bash
# Test with dynamic match discovery
python -m tests.test_scraper_live

# Test specific known matches
python -m tests.test_scraper_live --test-known
python -m tests.test_scraper_live --test-bayern
python -m tests.test_scraper_live --test-bremen
```

### Performance Notes

- **Total matches**: ~2,448 matches (8 seasons × ~306 matches/season)
- **Estimated time**: Several hours (depends on network speed and delays)
- **Memory usage**: Each process uses ~200-500MB RAM
- **Network**: Requires stable internet connection

### Troubleshooting

- **If a process fails**: The other processes continue running independently
- **To resume**: Simply run the script again - it will skip already downloaded matches
- **To force re-scrape**: Delete the specific match file or season directory
- **Check logs**: Each process logs independently with clear prefixes

## Scripts

### `src/data/kicker_scraper.py`

**Purpose:** Web scraper for Bundesliga match data from Kicker.de

**Usage:**
```bash
# Download all seasons (multiprocessing)
python src/data/kicker_scraper.py

# Use as a module
from src.data.kicker_scraper import KickerScraper
from pathlib import Path

scraper = KickerScraper(save_dir=Path("data/raw"))
result = scraper.scrape_full_match(
    match_url="https://www.kicker.de/...",
    season="2023-24",
    matchday=1
)
```

**Key Features:**
- Multi-tab scraping (Spielinfo + Ticker)
- Automatic leakage removal (removes "Nachspielzeit" announcements from text)
- Extracts targets (announced_time, actual_played_time) before cleaning text
- Handles JavaScript-rendered content with Selenium
- Anti-ban measures (incognito mode, random delays, user agents)

**Output:** JSON files in `data/raw/season_{season}/match_{match_id}.json`

### `tests/test_scraper_live.py`

**Purpose:** Integration tests for the KickerScraper

**Usage:**
```bash
# Dynamic match discovery (default)
python -m tests.test_scraper_live

# Test known matches
python -m tests.test_scraper_live --test-known
python -m tests.test_scraper_live --test-bayern
python -m tests.test_scraper_live --test-bremen
```

**Features:**
- Dynamically discovers match URLs
- Tests structure, targets, leakage detection, and data quality
- Saves results to `tests/artifacts/smoke_test_result.json`

### `src/features/preprocessing.py` (TODO)

**Purpose:** Text preprocessing and leakage removal utilities

**Status:** Not yet implemented

### `src/models/bert_module.py` (TODO)

**Purpose:** BERT fine-tuning module for regression

**Status:** Not yet implemented

**Planned Features:**
- Fine-tune `deepset/gbert-base` for regression
- Use `BertForSequenceClassification.from_pretrained()`
- Predict stoppage time from ticker text

### `src/visualization/plot_results.py` (TODO)

**Purpose:** Visualization utilities for results and analysis

**Status:** Not yet implemented

## Data Leakage Prevention

The scraper implements strict leakage prevention:

1. **Extract targets first**: `announced_time_90` and `actual_played_90` are extracted before text cleaning
2. **Remove time announcements**: All sentences containing "Nachspielzeit", "+X", or time patterns are removed
3. **Skip announcement events**: Events that are purely time announcements are excluded from ticker_data
4. **Pattern matching**: Multiple regex patterns catch variations like:
   - "4 Minuten Nachspielzeit"
   - "+4"
   - "Nachspielzeit: 4"
   - "Eine Minute obendrauf"

## Seasons Covered

- **2017-18, 2018-19**: Pre-Corona, Clean
- **2019-20, 2020-21, 2021-22**: Corona/Ghost Games (flagged but kept)
- **2022-23, 2023-24**: Post-Corona, Clean
- **2024-25**: Placebo Test Season

## Next Steps

1. **Preprocessing Module**: Implement text cleaning and feature engineering
2. **BERT Fine-Tuning**: Implement regression model using `deepset/gbert-base`
3. **Analysis**: Calculate residuals and perform Placebo DID analysis
4. **Visualization**: Create plots and tables for results

## License

[Add license information]

## Author

BundesligaBERT Project
