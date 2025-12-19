# ML-Based Adaptive Hyperparameter Selection - Training Pipeline

This directory contains the complete pipeline for training and using ML models for adaptive hyperparameter selection in MinHash LSH web crawling.

## Overview

The system uses **Bayesian Optimization** (Gaussian Process Regression) to predict optimal hyperparameters (`k`, `ell`, `tau`) based on HTML/DOM complexity features.

## Pipeline Steps

### 1. Scrape Websites (`scrape_websites.py`)

Collects HTML from 14 diverse websites across 7 categories:
- E-commerce (Amazon, Etsy)
- News (BBC, CNN)
- Blogs (Medium, WordPress)
- Forums (Reddit, StackOverflow)
- SPAs (quotes.toscrape.com, Airbnb)
- Documentation (Python docs, MDN)
- Web Apps (GitHub, YouTube)

**Usage:**
```bash
python scrape_websites.py
```

**Output:**
- `training_data/raw_html/` - Raw HTML files organized by type
- `training_data/features.json` - Extracted complexity features

**Time:** ~5-10 minutes (depending on network)

---

### 2. Collect Training Data (`collect_training_data.py`)

Runs grid search (125 combinations) on each website to find optimal hyperparameters based on coverage/redundancy trade-off.

**Grid Search Parameters:**
- k: [8, 10, 12, 15, 18]
- ell: [100, 150, 200, 250, 300]
- tau: [0.70, 0.75, 0.80, 0.85, 0.90]

**Usage:**
```bash
python collect_training_data.py
```

**Output:**
- `training_data/training_dataset.json` - Training samples with optimal params

**Time:** ~10-30 minutes (depends on number of pages)

---

### 3. Train Bayesian Models (`train_bayesian_model.py`)

Trains 3 Gaussian Process Regression models (one for each hyperparameter) using the collected training data.

**Usage:**
```bash
python train_bayesian_model.py
```

**Output:**
- `models/bayesian_optimizer.pkl` - Trained models
- Evaluation metrics (RMSE, R² score)

**Time:** ~1-2 minutes

---

### 4. Use Adaptive Mode

Once trained, use adaptive hyperparameter selection:

```python
from core import MinHashLSH

# Automatically selects hyperparameters using ML model
detector = MinHashLSH(adaptive=True, html_sample=first_page_html)

for html in html_pages:
    detector.add_state(html)

unique_states = detector.get_unique_states()
```

---

## Quick Start

Run all steps in sequence:

```bash
# Step 1: Scrape websites
python scrape_websites.py

# Step 2: Find optimal hyperparameters via grid search
python collect_training_data.py

# Step 3: Train ML models
python train_bayesian_model.py

# Step 4: Test the models
python hyperparameter_optimizer.py
```

---

## Requirements

```bash
pip install requests beautifulsoup4 selenium scikit-learn numpy
```

**Chrome Driver:** Required for Selenium (automatic download in newer versions)

---

## File Structure

```
project_code/
├── scrape_websites.py          # Website scraper
├── collect_training_data.py    # Grid search & data collection
├── train_bayesian_model.py     # Model training
├── hyperparameter_optimizer.py # ML model wrapper
├── feature_extractor.py        # Feature extraction
├── core.py                     # MinHashLSH with adaptive mode
├── fetcher.py                  # Web crawler
│
├── training_data/
│   ├── raw_html/              # Scraped HTML files
│   ├── features.json          # Extracted features
│   └── training_dataset.json  # Training data with optimal params
│
└── models/
    └── bayesian_optimizer.pkl # Trained GP models
```

---

## Important Notes

1. **No Heuristic Fallback**: The system now requires trained models. If models are missing, it will fail with clear instructions.

2. **Model Retraining**: To retrain with new data, simply re-run steps 1-3. The old model will be overwritten.

3. **Custom Websites**: Edit `WEBSITES` list in `scrape_websites.py` to add your own target websites.

4. **Grid Search Optimization**: Adjust `PARAM_GRID` in `collect_training_data.py` to explore different parameter ranges.

---

## Troubleshooting

**Error: "Trained model not found"**
- Run the training pipeline (steps 1-3)

**Error: "scikit-learn not installed"**
- Install: `pip install scikit-learn`

**Selenium errors:**
- Ensure Chrome/ChromeDriver is installed
- Try updating: `pip install --upgrade selenium`

**Poor prediction accuracy:**
- Collect more diverse websites
- Increase grid search resolution
- Check evaluation metrics in training output
