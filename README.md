# Sentiment Analysis Pipeline

A comprehensive sentiment analysis pipeline that implements data scraping from multiple sources, preprocessing, model training with three different algorithms, and detailed evaluation metrics.

## Overview

This project provides a complete end-to-end sentiment analysis workflow that:
- Scrapes data from three sources: Google Play Store reviews, Twitter tweets, and e-commerce product comments
- Performs comprehensive preprocessing and cleaning
- Trains three different models: Logistic Regression with TF-IDF, LSTM with Word2Vec, and CNN with Bag of Words
- Evaluates models with detailed metrics and visualizations
- Performs inference on unseen data
- Provides comparative analysis and recommendations

**Performance Target**: Achieve >85% accuracy across all models, with at least one model exceeding 92%

## Features

### 1. Data Scraping
- **Playstore Reviews**: Using `google-play-scraper` to extract app reviews with ratings
- **Twitter Tweets**: Using `tweepy` API to collect tweets with sentiment-specific hashtags
- **E-commerce Comments**: Using `beautifulsoup4` to scrape product reviews from e-commerce websites

### 2. Preprocessing and Cleaning
- Deduplication and null value removal
- Special character and numeric value removal
- URL, mention, and hashtag cleaning
- Stopword removal
- Tokenization
- Stemming and lemmatization
- Sentiment labeling (negative, neutral, positive)

### 3. Model Training
Three different algorithms trained on each dataset:

#### Logistic Regression with TF-IDF
- Fast and interpretable baseline model
- TF-IDF feature extraction with unigrams and bigrams
- Balanced class weights

#### LSTM with Word2Vec Embeddings
- Sequential pattern recognition
- Custom Word2Vec embeddings trained on the data
- Bidirectional LSTM layers with dropout

#### CNN with Bag of Words
- Efficient feature extraction
- Conv1D layers with global max pooling
- Fast inference for large-scale deployment

### 4. Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **Training History**: Accuracy and loss curves for deep learning models

### 5. Inference
- Testing on unseen data
- Real-world applicability demonstration
- Tabular results with predictions and expected sentiments

### 6. Comparison and Recommendations
- Dataset characteristics comparison
- Model performance analysis
- Recommendations for optimal configuration
- Guidance for achieving >92% accuracy

## Installation

### Requirements
- Python 3.8 or higher
- Jupyter Notebook

### Setup

1. Clone this repository:
```bash
git clone https://github.com/AlfanArzaqi/SentimenAnalisis.git
cd SentimenAnalisis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For Twitter scraping, set up API credentials:
```bash
export TWITTER_API_KEY="your_api_key"
export TWITTER_API_SECRET="your_api_secret"
export TWITTER_ACCESS_TOKEN="your_access_token"
export TWITTER_ACCESS_SECRET="your_access_secret"
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook sentiment_analysis_pipeline.ipynb
```

2. Run all cells sequentially or execute specific sections:
   - **Section 1**: Data Scraping
   - **Section 2**: Preprocessing and Cleaning
   - **Section 3**: Model Training
   - **Section 4**: Evaluation and Visualization
   - **Section 5**: Inference
   - **Section 6**: Comparison and Recommendations

3. All generated files will be saved in the `data/` directory:
   - Raw datasets (CSV files)
   - Cleaned datasets (CSV files)
   - Model results and metrics
   - Visualizations (PNG images)
   - Inference results

## Project Structure

```
SentimenAnalisis/
├── sentiment_analysis_pipeline.ipynb  # Main Jupyter notebook
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── data/                             # Generated data and results (created on first run)
    ├── playstore_reviews.csv         # Raw Playstore data
    ├── twitter_tweets.csv            # Raw Twitter data
    ├── ecommerce_comments.csv        # Raw E-commerce data
    ├── playstore_cleaned.csv         # Preprocessed Playstore data
    ├── twitter_cleaned.csv           # Preprocessed Twitter data
    ├── ecommerce_cleaned.csv         # Preprocessed E-commerce data
    ├── model_results.csv             # Performance metrics
    ├── confusion_matrix_*.png        # Confusion matrices
    ├── training_history_*.png        # Training progression plots
    ├── metrics_comparison.png        # Comparative analysis
    ├── inference_results.csv         # Predictions on unseen data
    └── dataset_comparison.csv        # Dataset characteristics
```

## Model Performance

The notebook trains 9 models total (3 algorithms × 3 datasets) and provides comprehensive performance comparison. Results are automatically saved and visualized.

### Evaluation Splits
- Primary split: 80/20 (train/test)
- Alternative: 70/30 (for comparison)
- Stratified sampling ensures balanced class distribution

### Metrics Tracked
- Accuracy, Precision, Recall, F1-Score
- Per-class performance
- Training/validation curves
- Confusion matrices

## Customization

### Using Different Data Sources
- Modify the scraping functions in Section 1
- Adjust sample counts and parameters
- Add new data sources following the same pattern

### Adjusting Preprocessing
- Configure stopword removal in Section 2.2
- Toggle stemming vs. lemmatization
- Customize text cleaning patterns

### Hyperparameter Tuning
- Modify model architectures in Section 3
- Adjust learning rates, batch sizes, epochs
- Change feature extraction parameters (TF-IDF, embedding dimensions)

## Notes

- Sample data is used when scraping APIs are unavailable
- The notebook includes detailed comments for each step
- Training deep learning models may take time depending on hardware
- GPU acceleration is recommended for faster training (automatically used if available)

## Requirements

Key dependencies:
- `google-play-scraper`: Playstore review scraping
- `tweepy`: Twitter API access
- `beautifulsoup4`: Web scraping
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning models and metrics
- `tensorflow`, `keras`: Deep learning models
- `nltk`, `gensim`: NLP preprocessing and embeddings
- `matplotlib`, `seaborn`: Visualization

See `requirements.txt` for complete list with versions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Google Play Scraper for app review collection
- Twitter API for tweet data
- scikit-learn and TensorFlow communities
- NLTK and Gensim for NLP tools

## Contact

For questions or issues, please open an issue on GitHub.