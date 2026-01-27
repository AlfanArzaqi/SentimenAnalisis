# Sentiment Analysis Pipeline

A comprehensive sentiment analysis pipeline for **Indonesian language** that implements data scraping from Google Play Store, context-based sentiment labeling, preprocessing, model training with three different algorithms, and detailed evaluation metrics with wordcloud visualizations.

## Overview

This project provides a complete end-to-end sentiment analysis workflow that:
- Scrapes **Indonesian language** data from Google Play Store reviews
- Uses **context-based sentiment labeling** algorithm (analyzes sentence structure, keywords, and context instead of ratings)
- Performs comprehensive preprocessing and cleaning with Indonesian stopwords
- Trains three different models: Logistic Regression with TF-IDF, LSTM with Word2Vec, and CNN with Bag of Words
- Evaluates models with detailed metrics and visualizations
- Generates **WordCloud visualizations** for each sentiment category
- Performs inference on unseen data
- Provides model performance analysis and recommendations

**Performance Target**: Achieve >85% accuracy across all models, with at least one model exceeding 92%

## Features

### 1. Data Scraping
- **Indonesian Play Store Reviews**: Using `google-play-scraper` to extract Indonesian app reviews with ratings
- Language: Indonesian (`id`)
- Country: Indonesia (`id`)

### 2. Preprocessing and Cleaning
- Deduplication and null value removal
- Special character and numeric value removal
- URL, mention, and hashtag cleaning
- **Indonesian stopword removal** (with English support for mixed language)
- Tokenization
- Stemming and lemmatization
- **Context-based sentiment labeling** (negative, neutral, positive)
  - Uses Indonesian sentiment lexicon (positive/negative keywords)
  - Analyzes sentence structure and context
  - Handles negation words and intensifiers
  - **Does NOT rely on rating scores** for labeling

### 3. Model Training
Three different algorithms trained on the Play Store dataset:

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
- **WordCloud**: Visual representation of word distribution for each sentiment category

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

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook sentiment_analysis_pipeline.ipynb
```

2. Run all cells sequentially or execute specific sections:
   - **Section 1**: Data Scraping from Play Store
   - **Section 2**: Preprocessing and Cleaning
   - **Section 3**: Model Training
   - **Section 4**: Evaluation and Visualization
   - **Section 5**: Inference
   - **Section 6**: Model Performance Analysis and Recommendations

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
    ├── playstore_reviews.csv         # Raw Play Store data
    ├── playstore_cleaned.csv         # Preprocessed Play Store data
    ├── model_results.csv             # Performance metrics
    ├── confusion_matrix_*.png        # Confusion matrices
    ├── training_history_*.png        # Training progression plots
    ├── metrics_comparison.png        # Comparative analysis
    ├── inference_results.csv         # Predictions on unseen data
    └── dataset_comparison.csv        # Dataset characteristics
```

## Model Performance

The notebook trains 3 models on Play Store review data and provides comprehensive performance comparison. Results are automatically saved and visualized.

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

### Context-Based Sentiment Labeling

This project uses a **context-based sentiment labeling algorithm** instead of relying on rating scores. The algorithm:

1. **Lexicon-Based Analysis**: Uses Indonesian sentiment lexicons containing positive and negative keywords
2. **Negation Handling**: Detects negation words (e.g., "tidak", "bukan") that flip sentiment
3. **Intensifier Detection**: Identifies intensifying words (e.g., "sangat", "banget") that strengthen sentiment
4. **Context Awareness**: Analyzes word sequences and their relationships
5. **Ratio-Based Classification**: Calculates positive/negative word ratios for final classification

**Why Context-Based?**
- Rating scores may not accurately reflect the actual sentiment expressed in text
- Users might give high ratings but write negative comments (or vice versa)
- Text analysis provides more accurate sentiment based on actual content
- Better captures nuanced opinions and mixed sentiments

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
- `google-play-scraper`: Play Store review scraping
- `beautifulsoup4`: Web scraping utilities
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning models and metrics
- `tensorflow`, `keras`: Deep learning models
- `nltk`, `gensim`: NLP preprocessing and embeddings
- `matplotlib`, `seaborn`: Visualization
- `wordcloud`: WordCloud generation for sentiment analysis

See `requirements.txt` for complete list with versions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Google Play Scraper for app review collection
- scikit-learn and TensorFlow communities
- NLTK and Gensim for NLP tools

## Contact

For questions or issues, please open an issue on GitHub.