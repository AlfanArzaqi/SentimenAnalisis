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

**Performance Target**: Achieve 92%+ accuracy with advanced ensemble methods and optimized model architectures

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
Four different approaches trained on the Play Store dataset:

#### Logistic Regression with Enhanced TF-IDF
- Fast and interpretable baseline model
- TF-IDF feature extraction with unigrams, bigrams, and trigrams
- ElasticNet regularization (L1 + L2)
- Balanced class weights for handling imbalance
- 10,000 feature vocabulary

#### Advanced BiLSTM with Attention Mechanism
- Bidirectional LSTM layers for capturing context from both directions
- Custom attention mechanism to focus on important words
- Word2Vec embeddings trained on the data
- Batch normalization and L2 regularization
- Dropout for preventing overfitting

#### Multi-Filter CNN
- Parallel convolutional layers with multiple filter sizes (2, 3, 4, 5)
- Efficient feature extraction from n-grams
- Global max pooling for dimension reduction
- Fast inference for large-scale deployment

#### Weighted Ensemble
- Combines predictions from all three models
- Weights based on validation accuracy
- Target: 92-95% accuracy through ensemble voting

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
   - **Section 2.1-2.4**: Basic Preprocessing and Cleaning
   - **Section 2.5**: Data Augmentation for Class Balancing (NEW)
   - **Section 2.6**: Feature Engineering (NEW)
   - **Section 3.1**: Data Preparation with Class Weights
   - **Section 3.2**: Improved Logistic Regression Training
   - **Section 3.3**: Advanced BiLSTM with Attention Training
   - **Section 3.4**: Multi-Filter CNN Training
   - **Section 3.5**: Weighted Ensemble (NEW)
   - **Section 4**: Comprehensive Evaluation and Visualization
   - **Section 5**: Inference on New Data
   - **Section 6**: Performance Analysis and Recommendations

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

### Recent Improvements (v2.0)

This version includes **12 major improvements** to achieve 92%+ accuracy:

#### Data Quality Enhancement
1. **Advanced Sentiment Labeling** - Context-aware with strong keyword detection and negation handling
2. **Enhanced Text Cleaning** - Indonesian slang normalization (60+ mappings) and repeated character reduction
3. **Data Augmentation** - Synonym replacement using nlpaug for class balancing
4. **Feature Engineering** - 9 sentiment-relevant features (text length, punctuation, word counts, etc.)

#### Model Architecture
5. **Improved Logistic Regression** - ElasticNet regularization, trigrams, 10K vocabulary
6. **BiLSTM with Attention** - Custom attention mechanism for focusing on important words
7. **Multi-Filter CNN** - Parallel convolutions with filters [2, 3, 4, 5]
8. **Weighted Ensemble** - Combines all models with validation-accuracy-based weights

#### Training Optimization
9. **Class Weight Balancing** - Handles severe class imbalance (78% positive, 19% negative, 3% neutral)
10. **Advanced Callbacks** - EarlyStopping (patience=15), ReduceLROnPlateau, ModelCheckpoint
11. **Extended Training** - 100 epochs with early stopping and learning rate scheduling
12. **Comprehensive Metrics** - Precision, Recall, F1-score per class + ROC curves

### Expected Performance

The notebook trains 4 models on Play Store review data and provides comprehensive performance comparison.

| Model | Expected Accuracy | Key Features |
|-------|-------------------|--------------|
| Improved Logistic Regression | 84-87% | Fast, interpretable, ElasticNet |
| Advanced BiLSTM + Attention | 88-91% | Context-aware, attention mechanism |
| Multi-Filter CNN | 87-90% | Parallel filters, efficient |
| **Weighted Ensemble** | **92-95%** ⭐ | **Best performance, robust** |

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
- `nlpaug`: Data augmentation for text (NEW)
- `matplotlib`, `seaborn`: Visualization
- `wordcloud`: WordCloud generation for sentiment analysis

See `requirements.txt` for complete list with versions.

## Documentation

- **IMPROVEMENTS_SUMMARY.md** - Detailed explanation of all 12 improvements
- **VERIFICATION_CHECKLIST.md** - Pre/post-training verification checklist
- **README.md** - This file with usage instructions

## Performance Tips

### For Best Results
1. **Use GPU**: Enable GPU runtime in Google Colab for 3-4x faster training
2. **Run Data Augmentation**: Essential for achieving 92%+ accuracy
3. **Enable All Callbacks**: Prevents overfitting and saves best models
4. **Use Ensemble**: Weighted ensemble typically adds 2-4% accuracy over best individual model

### If Training Takes Too Long
- Reduce `max_words` from 15000 to 5000
- Reduce `max_len` from 200 to 150  
- Reduce `batch_size` from 32 to 16
- Skip data augmentation (expect lower accuracy)

### If Out of Memory
- Use smaller batch size (16 or 8)
- Reduce embedding dimensions to 64-128
- Reduce LSTM/CNN units by half
- Skip data augmentation

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