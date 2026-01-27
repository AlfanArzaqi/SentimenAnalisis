# Sentiment Analysis Pipeline

A production-ready sentiment analysis pipeline for **Indonesian language** that implements multi-source data scraping, context-based sentiment labeling, advanced preprocessing, and training of **6 sophisticated models** designed to achieve **≥95% accuracy**.

## Overview

This project provides a complete end-to-end sentiment analysis workflow that:
- Scrapes **Indonesian language** data from **3 Google Play Store apps** (Instagram, TikTok, WhatsApp)
- Collects **25,000+ balanced samples** (addressing severe class imbalance)
- Uses **context-based sentiment labeling** algorithm (analyzes sentence structure, keywords, and context)
- Performs comprehensive preprocessing with Indonesian stopwords and data augmentation
- Trains **6 advanced models**: Baseline LR, Optimized LR with GridSearchCV, Basic LSTM, BiLSTM with Attention, Multi-Filter CNN, and Ensemble
- Implements **hyperparameter tuning** and **5-fold cross-validation**
- Evaluates models with **7 types of visualizations** and comprehensive metrics
- Generates **WordCloud visualizations** for each sentiment category
- Provides **production-ready inference pipeline** with confidence scores
- Complete model performance analysis and recommendations

**Performance Target**: Achieve **≥95% accuracy** with advanced ensemble methods, balanced data, and optimized model architectures

## What's New in v3.0 (Upgraded Pipeline) 🚀

### Major Upgrades:
✅ **Multi-Source Data Collection**: From 1 app → 3 apps (Instagram, TikTok, WhatsApp)  
✅ **Balanced Dataset**: From 78%/19%/3% imbalance → 37%/37%/26% balanced (25,000+ samples)  
✅ **6 Advanced Models**: Expanded from 4 to 6 models with ensemble voting  
✅ **Hyperparameter Optimization**: GridSearchCV with 20+ combinations  
✅ **Enhanced Features**: 9 engineered features + advanced TF-IDF  
✅ **Comprehensive Evaluation**: 7 visualization types + detailed metrics  
✅ **Production Pipeline**: Complete inference system with confidence scores  
✅ **Cross-Validation**: 5-fold stratified CV for robustness  

See **UPGRADE_IMPLEMENTATION.md** for detailed changes.

## Features

### 1. Multi-Source Data Scraping
- **3 Indonesian Apps**: Instagram, TikTok, WhatsApp (~8,500 reviews each)
- **Total Samples**: 25,000+ balanced samples
- **Class-Balanced Collection**: Smart scraping to achieve 37%/37%/26% distribution
- **Language**: Indonesian (`id`)
- **Country**: Indonesia (`id`)
- **Duplicate Removal**: Across all sources

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

### 3. Advanced Model Training
**Six sophisticated models** trained on balanced multi-source dataset:

#### Model 1: Baseline Logistic Regression
- Fast baseline with TF-IDF vectorization
- class_weight='balanced' for imbalance handling
- 10,000 feature vocabulary

#### Model 2: Optimized Logistic Regression with GridSearchCV ⭐
- **Hyperparameter tuning**: 20 combinations, 3-fold CV
- **9 engineered features**: text_length, word_count, punctuation, sentiment ratios, etc.
- Enhanced TF-IDF with trigrams (15,000 features)
- Best parameters automatically selected

#### Model 3: Basic LSTM
- Single LSTM layer (128 units)
- Dropout regularization (0.3)
- Simple baseline for deep learning

#### Model 4: Bidirectional LSTM with Attention ⭐
- **Custom attention mechanism** for important word focus
- 2 BiLSTM layers capturing bidirectional context
- Batch normalization and L2 regularization
- Dropout for preventing overfitting

#### Model 5: Multi-Filter CNN ⭐
- **Parallel 1D convolutions** with multiple filter sizes [2, 3, 4, 5]
- 128 filters per size (512 total filters)
- Global max pooling for dimension reduction
- Efficient multi-scale n-gram feature extraction

#### Model 6: Ensemble Model (Soft Voting) 🎯
- **Weighted combination**: Optimized LR (30%), BiLSTM (35%), CNN (35%)
- Soft voting with probability averaging
- Leverages strengths of multiple models
- **Target: ≥95% accuracy**
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

### 4. Comprehensive Evaluation Metrics
- **Accuracy**: Overall and per-class correctness
- **Precision**: Positive predictive value (per class)
- **Recall**: True positive rate (per class)
- **F1-Score**: Harmonic mean of precision and recall (per class)
- **ROC-AUC**: One-vs-rest for 3 classes
- **Confusion Matrix**: Visual representation of predictions (all 6 models)
- **Training/Inference Time**: Performance benchmarking
- **Cross-Validation**: 5-fold stratified CV for robustness

### 5. Advanced Visualizations (7 Types)
1. **Confusion Matrices**: All 6 models in 2x3 grid
2. **ROC Curves**: One-vs-rest with AUC scores
3. **Model Comparison**: Bar charts (accuracy, F1-score, training time)
4. **Training History**: Accuracy and loss curves for deep learning models
5. **Word Clouds**: Color-coded by sentiment (positive, negative, neutral)
6. **Class Distribution**: Before/after augmentation comparison
7. **Precision-Recall Analysis**: Per-class performance

### 6. Production-Ready Inference Pipeline
- **Unified Interface**: `predict_sentiment(text, model_type='ensemble')`
- **Confidence Scores**: Probability distribution for all classes
- **Multi-Model Support**: Test with any of the 6 models
- **Interactive Mode**: Real-time predictions with visual feedback
- **Demo Samples**: Pre-loaded Indonesian test cases
### 7. Model Comparison Dashboard
- Comprehensive performance table across all 6 models
- Best model identification (≥95% accuracy target)
- Detailed analysis and recommendations
- Guidance for achieving optimal performance

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
   - **Section 1**: Multi-Source Data Scraping (Instagram, TikTok, WhatsApp)
   - **Section 2.1-2.4**: Preprocessing and Cleaning
   - **Section 2.5**: Data Augmentation for Class Balancing
   - **Section 2.6**: Feature Engineering (9 additional features)
   - **Section 3**: Model Definitions (6 models)
   - **Section 4**: Complete Model Training Pipeline
     - 4.1: Data preparation with 70/30 split
     - 4.2-4.8: Train all 6 models sequentially
   - **Section 5**: Comprehensive Evaluation and Visualization
     - 5.1: Model comparison summary
     - 5.2-5.7: All 7 visualization types
   - **Section 6**: Production Inference Pipeline
     - 6.1: Inference function
     - 6.2: Demo with test samples
     - 6.3: Interactive mode
     - 6.4: Cross-validation
   - **Section 7**: Final Summary & Conclusions

3. All generated files will be saved in the `data/` directory:
   - Raw datasets (CSV files)
   - Cleaned datasets (CSV files)
   - Model results and metrics
   - Visualizations (PNG images)
   - Inference results

## Project Structure

```
SentimenAnalisis/
├── sentiment_analysis_pipeline.ipynb  # Main upgraded notebook (97 cells, 6 models)
├── sentiment_analysis_pipeline_backup.ipynb  # Original backup
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── UPGRADE_IMPLEMENTATION.md          # Technical upgrade details (NEW)
├── IMPLEMENTATION_SUMMARY.md          # Complete feature overview (NEW)
├── SECURITY_SUMMARY.md                # Security validation (NEW)
├── EXECUTION_CHECKLIST.md             # Verification checklist (NEW)
├── models/                            # Saved models (created on execution)
│   ├── baseline_lr_model.pkl
│   ├── optimized_lr_model.pkl
│   ├── basic_lstm_model.h5
│   ├── bilstm_model.h5
│   ├── cnn_model.h5
│   ├── *_tokenizer.pkl
│   └── *.png (all visualizations)
└── data/                              # Generated data (created on execution)
    ├── playstore_reviews.csv          # Combined raw data (25K+ samples)
    └── (other processed datasets)
```

## Model Performance

### Expected Performance (v3.0 Upgrade)

The notebook trains **6 advanced models** on balanced multi-source data and provides comprehensive performance comparison.

| Model | Expected Accuracy | Key Features |
|-------|-------------------|--------------|
| Baseline Logistic Regression | 84-87% | Fast, interpretable, balanced weights |
| **Optimized LR + GridSearchCV** | **88-91%** | 9 features, hyperparameter tuning |
| Basic LSTM | 87-90% | Simple deep learning baseline |
| **BiLSTM + Attention** | **91-94%** | Attention mechanism, batch norm |
| Multi-Filter CNN | 89-92% | Parallel convolutions, efficient |
| **Ensemble (Soft Voting)** | **≥95%** 🎯 | **Best performance, production-ready** |

### Major Improvements (v2.0 → v3.0)

This version includes **comprehensive upgrades** to achieve ≥95% accuracy:

#### Data Quality Enhancement
1. **Multi-Source Collection** - 3 apps (Instagram, TikTok, WhatsApp) instead of 1
2. **Balanced Dataset** - 25,000+ samples with 37%/37%/26% distribution (vs 78%/19%/3%)
3. **Class-Balanced Scraping** - Smart rating-based filtering with text validation
4. **Data Augmentation** - nlpaug with 60% minority target ratio

#### Model Architecture Expansion
5. **6 Advanced Models** - Expanded from 4 to 6 models
6. **Optimized LR with GridSearchCV** - 20 hyperparameter combinations, 3-fold CV
7. **Enhanced Features** - 9 engineered features + advanced TF-IDF
8. **Ensemble Voting** - Weighted combination of top 3 models

#### Training Optimization
9. **Hyperparameter Tuning** - GridSearchCV for systematic optimization
10. **Cross-Validation** - 5-fold stratified CV for robustness
11. **70/30 Train-Test Split** - Optimized from 80/20
12. **Class Weighting** - Applied to all models for imbalance handling

#### Evaluation & Visualization
13. **7 Visualization Types** - Comprehensive visual analysis
14. **Complete Metrics** - Accuracy, precision, recall, F1, ROC-AUC, timing
15. **Model Comparison Dashboard** - Side-by-side performance analysis

### Evaluation Splits
- **Primary split**: 70/30 (train/test) - optimized for performance
- Stratified sampling ensures balanced class distribution
- 5-fold cross-validation for robustness validation

### Metrics Tracked
- Accuracy (overall + per-class)
- Precision, Recall, F1-Score (per-class)
- ROC-AUC (one-vs-rest)
- Training/validation curves
- Confusion matrices (all 6 models)
- Training and inference time

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
- Currently configured for Instagram, TikTok, WhatsApp
- Add new apps by specifying their Play Store IDs
- Adjust sample counts per app as needed

### Adjusting Class Balance
- Configure target distribution in scraping functions
- Default: 37% positive, 37% negative, 26% neutral
- Modify augmentation ratio for minority classes
- Use class weighting for training optimization

### Adjusting Preprocessing
- Configure stopword removal in Section 2.2
- Toggle stemming vs. lemmatization
- Customize text cleaning patterns

### Hyperparameter Tuning
- Modify GridSearchCV parameters in Section 3
- Current: 20 combinations for Logistic Regression
- Adjust learning rates, batch sizes, epochs for deep learning
- Use RandomSearchCV for broader parameter exploration
- Change feature extraction parameters (TF-IDF, embedding dimensions)

## Notes

- Multi-source scraping may take 10-15 minutes for all apps
- Sample data fallback available when scraping APIs are unavailable
- The notebook includes detailed comments and markdown explanations
- Training deep learning models may take time depending on hardware
- **GPU acceleration recommended** for faster training (automatically used if available)
- Estimated total execution time: **50-90 minutes** (with GPU)
- All 97 cells designed for sequential execution
- Models automatically saved to `models/` directory
- Visualizations automatically saved as PNG files

## Requirements

Key dependencies:
- `google-play-scraper`: Play Store review scraping
- `beautifulsoup4`: Web scraping utilities (if needed)
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Machine learning models, metrics, and hyperparameter tuning
- `tensorflow`, `keras`: Deep learning models (LSTM, BiLSTM, CNN)
- `nltk`, `gensim`: NLP preprocessing and embeddings
- `nlpaug`: Data augmentation for text (synonym replacement)
- `matplotlib`, `seaborn`: Visualization and plotting
- `wordcloud`: WordCloud generation for sentiment analysis

See `requirements.txt` for complete list with versions.

## Documentation

- **README.md** - This file with complete overview and usage instructions
- **UPGRADE_IMPLEMENTATION.md** - Detailed technical implementation of v3.0 upgrade
- **IMPLEMENTATION_SUMMARY.md** - Complete feature overview and execution guide
- **SECURITY_SUMMARY.md** - Security validation and best practices
- **EXECUTION_CHECKLIST.md** - Pre/post-execution verification checklist
- **VERIFICATION_CHECKLIST.md** - Original verification checklist

## Performance Tips

### For Best Results (≥95% Accuracy)
1. **Use GPU**: Enable GPU runtime in Google Colab for 3-5x faster training
2. **Execute Data Augmentation**: Essential for class balance and achieving high accuracy
3. **Enable All Features**: Use all 9 engineered features + TF-IDF
4. **Run GridSearchCV**: Let hyperparameter optimization find best parameters
5. **Use Ensemble Model**: Weighted voting typically adds 2-5% over best individual model
6. **Ensure Balanced Data**: Verify class distribution is close to 37%/37%/26%

### If Training Takes Too Long
- Enable GPU: Runtime → Change runtime type → GPU (reduces time by 3-5x)
- Reduce `max_words` from 15000 to 10000
- Reduce `max_len` from 200 to 150  
- Reduce `batch_size` from 64 to 32
- Reduce epochs from 50 to 30
- Skip GridSearchCV (use default parameters)

### If Out of Memory
- **Enable GPU first** (most important)
- Use smaller batch size (32 → 16 or 8)
- Reduce embedding dimensions (200 → 128 or 64)
- Reduce LSTM/CNN units by half
- Reduce max_words (15000 → 10000 → 5000)
- Process data in smaller batches

### If Accuracy < 95%
- Verify class balance in training data (should be ~37%/37%/26%)
- Increase training epochs (50 → 100)
- Try different ensemble weights
- Collect more diverse data from additional apps
- Adjust augmentation ratio (increase minority class samples)
- Fine-tune GridSearchCV parameter ranges

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