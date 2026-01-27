#!/usr/bin/env python3
"""
Fix the sentiment_analysis_pipeline.ipynb notebook with all required corrections.
"""
import json
import copy

def load_notebook(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_notebook(notebook, filepath):
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

def create_cell(cell_type, source, metadata=None):
    """Create a new notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else [source]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def fix_notebook(notebook):
    """Apply all fixes to the notebook."""
    cells = notebook['cells']
    
    # Fix 1: Add preprocessing execution cell after cell 14
    print("Fix 1: Adding preprocessing execution cell after cell 14...")
    preprocessing_exec_cell = create_cell("code", [
        "# Apply preprocessing\n",
        "playstore_clean = preprocess_dataset(playstore_df, text_column='text')\n"
    ])
    cells.insert(15, preprocessing_exec_cell)
    print("  ✓ Added preprocessing execution cell at position 15")
    
    # Fix 2: Add train_logistic_regression() function before cell 27 (now 28 after insert)
    print("\nFix 2: Adding train_logistic_regression() baseline function...")
    lr_baseline_cell = create_cell("code", [
        "def train_logistic_regression(X_train, X_test, y_train, y_test, dataset_name='Dataset'):\n",
        "    \"\"\"\n",
        "    Train baseline Logistic Regression model.\n",
        "    \n",
        "    Args:\n",
        "        X_train, X_test: Feature matrices\n",
        "        y_train, y_test: Target labels\n",
        "        dataset_name: Name for reporting\n",
        "    \n",
        "    Returns:\n",
        "        Dictionary with model, predictions, and metrics\n",
        "    \"\"\"\n",
        "    print(f\"\\nTraining baseline Logistic Regression on {dataset_name}...\")\n",
        "    \n",
        "    # Train model with basic settings\n",
        "    model = LogisticRegression(\n",
        "        max_iter=1000,\n",
        "        random_state=42,\n",
        "        multi_class='multinomial',\n",
        "        solver='lbfgs'\n",
        "    )\n",
        "    \n",
        "    model.fit(X_train, y_train)\n",
        "    \n",
        "    # Make predictions\n",
        "    y_pred = model.predict(X_test)\n",
        "    y_pred_proba = model.predict_proba(X_test)\n",
        "    \n",
        "    # Calculate metrics\n",
        "    accuracy = accuracy_score(y_test, y_pred)\n",
        "    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)\n",
        "    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)\n",
        "    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)\n",
        "    \n",
        "    print(f\"  Accuracy: {accuracy:.4f}\")\n",
        "    print(f\"  Precision: {precision:.4f}\")\n",
        "    print(f\"  Recall: {recall:.4f}\")\n",
        "    print(f\"  F1-Score: {f1:.4f}\")\n",
        "    \n",
        "    return {\n",
        "        'model': model,\n",
        "        'y_pred': y_pred,\n",
        "        'y_pred_proba': y_pred_proba,\n",
        "        'accuracy': accuracy,\n",
        "        'precision': precision,\n",
        "        'recall': recall,\n",
        "        'f1': f1,\n",
        "        'dataset': dataset_name\n",
        "    }\n",
        "\n",
        "print('Baseline Logistic Regression function defined!')\n"
    ])
    # Find cell 27 (which defines train_improved_logistic_regression) - now at 28
    cells.insert(28, lr_baseline_cell)
    print("  ✓ Added train_logistic_regression() function before improved version")
    
    # Fix 3: Add train_basic_lstm() function before advanced LSTM (find where AttentionLayer is defined)
    print("\nFix 3: Adding train_basic_lstm() function...")
    basic_lstm_cell = create_cell("code", [
        "def train_basic_lstm(X_train, X_test, y_train, y_test, vocab_size, max_length=200, dataset_name='Dataset'):\n",
        "    \"\"\"\n",
        "    Train basic LSTM model with standard architecture.\n",
        "    \n",
        "    Args:\n",
        "        X_train, X_test: Padded sequences\n",
        "        y_train, y_test: Target labels\n",
        "        vocab_size: Size of vocabulary\n",
        "        max_length: Maximum sequence length\n",
        "        dataset_name: Name for reporting\n",
        "    \n",
        "    Returns:\n",
        "        Dictionary with model, predictions, and metrics\n",
        "    \"\"\"\n",
        "    print(f\"\\nTraining Basic LSTM on {dataset_name}...\")\n",
        "    \n",
        "    # Determine number of classes\n",
        "    num_classes = len(np.unique(y_train))\n",
        "    \n",
        "    # Build model\n",
        "    model = Sequential([\n",
        "        Embedding(vocab_size, 128, input_length=max_length),\n",
        "        LSTM(128, dropout=0.2, recurrent_dropout=0.2),\n",
        "        Dense(64, activation='relu'),\n",
        "        Dropout(0.5),\n",
        "        Dense(num_classes, activation='softmax')\n",
        "    ])\n",
        "    \n",
        "    model.compile(\n",
        "        optimizer='adam',\n",
        "        loss='sparse_categorical_crossentropy',\n",
        "        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]\n",
        "    )\n",
        "    \n",
        "    # Callbacks\n",
        "    early_stopping = EarlyStopping(\n",
        "        monitor='val_loss',\n",
        "        patience=3,\n",
        "        restore_best_weights=True\n",
        "    )\n",
        "    \n",
        "    # Train model\n",
        "    history = model.fit(\n",
        "        X_train, y_train,\n",
        "        validation_split=0.2,\n",
        "        epochs=10,\n",
        "        batch_size=32,\n",
        "        callbacks=[early_stopping],\n",
        "        verbose=0\n",
        "    )\n",
        "    \n",
        "    # Make predictions\n",
        "    y_pred_proba = model.predict(X_test, verbose=0)\n",
        "    y_pred = np.argmax(y_pred_proba, axis=1)\n",
        "    \n",
        "    # Calculate metrics\n",
        "    accuracy = accuracy_score(y_test, y_pred)\n",
        "    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)\n",
        "    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)\n",
        "    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)\n",
        "    \n",
        "    print(f\"  Accuracy: {accuracy:.4f}\")\n",
        "    print(f\"  Precision: {precision:.4f}\")\n",
        "    print(f\"  Recall: {recall:.4f}\")\n",
        "    print(f\"  F1-Score: {f1:.4f}\")\n",
        "    \n",
        "    return {\n",
        "        'model': model,\n",
        "        'y_pred': y_pred,\n",
        "        'y_pred_proba': y_pred_proba,\n",
        "        'accuracy': accuracy,\n",
        "        'precision': precision,\n",
        "        'recall': recall,\n",
        "        'f1': f1,\n",
        "        'history': history.history,\n",
        "        'dataset': dataset_name\n",
        "    }\n",
        "\n",
        "def build_bidirectional_lstm_with_attention(vocab_size, embedding_dim=128, max_length=200, num_classes=3):\n",
        "    \"\"\"\n",
        "    Build Bidirectional LSTM with Attention mechanism.\n",
        "    \"\"\"\n",
        "    print(f\"Building BiLSTM with Attention (vocab_size={vocab_size}, embedding_dim={embedding_dim})...\")\n",
        "    \n",
        "    # Input layer\n",
        "    inputs = Input(shape=(max_length,))\n",
        "    \n",
        "    # Embedding layer\n",
        "    x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inputs)\n",
        "    \n",
        "    # Bidirectional LSTM\n",
        "    lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)\n",
        "    \n",
        "    # Attention layer\n",
        "    attention_out = AttentionLayer()(lstm_out)\n",
        "    \n",
        "    # Dense layers\n",
        "    x = Dense(64, activation='relu')(attention_out)\n",
        "    x = Dropout(0.5)(x)\n",
        "    outputs = Dense(num_classes, activation='softmax')(x)\n",
        "    \n",
        "    model = Model(inputs=inputs, outputs=outputs)\n",
        "    \n",
        "    model.compile(\n",
        "        optimizer='adam',\n",
        "        loss='sparse_categorical_crossentropy',\n",
        "        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]\n",
        "    )\n",
        "    \n",
        "    return model\n",
        "\n",
        "print('Basic LSTM and BiLSTM functions defined!')\n"
    ])
    # Insert before AttentionLayer definition (cell 31, now 33)
    cells.insert(33, basic_lstm_cell)
    print("  ✓ Added train_basic_lstm() and build_bidirectional_lstm_with_attention() functions")
    
    # Fix 4: Add Feature Extraction Section before Section 4 (cell 36, now 40)
    print("\nFix 4: Adding Feature Extraction Section...")
    feature_extraction_header = create_cell("markdown", [
        "## 3.8 Feature Extraction with TF-IDF\n",
        "\n",
        "Extract features from cleaned text using TF-IDF vectorization for classical ML models.\n"
    ])
    feature_extraction_code = create_cell("code", [
        "# Initialize TF-IDF vectorizer\n",
        "print(\"Extracting TF-IDF features...\")\n",
        "tfidf_vectorizer = TfidfVectorizer(\n",
        "    max_features=5000,\n",
        "    ngram_range=(1, 2),\n",
        "    min_df=2,\n",
        "    max_df=0.8\n",
        ")\n",
        "\n",
        "# Prepare data for training\n",
        "X = playstore_clean['cleaned_text']\n",
        "y = playstore_clean['sentiment_encoded']\n",
        "\n",
        "# Split data\n",
        "X_train_text, X_test_text, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.3, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "# Fit and transform\n",
        "X_train = tfidf_vectorizer.fit_transform(X_train_text)\n",
        "X_test = tfidf_vectorizer.transform(X_test_text)\n",
        "\n",
        "print(f\"\\nTF-IDF Features extracted!\")\n",
        "print(f\"  Training set: {X_train.shape}\")\n",
        "print(f\"  Test set: {X_test.shape}\")\n",
        "print(f\"  Number of features: {X_train.shape[1]}\")\n",
        "print(f\"\\nClass distribution:\")\n",
        "print(f\"  Training: {np.bincount(y_train)}\")\n",
        "print(f\"  Test: {np.bincount(y_test)}\")\n"
    ])
    cells.insert(40, feature_extraction_header)
    cells.insert(41, feature_extraction_code)
    print("  ✓ Added TF-IDF Feature Extraction section")
    
    # Fix 5: Add Sequential Data Preparation section after TF-IDF
    print("\nFix 5: Adding Sequential Data Preparation section...")
    seq_prep_header = create_cell("markdown", [
        "## 3.9 Sequential Data Preparation for Deep Learning\n",
        "\n",
        "Tokenize and pad sequences for LSTM and CNN models.\n"
    ])
    seq_prep_code = create_cell("code", [
        "# Prepare sequential data for deep learning models\n",
        "print(\"\\nPreparing sequential data for deep learning models...\")\n",
        "\n",
        "# Initialize tokenizer\n",
        "MAX_WORDS = 5000\n",
        "MAX_LENGTH = 200\n",
        "\n",
        "tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')\n",
        "tokenizer.fit_on_texts(X_train_text)\n",
        "\n",
        "# Convert texts to sequences\n",
        "X_train_seq = tokenizer.texts_to_sequences(X_train_text)\n",
        "X_test_seq = tokenizer.texts_to_sequences(X_test_text)\n",
        "\n",
        "# Pad sequences\n",
        "X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')\n",
        "X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')\n",
        "\n",
        "vocab_size = min(MAX_WORDS, len(tokenizer.word_index)) + 1\n",
        "\n",
        "print(f\"\\nSequential data prepared!\")\n",
        "print(f\"  Vocabulary size: {vocab_size}\")\n",
        "print(f\"  Max sequence length: {MAX_LENGTH}\")\n",
        "print(f\"  Training sequences: {X_train_padded.shape}\")\n",
        "print(f\"  Test sequences: {X_test_padded.shape}\")\n",
        "\n",
        "# Store for later use\n",
        "print(f\"\\nData prepared for model training!\")\n"
    ])
    cells.insert(42, seq_prep_header)
    cells.insert(43, seq_prep_code)
    print("  ✓ Added Sequential Data Preparation section")
    
    # Fix 6 & 7: Update model training cells to use correct data and standardize keys
    print("\nFix 6 & 7: Updating model training cells...")
    
    # Now we need to find and update cells 43, 47, 49, 51 (baseline LR, basic LSTM, BiLSTM, CNN)
    # But with all our insertions, positions have changed. Let's search by content.
    
    for idx, cell in enumerate(cells):
        if cell['cell_type'] == 'code' and cell['source']:
            source_text = ''.join(cell['source'])
            
            # Fix cell that calls train_logistic_regression (was cell 43)
            if 'TRAINING MODEL 1: BASELINE LOGISTIC REGRESSION' in source_text:
                print(f"  Updating cell {idx}: Baseline LR training call...")
                cells[idx]['source'] = [
                    "print(\"\\n\" + \"=\"*70)\n",
                    "print(\"TRAINING MODEL 1: BASELINE LOGISTIC REGRESSION\")\n",
                    "print(\"=\"*70)\n",
                    "\n",
                    "baseline_lr_results = train_logistic_regression(\n",
                    "    X_train, X_test, y_train, y_test,\n",
                    "    dataset_name='Playstore Reviews'\n",
                    ")\n",
                    "\n",
                    "print(\"\\n✓ Baseline Logistic Regression training complete!\")\n"
                ]
            
            # Fix cell that calls train_basic_lstm (was cell 47)
            elif 'TRAINING MODEL 3: BASIC LSTM' in source_text and 'train_basic_lstm' not in source_text:
                print(f"  Updating cell {idx}: Basic LSTM training call...")
                cells[idx]['source'] = [
                    "print(\"\\n\" + \"=\"*70)\n",
                    "print(\"TRAINING MODEL 3: BASIC LSTM\")\n",
                    "print(\"=\"*70)\n",
                    "\n",
                    "basic_lstm_results = train_basic_lstm(\n",
                    "    X_train_padded, X_test_padded, y_train, y_test,\n",
                    "    vocab_size=vocab_size,\n",
                    "    max_length=MAX_LENGTH,\n",
                    "    dataset_name='Playstore Reviews'\n",
                    ")\n",
                    "\n",
                    "print(\"\\n✓ Basic LSTM training complete!\")\n"
                ]
            
            # Fix BiLSTM training call (was cell 49)
            elif 'TRAINING MODEL 4: BiLSTM WITH ATTENTION' in source_text:
                print(f"  Updating cell {idx}: BiLSTM training call...")
                # Extract the existing code and update data references
                new_source = source_text.replace('X_train, X_test', 'X_train_padded, X_test_padded')
                new_source = new_source.replace("'probabilities'", "'y_pred_proba'")
                cells[idx]['source'] = new_source.split('\n')
                cells[idx]['source'] = [line + '\n' for line in cells[idx]['source']]
            
            # Fix CNN training call (was cell 51)
            elif 'TRAINING MODEL 5: MULTI-FILTER CNN' in source_text:
                print(f"  Updating cell {idx}: CNN training call...")
                new_source = source_text.replace('X_train, X_test', 'X_train_padded, X_test_padded')
                new_source = new_source.replace("'probabilities'", "'y_pred_proba'")
                cells[idx]['source'] = new_source.split('\n')
                cells[idx]['source'] = [line + '\n' for line in cells[idx]['source']]
            
            # Fix ensemble call to use y_pred_proba
            elif 'create_ensemble_predictions' in source_text and 'probabilities' in source_text:
                print(f"  Updating cell {idx}: Ensemble predictions call...")
                new_source = source_text.replace("'probabilities'", "'y_pred_proba'")
                cells[idx]['source'] = new_source.split('\n')
                cells[idx]['source'] = [line + '\n' for line in cells[idx]['source']]
            
            # Fix any training function that returns 'probabilities' instead of 'y_pred_proba'
            elif 'def train_' in source_text and "'probabilities':" in source_text:
                print(f"  Updating cell {idx}: Standardizing return key to y_pred_proba...")
                new_source = source_text.replace("'probabilities':", "'y_pred_proba':")
                cells[idx]['source'] = new_source.split('\n')
                cells[idx]['source'] = [line + '\n' for line in cells[idx]['source']]
    
    print("  ✓ Updated all model training calls")
    
    return notebook

def main():
    print("="*70)
    print("FIXING SENTIMENT ANALYSIS NOTEBOOK")
    print("="*70)
    
    # Load notebook
    print("\nLoading notebook...")
    notebook = load_notebook('sentiment_analysis_pipeline.ipynb')
    print(f"  Loaded {len(notebook['cells'])} cells")
    
    # Apply fixes
    print("\nApplying fixes...\n")
    fixed_notebook = fix_notebook(notebook)
    
    # Save fixed notebook
    print("\nSaving fixed notebook...")
    save_notebook(fixed_notebook, 'sentiment_analysis_pipeline.ipynb')
    print(f"  Saved {len(fixed_notebook['cells'])} cells")
    
    print("\n" + "="*70)
    print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
    print("="*70)
    print(f"\nTotal cells: {len(fixed_notebook['cells'])}")

if __name__ == '__main__':
    main()
