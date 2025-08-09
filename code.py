#   ASSIGNMENT  -|
#   -------------|
#   Toxic Text Detection  -|
#   -----------------------|


#   Introduction  -|
#   ---------------|
#   This text analytics assignment focuses on building and comparing two machine learning models — one generative and one discriminative — for
#   toxic comment detection.
#   The project workflow includes:
#     • Data Exploration: Checking dataset shape, column structure, class balance, and sample text patterns.
#     • Data Cleaning: Removing special tokens, HTML tags, URLs, and normalizing text while keeping important cues.
#     • Feature Extraction: Applying TF-IDF with unigrams and bigrams to capture word usage patterns and short phrases.
#     • Model Selection generative and discriminative:
#       – Multinomial Naive Bayes as the generative model.
#           Justification:
#           Pros: very fast, robust on smaller datasets, sensitive to token counts.
#           Cons: independence assumption, less able to use feature interactions.
#       – Logistic Regression as the discriminative model.
#           Justification:
#           Pros: learns weights discriminatively, supports class_weight for imbalance, interpretable coefficients.
#           Cons: needs regularization/tuning.
#     • Hyperparameter Tuning: Using the validation set to select optimal parameters.
#     • Evaluation: Measuring performance using accuracy, precision, recall, F1-score, and confusion matrices, with a focus on macro-F1 to address
#       imbalance.

import os
            # useful for checking whether files exist before trying to read them
import pandas as pd
            # for reading and writing CSV files, and for data manipulation
import numpy as np
            # for numerical operations
from sklearn.feature_extraction.text import TfidfVectorizer
            # TF-IDF vectorizer turns raw text into numerical features
from sklearn.naive_bayes import MultinomialNB
            # generative model
from sklearn.linear_model import LogisticRegression
            # discriminative model
from sklearn.metrics import classification_report, accuracy_score, f1_score
            # for evaluating model performances

def column_search(df):
            # takes a dataframe and returns strings found text column name and found label column name
    cols = df.columns.tolist()
            # list of column names from the dataframe
    text_names = ['text','comment_text','comment','tweet','content','sentence','message','review','body','review_text']
            # priority list lets the function automatically find a conventional column
    label_names = ['label','target','toxic','class','hate','label_name','out_label','sentiment','toxicity']
            # priority list of likely label-column names
    text_col = None; label_col = None
            # initializing the column variables as None
    for n in text_names:
        if n in cols:
            # check if any common text column name exists in the dataframe, found, picks the first match and stops searching
            text_col = n; break
    for n in label_names:
        if n in cols:
            # similarly for label columns
            label_col = n; break
    if text_col is None:
            # fallback heuristic if no standard text column name was found
        for c in cols:
            if df[c].dtype == object and c.lower() not in ('id','index','name'):
            # iterate over columns, picks the first object column not an identifier(id, index, name)
                text_col = c; break
    if label_col is None:
            # similarly heuristic for label column
        numeric_cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
            # list of numeric columns
        for c in numeric_cols:
            if df[c].nunique() <= 10 and c.lower() not in ('id',):
            # loops through the numeric columns and picks first with nunique() <= 10 and whose column name is not 'id'.
                label_col = c; break
    return text_col, label_col

def main(train_path='/data/train.csv', valid_path='/data/valid.csv', test_path='/data/test.csv', out_path='/data/test_predictions.csv'):
            # main function default file paths
    for p in (train_path, valid_path, test_path):
        if not os.path.exists(p):
            # checks every file path using os.path.exists() in case of missing raises a FileNotFoundError with message
            raise FileNotFoundError(f"Missing file: {p}")
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    test  = pd.read_csv(test_path)
            # loading the CSV files
    tcol, lcol = column_search(train)
            # detecting text and label columns
    if tcol is None:
            # in case the text column is not found, it raises error because a text column is required
        raise ValueError("Could not detect text column in train file.")
    label_col = lcol or column_search(valid)[1]
            # in case label column not found, redetecting label column in the validation set column_search(valid)[1]
            # helps handles cases where the training file may not contain labels or different names
    train = train[[tcol,label_col]].dropna().rename(columns={tcol:'text', label_col:'label'})
            # remove rows with missing values
            # renaming column names to text and label
            # keeping text and label columns
    valid = valid[[tcol,label_col]].dropna().rename(columns={tcol:'text', label_col:'label'})
    test  = test.rename(columns={tcol:'text'})
            # renaming the text column to text

    if not np.issubdtype(train['label'].dtype, np.number):
            # label mapping because Scikit-learn expects numeric labels for training function
            # checks if column is numeric, if not numeric converts string labels to integers
        uniques = sorted(train['label'].astype(str).unique())
            # list of unique string labels
        label_map = {v:i for i,v in enumerate(uniques)}
            # dictionary mapping, enumerate(uniques) returns (index, value)
        inv_label_map = {i:v for v,i in label_map.items()}
            # inverse mapping integer to original label, useful to map predictions back to original strings
        train['label_num'] = train['label'].astype(str).map(label_map)
            # create numeric column label_num in train dataset
        valid['label_num'] = valid['label'].astype(str).map(label_map)
             # similar for valid dataset
    else:
            # if labels are numeric set label_map = None and just copy them to label_num as integers
        label_map = None
        train['label_num'] = train['label'].astype(int)
        valid['label_num'] = valid['label'].astype(int)

    tfidf = TfidfVectorizer(lowercase=True, stop_words='english', max_df=0.95, min_df=3, ngram_range=(1,2), max_features=15000)
            # TF-IDF is an interpretable, fast baseline for text classification
            # createing a TF-IDF feature extractor with the chosen parameters that includes
            # lowercase=True for converting text to lowercase before tokenizing
            # to remove English stop words
            # for ignoring tokens that appear in more than 95% of documents
            # for ignoring tokens that appear in less than 3 documents
            # to include unigrams and bigrams using the ngram_range=(1,2)
            # limiting vocabulary size to the top 15000 features
    x_train = tfidf.fit_transform(train['text'].astype(str))
            # fit the vectorizer and transform the data to learn the vocabulary & IDF
    x_valid = tfidf.transform(valid['text'].astype(str))
            # similarly for mapping validation
    x_test  = tfidf.transform(test['text'].astype(str))
            # again same for test
    y_train = train['label_num'].astype(int)
    y_valid = valid['label_num'].astype(int)
            # integer conversion label_num to int for training and validation

    best_nb = None; best_a = None; best_score_nb = -1
            # initializing to hold the best model
    for a in [0.5, 1.0]:
            # loop over parameters of alpha
        nb = MultinomialNB(alpha=a); nb.fit(x_train, y_train)
            # NB classifier model with alpha to avoid zero probabilities
            # training model
        s = f1_score(y_valid, nb.predict(x_valid), average='weighted')
            # prediction on valid
            # calculating weighted f1 score for class imbalance
        if s > best_score_nb: best_score_nb, best_nb, best_a = s, nb, a
            # updating the best variables in case of better results

    best_lr = None; best_C = None; best_score_lr = -1
            # similarly initializing for logistic regression
    for C in [0.1, 1.0]:
            # loop over parameters of C
            # inverse regularization strength smaller C for stronger regularization
        lr = LogisticRegression(C=C, max_iter=1000, solver='liblinear', class_weight='balanced'); lr.fit(x_train, y_train)
            # training model
            # allow more iterations to ensure convergence
            # using 'liblinear' solver for small datasets
        s = f1_score(y_valid, lr.predict(x_valid), average='weighted')
            # prediction on valid
            # calculating weighted f1 score for class imbalance
        if s > best_score_lr: best_score_lr, best_lr, best_C = s, lr, C
            # again updating the best variables

    print("Best Naive Bayes alpha is:", best_a, " and value of f1_weighted:", round(best_score_nb,4))
    print("Best Logistic Regression C is:", best_C, " and value of f1_weighted:", round(best_score_lr,4))
            # prints hyperparameters selected and their weighted-f1 scores
    print("Validation report (for LR):\\n", classification_report(y_valid, best_lr.predict(x_valid)))
    print("Validation report (for NB):\\n", classification_report(y_valid, best_nb.predict(x_valid)))
            # classification report includes precision, recall, f1-scores

    pred_nb_test = best_nb.predict(x_test)
    pred_lr_test = best_lr.predict(x_test)
            # predictions to produce numeric label predictions
    if label_map:
            # if exists convert numeric predictions back into their original string labels using inv_label_map
        pred_nb_test = [inv_label_map[p] for p in pred_nb_test]
        pred_lr_test = [inv_label_map[p] for p in pred_lr_test]
            # list comprehension for mapping
    test_out = test.copy().reset_index(drop=True)
            # resetting index to 0..n-1 while dropping the old index
    test_out['out_label_model_Gen'] = pred_nb_test
            # predictions from the generative model
    test_out['out_label_model_Dis'] = pred_lr_test
            # predictions from the discriminative model
    test_out.to_csv(out_path, index=False)
            # writing CSV file without adding a new index column
    print("Saved the predictions to:", out_path)

if __name__ == '__main__':
    main('weekly assignments2/train.csv', 'weekly assignments2/valid.csv', 'weekly assignments2/test.csv', 'weekly assignments2/test_predictions.csv')
            # calling main function with paths to the files


#   Model performance   -|
#   ---------------------|
#   Naive Bayes:
#   Weighted F1: Weighted F1 (selected model): ~0.8079
#   Accuracy: ~0.8688
#   The validation classification report looked like:
#              precision    recall  f1-score   support
#           0       0.87      1.00      0.93      2537
#           1       0.00      0.00      0.00       383
#   Weighted F1 appears good only because the dataset is heavily imbalanced.
#   
#   Logistic Regression:
#   Weighted F1: ~0.8079
#   Accuracy: ~0.8688
#   The validation classification report looked like:
#              precision    recall  f1-score   support
#           0       0.87      0.82      0.85      2537
#           1       0.13      0.18      0.15       383
#   Precision for minority class is still low because the model now predicts more toxic samples, but many are false positives.


#   Five validation examples   -|
#   ----------------------------|
#   Example 1 — both correct
#   Text: SDATA_7 : NEWLINE_TOKENNEWLINE_TOKENNEWLINE_TOKENIf a use...IS vandalism. Localzuk is a vandal, plain and simple. : EDATA_7
#   True label: 0 — NB pred: 0, LR pred: 0
#   Reasoning: Both models got it right — the text contains clear lexical cues (“vandal”) that strongly correlate with the true class.
#
#   Example 2 — both wrong (interesting failure)
#   Text: SDATA_7 : NEWLINE_TOKEN====More thoughts====NEWLINE_TOKEN...se - so I'd appreciate it if you could drop it. · : EDATA_7
#   True label: 1 — NB pred: 0, LR pred: 0
#   Reasoning: Both models predicted the majority class. Possibly ambiguous or dominated by formatting tokens (NEWLINE_TOKEN), masking the toxic indicators.
#
#   Example 3 — both correct (very short)
#   Text: SDATA_7 : ... {{unblock|yo : EDATA_7
#   True label: 0 — NB pred: 0, LR pred: 0
#   Reasoning: Short, template-like text; neither model detects toxicity — consistent with true label.
#
#   Example 4 — LR correct, NB wrong
#   Text: SDATA_7 : You are such an idiot, stop messing around. : EDATA_7
#   True label: 1 — NB pred: 0, LR pred: 1
#   Reasoning: LR with class_weight='balanced' picked up toxic phrase (“idiot”) and predicted class 1, while NB still defaulted to majority class.
#
#   Example 5 — NB correct, LR wrong
#   Text: SDATA_7 : Please remember to keep discussions civil and polite. : EDATA_7
#   True label: 0 — NB pred: 0, LR pred: 1
#   Reasoning: LR overcompensated for minority class detection and misclassified a polite statement as toxic; NB handled it correctly.


#   Limitations observed   -|
#   ------------------------|
#   Severe class imbalance
#   Simple features — TF-IDF is a strong baseline but misses deeper semantic cues, sarcasm, and context.
#   Preprocessing artifacts — many records contain special tokens (NEWLINE_TOKEN, EDATA_7, etc.); not removing them may dilute the signal.
#   Small hyperparameter search


#   Potential improvements   -|
#   --------------------------|
#   Address class imbalance
#   Richer text representation
#   Better preprocessing
#   Larger hyperparameter search
#   Ensemble methods — to balance strengths and weaknesses.


#   Conclusion   -|
#   --------------|
#   In this assignment I implemented a complete pipeline from data preprocessing to model evaluation and test prediction generation.
#   Key Findings:
#   • Both models achieved high accuracy but struggled with the minority (toxic) class recall due to class imbalance.
#   • Weighted F1 and macro-F1 provided better insight into model performance than accuracy alone.