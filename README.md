# **Toxic Text Detection**

## **Overview**
This project focuses on detecting toxic comments from online text using **two machine learning approaches**:  
- **Multinomial Naive Bayes** – a generative model.  
- **Logistic Regression** – a discriminative model.

The goal is to **compare model performance** on speed, accuracy, and ability to handle class imbalance. Here, *toxic text* refers to language that is rude, abusive, or offensive.

## **Dataset**
Theassignment uses **three CSV files**:  
- **train.csv**
- **valid.csv**
- **test.csv**

## **Project Workflow**
### **1. Data Exploration**
- Identify text and label columns automatically.
- Check dataset shape, class distribution, and sample text patterns.

### **2. Data Cleaning**
- Handle missing values.
- Remove special tokens (`NEWLINE_TOKEN`, `EDATA_7`) and other artifacts.
- Normalize text to lowercase.
- Remove stopwords (via TF-IDF preprocessing).

### **3. Feature Extraction**
- **TF-IDF Vectorizer** with:
  - Lowercasing  
  - English stopword removal  
  - Unigrams & bigrams  
  - `max_features=15000`  
  - Filtering very common/rare terms

### **4. Model Training**
- **Naive Bayes**:
  - Tuned over `alpha = [0.5, 1.0]`  
  - Fast, simple, but assumes feature independence.
- **Logistic Regression**:
  - Tuned over `C = [0.1, 1.0]`  
  - Used `class_weight='balanced'` to handle imbalance.
  - Interpretable, better at capturing feature interactions.

### **5. Evaluation**
**Validation metrics:**

| Model                  | Accuracy | Weighted F1 |
|------------------------|----------|-------------|
| Naive Bayes (α=1.0)    | 0.8688   | 0.8079      |
| Logistic Regression (C=1.0) | 0.7400   | 0.7552      |

---

## **Key Observations**
- **Naive Bayes** predicts almost exclusively the majority class, leading to poor recall for toxic comments (`recall = 0.00` for class 1).  
- **Logistic Regression** with `class_weight='balanced'` improves minority class detection, but still misclassifies many cases and lowers overall accuracy.
- Dataset is **heavily imbalanced**, affecting performance metrics — **macro-F1** is more informative than accuracy here.

---

## **Output**
Final predictions on the **test.csv** include:
- `out_label_model_Gen` → Predictions from Naive Bayes  
- `out_label_model_Dis` → Predictions from Logistic Regression  
