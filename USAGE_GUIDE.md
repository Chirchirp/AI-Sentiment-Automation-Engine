# Usage Guide - AI Sentiment Automation Engine

## Complete Walkthrough

### Scenario: E-commerce Customer Feedback Analysis

You work as a data analyst for an e-commerce company. Every day you receive customer feedback that needs sentiment labelling for reporting.

---

## Part 1: Initial Setup (One-Time)

### Step 1: Collect Training Data

You export historical customer reviews that were manually labelled:

```csv
text,sentiment,product_category,date
"Absolutely love this product! Fast shipping too.",positive,electronics,2024-01-10
"Very disappointed. Quality is poor.",negative,clothing,2024-01-11
"It's okay. Does the job.",neutral,home,2024-01-12
"Best purchase ever! Highly recommend.",positive,electronics,2024-01-13
"Terrible customer service. Won't buy again.",negative,electronics,2024-01-14
...
(500 more rows)
```

**Save as:** `historical_reviews_labelled.csv`

### Step 2: Train Your Model

1. **Open the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Go to "A. Model Training" tab**

3. **Upload:** `historical_reviews_labelled.csv`

4. **Select columns:**
   - Text column: `text`
   - Sentiment label column: `sentiment`

5. **Click "Train Classifier"**

6. **Review results:**
   ```
   ✅ Model trained successfully!
   
   Accuracy: 92.5%
   Precision: 91.8%
   Recall: 92.1%
   F1-Score: 91.9%
   
   Trained on: 400 samples
   Tested on: 100 samples
   ```

7. **Model saved automatically** to `models/` directory

---

## Part 2: Daily Operations (Zero Cost)

### Day 1: Process New Feedback

You receive today's customer feedback (unlabelled):

```csv
text,customer_id,order_id,product_category
"Great product, arrived quickly",C1001,O5001,electronics
"Not what I expected from the photos",C1002,O5002,clothing
"Decent quality for the price",C1003,O5003,home
"Worst purchase I've made",C1004,O5004,electronics
"Exactly as described, very happy",C1005,O5005,home
...
(1000 more rows)
```

**Save as:** `daily_feedback_2024-02-16.csv`

### Process in Auto-Labelling Section

1. **Go to "B. Auto-Labelling" tab**

2. **Upload:** `daily_feedback_2024-02-16.csv`

3. **Select text column:** `text`

4. **Click "Predict Sentiments"**

5. **View results:**
   ```
   ✅ Labelled 1000 rows successfully!
   
   Sentiment Distribution:
   Positive: 652
   Neutral: 241
   Negative: 107
   
   Average Confidence: 87.3%
   ```

6. **Download labelled file:**
   - Click "Download Labelled Data (CSV)"
   - File includes: original columns + `sentiment` + `confidence` + probabilities

### Output File Example

```csv
text,customer_id,order_id,product_category,sentiment,confidence,prob_positive,prob_neutral,prob_negative
"Great product, arrived quickly",C1001,O5001,electronics,positive,0.95,0.95,0.03,0.02
"Not what I expected from the photos",C1002,O5002,clothing,negative,0.82,0.08,0.10,0.82
"Decent quality for the price",C1003,O5003,home,neutral,0.78,0.15,0.78,0.07
```

### Use in Your Reporting

Import to Excel, Power BI, Tableau, or Python:

```python
import pandas as pd

df = pd.read_csv('labelled_daily_feedback_20240216.csv')

# Daily sentiment report
daily_stats = df.groupby('sentiment').agg({
    'customer_id': 'count',
    'confidence': 'mean'
})

print(daily_stats)
```

**Time taken:** 30 seconds for 1000 rows  
**Cost:** $0.00

---

## Part 3: Quality Assurance

### Validate Predictions (If Ground Truth Available)

Sometimes you have both unlabelled data AND verified labels (for QA):

```csv
text,verified_sentiment
"Excellent service!",positive
"Mediocre product",neutral
```

When you upload this:
1. Select text column: `text`
2. Select ground truth column: `verified_sentiment`
3. After prediction, see evaluation:

```
📈 Evaluation Metrics
Accuracy: 94.2%
Precision: 93.8%
Recall: 94.1%
F1-Score: 93.9%
```

This tells you how well your model performs on new data.

---

## Part 4: Model Improvement

### Week 3: Identify Mislabelled Cases

Review predictions with low confidence:

```python
df = pd.read_csv('labelled_daily_feedback.csv')

# Find low confidence predictions
low_conf = df[df['confidence'] < 0.6]
print(low_conf[['text', 'sentiment', 'confidence']])
```

Output:
```
text                                    sentiment  confidence
"Not bad, not great either"             neutral    0.58
"I guess it's fine"                     neutral    0.55
"Could be better but works"             neutral    0.52
```

### Manually Review and Correct

Create corrected dataset:

```csv
text,sentiment
"Not bad, not great either",neutral
"I guess it's fine",positive
"Could be better but works",neutral
```

### Retrain Model

1. **Combine old training data + new corrections:**
   ```python
   old_data = pd.read_csv('historical_reviews_labelled.csv')
   corrections = pd.read_csv('manual_corrections.csv')
   
   combined = pd.concat([old_data, corrections])
   combined.to_csv('updated_training_data.csv', index=False)
   ```

2. **Upload `updated_training_data.csv` to Model Training section**

3. **Retrain:**
   ```
   Old accuracy: 92.5%
   New accuracy: 94.8%
   
   ✅ Model improved by 2.3%!
   ```

4. **Continue daily predictions with better model**

---

## Part 5: Using the Chatbot

### Without LLM (Local Logic Only)

Ask basic questions:

**Q:** "How many total rows?"  
**A:** [Local Logic] There are 1000 total rows in the dataset.

**Q:** "What's the accuracy?"  
**A:** [Local Logic] The model accuracy is 94.80%

**Q:** "Show sentiment distribution"  
**A:** [Local Logic] Sentiment distribution: {'positive': 652, 'neutral': 241, 'negative': 107}

**Q:** "What's the average confidence?"  
**A:** [Local Logic] Average confidence score: 87.30%

### With LLM (Groq API)

For complex analysis, add your Groq API key in the sidebar.

**Q:** "Why do we have more negative sentiments in electronics than clothing?"  
**A:** [LLM] Based on the data, electronics categories show 15.2% negative sentiment vs 8.1% in clothing. This could indicate: 1) Higher customer expectations for electronics, 2) More technical issues with electronic products, 3) Shipping damage concerns for delicate items. I'd recommend analyzing the specific complaints in electronics reviews to identify common issues.

**Q:** "What trends do you see in the neutral feedback?"  
**A:** [LLM] Neutral sentiments (24.1% of total) often contain phrases like "okay", "decent", "does the job" - suggesting adequate but not exceptional experiences. These represent opportunities for improvement, as they're one step away from positive reviews.

**Cost:** ~$0.0001 per question

---

## Part 6: Advanced Workflows

### Automated Pipeline

Create a Python script for daily automation:

```python
"""
daily_sentiment_pipeline.py
Run this script daily via cron or Task Scheduler
"""

import pandas as pd
import pickle
import sys
from datetime import datetime

# Load model
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load today's data
input_file = sys.argv[1]  # e.g., "daily_feedback.csv"
df = pd.read_csv(input_file)

# Predict
texts = df['text'].tolist()
X = vectorizer.transform(texts)
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Add results
df['sentiment'] = predictions
df['confidence'] = probabilities.max(axis=1)

# Save
output_file = f"labelled_{datetime.now().strftime('%Y%m%d')}.csv"
df.to_csv(output_file, index=False)

print(f"✅ Processed {len(df)} rows. Saved to {output_file}")
```

**Run daily:**
```bash
python daily_sentiment_pipeline.py today_feedback.csv
```

### Integration with BI Tools

**Power BI:**
1. Use "Get Data" → "Text/CSV"
2. Load labelled file
3. Create visualizations:
   - Sentiment over time
   - Sentiment by category
   - Confidence distribution

**Tableau:**
1. Connect to CSV
2. Create calculated field for sentiment colors
3. Build dashboard

**Google Sheets:**
1. Import CSV
2. Use conditional formatting for sentiments
3. Create charts

---

## Part 7: Troubleshooting Real Scenarios

### Issue: Low Confidence Scores

**Symptoms:**
```
Average Confidence: 62%
(should be >80%)
```

**Causes & Solutions:**

1. **Insufficient training data**
   - Solution: Collect 500-1000+ labelled samples

2. **Imbalanced classes**
   - Check distribution:
   ```python
   training_data['sentiment'].value_counts()
   # Output: positive: 800, negative: 150, neutral: 50
   ```
   - Solution: Collect more neutral/negative samples

3. **Domain mismatch**
   - Training: Product reviews
   - Prediction: Customer service tickets
   - Solution: Retrain with domain-specific data

### Issue: Model Overfitting

**Symptoms:**
```
Training accuracy: 98%
Test accuracy: 75%
```

**Solution:**
Modify `streamlit_app.py`:
```python
# Increase regularization
model = LogisticRegression(
    C=0.1,  # Lower C = more regularization (was 1.0)
    max_iter=1000
)
```

### Issue: Slow Predictions

**Symptoms:**
- 10,000 rows takes >2 minutes

**Solutions:**

1. **Reduce TF-IDF features:**
   ```python
   vectorizer = TfidfVectorizer(
       max_features=2000,  # Reduce from 5000
       ngram_range=(1, 1)  # Remove bigrams
   )
   ```

2. **Process in batches:**
   ```python
   batch_size = 1000
   for i in range(0, len(texts), batch_size):
       batch = texts[i:i+batch_size]
       # predict batch
   ```

---

## Part 8: Monitoring & Maintenance

### Weekly Review Checklist

- [ ] Check average confidence score (should be >80%)
- [ ] Review 10-20 random predictions
- [ ] Track sentiment distribution trends
- [ ] Collect mislabelled cases for retraining

### Monthly Tasks

- [ ] Retrain model with new labelled data
- [ ] Compare new vs old accuracy
- [ ] Update documentation with new use cases
- [ ] Review chatbot usage patterns

### Quarterly Goals

- [ ] Achieve 95%+ accuracy
- [ ] Process 100,000+ predictions/month
- [ ] Zero prediction costs
- [ ] <1% manual correction rate

---

## Summary

**Daily Workflow:**
1. Upload unlabelled data (30 sec)
2. Predict sentiments (30 sec)
3. Download results (5 sec)
4. Use in reporting (5 min)

**Total time:** <10 minutes  
**Cost:** $0.00  
**Accuracy:** 90-95%  

**Weekly Workflow:**
1. Review predictions (15 min)
2. Identify corrections (10 min)
3. Update training data (5 min)

**Monthly Workflow:**
1. Retrain model (5 min)
2. Evaluate improvement (10 min)
3. Deploy new model (1 min)

**Result:** Self-improving system that gets better over time!
