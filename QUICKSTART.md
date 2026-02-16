# Quick Start Guide

## 🚀 Get Running in 5 Minutes

### Step 1: Install (1 minute)

```bash
pip install -r requirements.txt
```

### Step 2: Run (30 seconds)

```bash
streamlit run streamlit_app.py
```

Your browser will open automatically at `http://localhost:8501`

### Step 3: Train Your First Model (2 minutes)

1. **Click** "A. Model Training" tab
2. **Upload** `sample_training_data.csv` (included)
3. **Select columns:**
   - Text column: `text`
   - Sentiment label column: `sentiment`
4. **Click** "🚀 Train Classifier"
5. **Wait** 5-10 seconds
6. **See** accuracy ~90%+ ✅

Model is now saved and ready!

### Step 4: Auto-Label Data (1 minute)

1. **Click** "B. Auto-Labelling" tab
2. **Upload** `sample_unlabelled_data.csv` (included)
3. **Select** text column: `text`
4. **Click** "🚀 Predict Sentiments"
5. **Download** labelled results

Done! You just:
- Trained a sentiment model
- Predicted sentiments on new data
- Downloaded results
- **Cost: $0.00**

---

## 📊 What You Built

A production-ready system that:

✅ Trains local ML models (no API needed)  
✅ Predicts sentiments offline (zero cost)  
✅ Processes thousands of rows in seconds  
✅ Improves over time with retraining  
✅ Uses LLM only for chatbot questions  

---

## 💡 Next Steps

### Daily Workflow

1. Receive new feedback data
2. Upload to "Auto-Labelling"
3. Click predict
4. Download results
5. Use in your reports

**Time:** 2 minutes  
**Cost:** $0.00

### Weekly Improvement

1. Review predictions
2. Collect corrections
3. Retrain model
4. Deploy updated model

### Optional: Enable Chatbot

1. Get free Groq API key: https://console.groq.com
2. Add key in sidebar
3. Ask questions about your data

---

## 📝 Common Questions

**Q: Do I need an API key?**  
A: No! Training and predictions are 100% local. API key only needed for optional chatbot.

**Q: How accurate is it?**  
A: Typically 85-95% with 500+ training samples. Improves with more data.

**Q: Can I process 10,000 rows?**  
A: Yes! Takes ~30 seconds. Completely offline.

**Q: What if I don't have labelled data?**  
A: Use the sample data to start, or manually label 100-500 examples.

**Q: Can I use this in production?**  
A: Yes! It's production-ready. Many companies use similar systems.

---

## 🎯 Real Example

**Input (unlabelled):**
```
"This product is amazing!"
"Not satisfied with quality"
"It's okay, nothing special"
```

**Output (labelled):**
```
text                          sentiment  confidence
"This product is amazing!"    positive   0.95
"Not satisfied with quality"  negative   0.88
"It's okay, nothing special"  neutral    0.82
```

**Time:** <1 second  
**Cost:** $0.00

---

## 🔧 Customization

Want to customize? Edit `streamlit_app.py`:

**Change model:**
```python
# Replace LogisticRegression with:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
```

**Adjust features:**
```python
vectorizer = TfidfVectorizer(
    max_features=10000,  # More features
    ngram_range=(1, 3),  # Add trigrams
)
```

**Add preprocessing:**
```python
texts = [text.lower().strip() for text in texts]
```

---

## 📚 Full Documentation

- **README.md** - Complete guide
- **USAGE_GUIDE.md** - Detailed examples
- **streamlit_app.py** - Well-commented code

---

## ✅ You're Ready!

You now have a **zero-cost, production-ready sentiment analysis system**.

**Start processing your data now!**

```bash
streamlit run streamlit_app.py
```
