"""
AI Sentiment Automation Engine
A self-improving sentiment analysis system with local ML and selective LLM usage
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import json

# Import ML components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Import Groq for chatbot
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Automation",
    page_icon="🤖",
    layout="wide"
)

# Constants
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'labelled_data' not in st.session_state:
    st.session_state.labelled_data = None
if 'model_stats' not in st.session_state:
    st.session_state.model_stats = None


def load_model():
    """Load trained model and vectorizer from disk"""
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return model, vectorizer, metadata
    except FileNotFoundError:
        return None, None, None


def save_model(model, vectorizer, metadata):
    """Save trained model and vectorizer to disk"""
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def train_sentiment_model(texts, labels):
    """
    Train TF-IDF + Logistic Regression sentiment classifier
    
    Args:
        texts: List of text strings
        labels: List of sentiment labels (positive, neutral, negative)
    
    Returns:
        model, vectorizer, metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Transform text to features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Detailed metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    return model, vectorizer, metrics


def predict_sentiment(model, vectorizer, texts):
    """
    Predict sentiment for texts using trained model
    
    Args:
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        texts: List of text strings
    
    Returns:
        predictions, probabilities
    """
    X_tfidf = vectorizer.transform(texts)
    predictions = model.predict(X_tfidf)
    probabilities = model.predict_proba(X_tfidf)
    
    return predictions, probabilities


def get_local_stats(df, text_column):
    """Compute comprehensive statistics for chatbot context"""
    stats = {
        'total_rows': len(df),
        'text_column': text_column,
        'avg_text_length': df[text_column].str.len().mean(),
        'min_text_length': df[text_column].str.len().min(),
        'max_text_length': df[text_column].str.len().max(),
        'all_columns': df.columns.tolist()
    }
    
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        stats['sentiment_distribution'] = sentiment_counts
        stats['most_common_sentiment'] = df['sentiment'].mode()[0]
        
        # Breakdown by category if available
        category_cols = [col for col in df.columns if 'category' in col.lower() or 'product' in col.lower()]
        if category_cols:
            for cat_col in category_cols:
                breakdown = df.groupby([cat_col, 'sentiment']).size().to_dict()
                stats[f'sentiment_by_{cat_col}'] = breakdown
        
        # Date-based analysis if date columns exist
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            for date_col in date_cols:
                try:
                    df_copy = df.copy()
                    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                    df_copy = df_copy.dropna(subset=[date_col])
                    if len(df_copy) > 0:
                        date_sentiment = df_copy.groupby(pd.Grouper(key=date_col, freq='D'))['sentiment'].value_counts().to_dict()
                        stats[f'sentiment_by_{date_col}'] = str(date_sentiment)
                        stats[f'{date_col}_range'] = f"{df_copy[date_col].min()} to {df_copy[date_col].max()}"
                except:
                    pass
    
    if 'confidence' in df.columns:
        stats['avg_confidence'] = df['confidence'].mean()
        stats['min_confidence'] = df['confidence'].min()
        stats['max_confidence'] = df['confidence'].max()
        
        # Low confidence items
        low_conf_count = len(df[df['confidence'] < 0.7])
        stats['low_confidence_count'] = low_conf_count
    
    return stats


def chat_with_local_logic(user_question, context_data):
    """
    Attempt to answer using local logic first
    
    Args:
        user_question: User's question
        context_data: Dictionary with stats and data
    
    Returns:
        answer (str) or None if LLM needed
    """
    question_lower = user_question.lower()
    
    # Handle basic questions locally
    if 'how many' in question_lower or 'total' in question_lower:
        if context_data and 'total_rows' in context_data:
            return f"There are {context_data['total_rows']} total rows in the dataset."
    
    if 'accuracy' in question_lower:
        if context_data and 'accuracy' in context_data:
            return f"The model accuracy is {context_data['accuracy']:.2%}"
    
    if 'distribution' in question_lower or 'breakdown' in question_lower:
        if context_data and 'sentiment_distribution' in context_data:
            dist = context_data['sentiment_distribution']
            return f"Sentiment distribution: {dist}"
    
    if 'average confidence' in question_lower or 'avg confidence' in question_lower:
        if context_data and 'avg_confidence' in context_data:
            return f"Average confidence score: {context_data['avg_confidence']:.2%}"
    
    if 'low confidence' in question_lower:
        if context_data and 'low_confidence_count' in context_data:
            return f"Found {context_data['low_confidence_count']} predictions with confidence below 70%"
    
    if 'columns' in question_lower or 'what data' in question_lower:
        if context_data and 'all_columns' in context_data:
            return f"Available columns: {', '.join(context_data['all_columns'])}"
    
    # For anything requiring analysis, return None to trigger LLM
    return None


def chat_with_llm(api_key, user_question, context_data, full_dataframe=None):
    """Use Groq LLM for complex questions with full data context"""
    if not GROQ_AVAILABLE or not api_key:
        return "LLM not available. Please provide a Groq API key."
    
    try:
        client = Groq(api_key=api_key)
        
        # Build comprehensive context
        context_str = "Dataset Analysis Context:\n"
        context_str += "="*50 + "\n"
        
        if context_data:
            # Basic stats
            if 'total_rows' in context_data:
                context_str += f"Total rows: {context_data['total_rows']}\n"
            
            if 'all_columns' in context_data:
                context_str += f"Columns: {', '.join(context_data['all_columns'])}\n"
            
            # Sentiment distribution
            if 'sentiment_distribution' in context_data:
                context_str += f"\nSentiment Distribution:\n"
                for sentiment, count in context_data['sentiment_distribution'].items():
                    pct = (count / context_data['total_rows'] * 100) if context_data['total_rows'] > 0 else 0
                    context_str += f"  {sentiment.title()}: {count} ({pct:.1f}%)\n"
            
            # Category breakdowns
            for key, value in context_data.items():
                if 'sentiment_by_' in key and 'date' not in key:
                    context_str += f"\n{key.replace('_', ' ').title()}:\n"
                    context_str += f"  {value}\n"
            
            # Date ranges
            for key, value in context_data.items():
                if '_range' in key:
                    context_str += f"{key.replace('_', ' ').title()}: {value}\n"
            
            # Confidence stats
            if 'avg_confidence' in context_data:
                context_str += f"\nConfidence Metrics:\n"
                context_str += f"  Average: {context_data['avg_confidence']:.2%}\n"
                if 'low_confidence_count' in context_data:
                    context_str += f"  Low confidence (<70%): {context_data['low_confidence_count']}\n"
        
        # Add sample data if available
        if full_dataframe is not None and len(full_dataframe) > 0:
            context_str += f"\nSample Data (first 5 rows):\n"
            context_str += full_dataframe.head(5).to_string(index=False)
            
            # Add negative examples if asking about negatives
            if 'negative' in user_question.lower():
                negatives = full_dataframe[full_dataframe['sentiment'] == 'negative'].head(3)
                if len(negatives) > 0:
                    context_str += f"\n\nSample Negative Sentiments:\n"
                    context_str += negatives.to_string(index=False)
        
        context_str += "\n" + "="*50 + "\n"
        
        prompt = f"""{context_str}

User Question: {user_question}

Provide a detailed, actionable analysis based on the data above. Include:
- Key insights from the data
- Patterns or trends identified
- Specific recommendations
- Focus on business value and actionable next steps

Keep your response concise but comprehensive."""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Use more powerful model for analysis
            messages=[
                {"role": "system", "content": "You are an expert data analyst specializing in sentiment analysis and business insights. Provide actionable, data-driven recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"


# ============================================================================
# MAIN UI
# ============================================================================

st.title("🤖 AI Sentiment Automation Engine")
st.markdown("**Self-improving sentiment analysis with local ML + selective LLM usage**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check for existing model
    model, vectorizer, metadata = load_model()
    
    if model is not None:
        st.success("✅ Model loaded")
        st.metric("Trained on", metadata.get('train_size', 'N/A'))
        st.metric("Accuracy", f"{metadata.get('accuracy', 0):.2%}")
        st.metric("Last trained", metadata.get('trained_at', 'Unknown'))
    else:
        st.warning("⚠️ No model found. Train one first!")
    
    st.markdown("---")
    
    # Groq API key for chatbot
    st.subheader("Chatbot (Optional)")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Only needed for chatbot")
    
    if groq_api_key:
        st.info("💬 Chatbot enabled")
    else:
        st.info("ℹ️ Chatbot will use local logic only")


# Main tabs
tab1, tab2, tab3 = st.tabs([
    "📚 A. Model Training",
    "🤖 B. Auto-Labelling",
    "💬 C. Chatbot"
])

# ============================================================================
# TAB 1: MODEL TRAINING
# ============================================================================

with tab1:
    st.header("📚 Model Training Section")
    st.markdown("Train or retrain your local sentiment classifier using labelled data.")
    
    st.subheader("1️⃣ Upload Labelled Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file with labelled data",
        type=['csv', 'xlsx', 'xls'],
        key='training_upload'
    )
    
    if uploaded_file:
        # Load file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("2️⃣ Select Columns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox(
                    "Text column:",
                    options=df.columns.tolist(),
                    key='train_text_col'
                )
            
            with col2:
                label_column = st.selectbox(
                    "Sentiment label column:",
                    options=df.columns.tolist(),
                    key='train_label_col'
                )
            
            # Validate labels
            if text_column and label_column:
                unique_labels = df[label_column].unique()
                st.write(f"**Found labels:** {', '.join(map(str, unique_labels))}")
                
                # Check if labels are valid
                valid_labels = {'positive', 'neutral', 'negative'}
                current_labels = set(df[label_column].str.lower().unique())
                
                if not current_labels.issubset(valid_labels):
                    st.warning(f"⚠️ Labels should be: positive, neutral, negative. Found: {current_labels}")
                    st.info("Normalizing labels to lowercase...")
                
                # Normalize labels
                df[label_column] = df[label_column].str.lower()
                
                # Filter to valid labels only
                df = df[df[label_column].isin(valid_labels)]
                
                st.write(f"**Valid rows after filtering:** {len(df)}")
                
                st.subheader("3️⃣ Train Model")
                
                if st.button("🚀 Train Classifier", type="primary"):
                    if len(df) < 10:
                        st.error("❌ Need at least 10 labelled samples to train")
                    else:
                        with st.spinner("Training model..."):
                            # Prepare data
                            texts = df[text_column].astype(str).tolist()
                            labels = df[label_column].tolist()
                            
                            # Train
                            model, vectorizer, metrics = train_sentiment_model(texts, labels)
                            
                            # Save model
                            metadata = {
                                'accuracy': metrics['accuracy'],
                                'precision': metrics['precision'],
                                'recall': metrics['recall'],
                                'f1_score': metrics['f1_score'],
                                'train_size': metrics['train_size'],
                                'test_size': metrics['test_size'],
                                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            save_model(model, vectorizer, metadata)
                            st.session_state.model_stats = metrics
                            
                            st.success("✅ Model trained and saved successfully!")
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Display model performance if available
    if st.session_state.model_stats:
        st.subheader("📊 Model Performance")
        
        metrics = st.session_state.model_stats
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
        
        st.markdown("#### Classification Report")
        
        # Format classification report
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df.style.format("{:.2%}"), use_container_width=True)
        
        st.markdown("#### Confusion Matrix")
        conf_matrix = np.array(metrics['confusion_matrix'])
        conf_df = pd.DataFrame(
            conf_matrix,
            index=['True Positive', 'True Neutral', 'True Negative'],
            columns=['Pred Positive', 'Pred Neutral', 'Pred Negative']
        )
        st.dataframe(conf_df, use_container_width=True)


# ============================================================================
# TAB 2: AUTO-LABELLING
# ============================================================================

with tab2:
    st.header("🤖 Auto-Labelling Section")
    st.markdown("Automatically label unlabelled data using the trained local model (no API calls).")
    
    # Check if model exists
    model, vectorizer, metadata = load_model()
    
    if model is None:
        st.error("❌ No trained model found. Please train a model first in the 'Model Training' section.")
    else:
        st.info(f"✅ Using model trained on {metadata.get('train_size', 'N/A')} samples with {metadata.get('accuracy', 0):.2%} accuracy")
        
        st.subheader("1️⃣ Upload Unlabelled Data")
        
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file (with or without labels)",
            type=['csv', 'xlsx', 'xls'],
            key='labelling_upload'
        )
        
        if uploaded_file:
            try:
                # Load file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded {len(df)} rows")
                st.dataframe(df.head(), use_container_width=True)
                
                st.subheader("2️⃣ Select Text Column")
                
                text_column = st.selectbox(
                    "Text column to analyze:",
                    options=df.columns.tolist(),
                    key='label_text_col'
                )
                
                # Check if ground truth exists
                has_ground_truth = 'sentiment' in df.columns or any('label' in col.lower() for col in df.columns)
                
                if has_ground_truth:
                    st.info("ℹ️ Ground truth labels detected. Evaluation metrics will be computed.")
                    ground_truth_col = st.selectbox(
                        "Ground truth label column (optional):",
                        options=['None'] + df.columns.tolist(),
                        key='ground_truth_col'
                    )
                else:
                    ground_truth_col = 'None'
                
                st.subheader("3️⃣ Auto-Label")
                
                if st.button("🚀 Predict Sentiments", type="primary"):
                    with st.spinner("Predicting sentiments..."):
                        # Prepare texts
                        texts = df[text_column].astype(str).tolist()
                        
                        # Predict
                        predictions, probabilities = predict_sentiment(model, vectorizer, texts)
                        
                        # Add to dataframe
                        df['sentiment'] = predictions
                        
                        # Add confidence (max probability)
                        df['confidence'] = probabilities.max(axis=1)
                        
                        # Add individual class probabilities
                        classes = model.classes_
                        for i, cls in enumerate(classes):
                            df[f'prob_{cls}'] = probabilities[:, i]
                        
                        st.session_state.labelled_data = df
                        
                        st.success(f"✅ Labelled {len(df)} rows successfully!")
                        
                        # Show results
                        st.subheader("📊 Results Preview (All Columns)")
                        st.info("💡 All original columns are preserved. Scroll right to see sentiment predictions.")
                        
                        # Reorder columns: original columns first, then sentiment columns at the end
                        original_cols = [col for col in df.columns if col not in ['sentiment', 'confidence', 'prob_positive', 'prob_neutral', 'prob_negative']]
                        sentiment_cols = ['sentiment', 'confidence'] + [col for col in df.columns if col.startswith('prob_')]
                        ordered_cols = original_cols + sentiment_cols
                        
                        st.dataframe(df[ordered_cols].head(20), use_container_width=True)
                        
                        # Sentiment distribution
                        st.markdown("#### Sentiment Distribution")
                        sentiment_counts = df['sentiment'].value_counts()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Positive", sentiment_counts.get('positive', 0))
                        with col2:
                            st.metric("Neutral", sentiment_counts.get('neutral', 0))
                        with col3:
                            st.metric("Negative", sentiment_counts.get('negative', 0))
                        
                        # Average confidence
                        st.metric("Average Confidence", f"{df['confidence'].mean():.2%}")
                        
                        # Show breakdown by category/product if exists
                        category_cols = [col for col in df.columns if 'category' in col.lower() or 'product' in col.lower()]
                        if category_cols:
                            st.markdown("#### Sentiment by Category")
                            for cat_col in category_cols[:2]:  # Show first 2 category columns
                                st.write(f"**{cat_col}:**")
                                breakdown = pd.crosstab(df[cat_col], df['sentiment'], margins=True)
                                st.dataframe(breakdown, use_container_width=True)
                        
                        # Evaluation metrics if ground truth exists
                        if ground_truth_col != 'None' and ground_truth_col in df.columns:
                            st.subheader("📈 Evaluation Metrics")
                            
                            # Normalize ground truth
                            df[ground_truth_col] = df[ground_truth_col].str.lower()
                            
                            # Compute metrics
                            accuracy = accuracy_score(df[ground_truth_col], df['sentiment'])
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                df[ground_truth_col], df['sentiment'], 
                                average='weighted', zero_division=0
                            )
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Accuracy", f"{accuracy:.2%}")
                            with col2:
                                st.metric("Precision", f"{precision:.2%}")
                            with col3:
                                st.metric("Recall", f"{recall:.2%}")
                            with col4:
                                st.metric("F1-Score", f"{f1:.2%}")
                            
                            # Confusion matrix
                            st.markdown("#### Confusion Matrix")
                            conf_matrix = confusion_matrix(
                                df[ground_truth_col], df['sentiment'],
                                labels=['positive', 'neutral', 'negative']
                            )
                            conf_df = pd.DataFrame(
                                conf_matrix,
                                index=['True Positive', 'True Neutral', 'True Negative'],
                                columns=['Pred Positive', 'Pred Neutral', 'Pred Negative']
                            )
                            st.dataframe(conf_df, use_container_width=True)
                
                # Download button
                if st.session_state.labelled_data is not None:
                    st.subheader("4️⃣ Download Results")
                    
                    # Prepare download
                    csv = st.session_state.labelled_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Labelled Data (CSV)",
                        data=csv,
                        file_name=f"labelled_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")


# ============================================================================
# TAB 3: CHATBOT
# ============================================================================

with tab3:
    st.header("💬 Advanced Chatbot Section")
    st.markdown("Ask questions about your sentiment analysis results. Performs deep analysis on entire dataset including products, dates, and trends.")
    
    # Gather context data
    context_data = {}
    full_df = None
    
    if metadata:
        context_data.update(metadata)
    
    if st.session_state.labelled_data is not None:
        full_df = st.session_state.labelled_data
        # Determine text column
        text_col = 'text' if 'text' in full_df.columns else full_df.columns[0]
        local_stats = get_local_stats(full_df, text_col)
        context_data.update(local_stats)
    
    if st.session_state.model_stats:
        context_data.update(st.session_state.model_stats)
    
    # Display available context
    with st.expander("📊 Available Data for Analysis"):
        if context_data:
            st.write("**Dataset Overview:**")
            if 'total_rows' in context_data:
                st.write(f"- Total rows: {context_data['total_rows']}")
            if 'all_columns' in context_data:
                st.write(f"- Columns: {', '.join(context_data['all_columns'])}")
            if 'sentiment_distribution' in context_data:
                st.write(f"- Sentiments: {context_data['sentiment_distribution']}")
            
            st.write("\n**Advanced Analysis Available:**")
            st.write("- Product/Category breakdowns")
            st.write("- Date range analysis")
            st.write("- Confidence score insights")
            st.write("- Pattern identification")
            st.write("- Actionable recommendations")
        else:
            st.info("No data loaded yet. Upload and process data first in 'Auto-Labelling' section.")
    
    st.markdown("---")
    
    # Chat interface
    st.subheader("Ask Advanced Questions")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Basic (Local Logic - Free):**
        - How many total rows?
        - What's the accuracy?
        - Show sentiment distribution
        - What columns are available?
        
        **Advanced (LLM Analysis - Requires API):**
        - Which products have the most negative feedback?
        - What date ranges show the worst sentiment?
        - Why are customers unhappy in category X?
        - What actionable steps should I take?
        - Compare sentiment trends across products
        - What patterns do you see in low-confidence predictions?
        """)
    
    user_question = st.text_input(
        "Your question:", 
        placeholder="e.g., Which products have the most negatives? What trends do you see in the data?"
    )
    
    if st.button("Ask", type="primary") and user_question:
        with st.spinner("Analyzing..."):
            # Try local logic first
            local_answer = chat_with_local_logic(user_question, context_data)
            
            if local_answer:
                response = local_answer
                method = "local"
                st.success("✅ Answered using local logic (free)")
            elif groq_api_key:
                response = chat_with_llm(groq_api_key, user_question, context_data, full_df)
                method = "llm"
                st.info("🤖 Used LLM for advanced analysis (~$0.001)")
            else:
                response = "I can't answer that with local logic alone. Please provide a Groq API key in the sidebar to enable advanced analysis with product breakdowns, date trends, and actionable insights."
                method = "none"
                st.warning("⚠️ API key needed for advanced analysis")
            
            # Add to history
            st.session_state.chat_history.append({
                'question': user_question,
                'answer': response,
                'method': method,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            method_emoji = "💻" if chat['method'] == 'local' else "🤖" if chat['method'] == 'llm' else "❓"
            method_label = "Local" if chat['method'] == 'local' else "LLM" if chat['method'] == 'llm' else "No API"
            
            with st.container():
                st.markdown(f"**{method_emoji} [{method_label}] [{chat['timestamp']}]**")
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.markdown("---")
        
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>AI Sentiment Automation Engine</strong> | Local ML + Selective LLM | Zero-cost daily predictions</p>
</div>
""", unsafe_allow_html=True)
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import json

# Import ML components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Import Groq for chatbot
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Automation",
    page_icon="🤖",
    layout="wide"
)

# Constants
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'labelled_data' not in st.session_state:
    st.session_state.labelled_data = None
if 'model_stats' not in st.session_state:
    st.session_state.model_stats = None


def load_model():
    """Load trained model and vectorizer from disk"""
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return model, vectorizer, metadata
    except FileNotFoundError:
        return None, None, None


def save_model(model, vectorizer, metadata):
    """Save trained model and vectorizer to disk"""
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)


def train_sentiment_model(texts, labels):
    """
    Train TF-IDF + Logistic Regression sentiment classifier
    
    Args:
        texts: List of text strings
        labels: List of sentiment labels (positive, neutral, negative)
    
    Returns:
        model, vectorizer, metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Transform text to features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Detailed metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['positive', 'neutral', 'negative'])
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    return model, vectorizer, metrics


def predict_sentiment(model, vectorizer, texts):
    """
    Predict sentiment for texts using trained model
    
    Args:
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        texts: List of text strings
    
    Returns:
        predictions, probabilities
    """
    X_tfidf = vectorizer.transform(texts)
    predictions = model.predict(X_tfidf)
    probabilities = model.predict_proba(X_tfidf)
    
    return predictions, probabilities


def get_local_stats(df, text_column):
    """Compute comprehensive statistics for chatbot context"""
    stats = {
        'total_rows': len(df),
        'text_column': text_column,
        'avg_text_length': df[text_column].str.len().mean(),
        'min_text_length': df[text_column].str.len().min(),
        'max_text_length': df[text_column].str.len().max(),
        'all_columns': df.columns.tolist()
    }
    
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        stats['sentiment_distribution'] = sentiment_counts
        stats['most_common_sentiment'] = df['sentiment'].mode()[0]
        
        # Breakdown by category if available
        category_cols = [col for col in df.columns if 'category' in col.lower() or 'product' in col.lower()]
        if category_cols:
            for cat_col in category_cols:
                breakdown = df.groupby([cat_col, 'sentiment']).size().to_dict()
                stats[f'sentiment_by_{cat_col}'] = breakdown
        
        # Date-based analysis if date columns exist
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            for date_col in date_cols:
                try:
                    df_copy = df.copy()
                    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                    df_copy = df_copy.dropna(subset=[date_col])
                    if len(df_copy) > 0:
                        date_sentiment = df_copy.groupby(pd.Grouper(key=date_col, freq='D'))['sentiment'].value_counts().to_dict()
                        stats[f'sentiment_by_{date_col}'] = str(date_sentiment)
                        stats[f'{date_col}_range'] = f"{df_copy[date_col].min()} to {df_copy[date_col].max()}"
                except:
                    pass
    
    if 'confidence' in df.columns:
        stats['avg_confidence'] = df['confidence'].mean()
        stats['min_confidence'] = df['confidence'].min()
        stats['max_confidence'] = df['confidence'].max()
        
        # Low confidence items
        low_conf_count = len(df[df['confidence'] < 0.7])
        stats['low_confidence_count'] = low_conf_count
    
    return stats


def chat_with_local_logic(user_question, context_data):
    """
    Attempt to answer using local logic first
    
    Args:
        user_question: User's question
        context_data: Dictionary with stats and data
    
    Returns:
        answer (str) or None if LLM needed
    """
    question_lower = user_question.lower()
    
    # Handle basic questions locally
    if 'how many' in question_lower or 'total' in question_lower:
        if context_data and 'total_rows' in context_data:
            return f"There are {context_data['total_rows']} total rows in the dataset."
    
    if 'accuracy' in question_lower:
        if context_data and 'accuracy' in context_data:
            return f"The model accuracy is {context_data['accuracy']:.2%}"
    
    if 'distribution' in question_lower or 'breakdown' in question_lower:
        if context_data and 'sentiment_distribution' in context_data:
            dist = context_data['sentiment_distribution']
            return f"Sentiment distribution: {dist}"
    
    if 'average confidence' in question_lower or 'avg confidence' in question_lower:
        if context_data and 'avg_confidence' in context_data:
            return f"Average confidence score: {context_data['avg_confidence']:.2%}"
    
    if 'low confidence' in question_lower:
        if context_data and 'low_confidence_count' in context_data:
            return f"Found {context_data['low_confidence_count']} predictions with confidence below 70%"
    
    if 'columns' in question_lower or 'what data' in question_lower:
        if context_data and 'all_columns' in context_data:
            return f"Available columns: {', '.join(context_data['all_columns'])}"
    
    # For anything requiring analysis, return None to trigger LLM
    return None


def chat_with_llm(api_key, user_question, context_data, full_dataframe=None):
    """Use Groq LLM for complex questions with full data context"""
    if not GROQ_AVAILABLE or not api_key:
        return "LLM not available. Please provide a Groq API key."
    
    try:
        client = Groq(api_key=api_key)
        
        # Build comprehensive context
        context_str = "Dataset Analysis Context:\n"
        context_str += "="*50 + "\n"
        
        if context_data:
            # Basic stats
            if 'total_rows' in context_data:
                context_str += f"Total rows: {context_data['total_rows']}\n"
            
            if 'all_columns' in context_data:
                context_str += f"Columns: {', '.join(context_data['all_columns'])}\n"
            
            # Sentiment distribution
            if 'sentiment_distribution' in context_data:
                context_str += f"\nSentiment Distribution:\n"
                for sentiment, count in context_data['sentiment_distribution'].items():
                    pct = (count / context_data['total_rows'] * 100) if context_data['total_rows'] > 0 else 0
                    context_str += f"  {sentiment.title()}: {count} ({pct:.1f}%)\n"
            
            # Category breakdowns
            for key, value in context_data.items():
                if 'sentiment_by_' in key and 'date' not in key:
                    context_str += f"\n{key.replace('_', ' ').title()}:\n"
                    context_str += f"  {value}\n"
            
            # Date ranges
            for key, value in context_data.items():
                if '_range' in key:
                    context_str += f"{key.replace('_', ' ').title()}: {value}\n"
            
            # Confidence stats
            if 'avg_confidence' in context_data:
                context_str += f"\nConfidence Metrics:\n"
                context_str += f"  Average: {context_data['avg_confidence']:.2%}\n"
                if 'low_confidence_count' in context_data:
                    context_str += f"  Low confidence (<70%): {context_data['low_confidence_count']}\n"
        
        # Add sample data if available
        if full_dataframe is not None and len(full_dataframe) > 0:
            context_str += f"\nSample Data (first 5 rows):\n"
            context_str += full_dataframe.head(5).to_string(index=False)
            
            # Add negative examples if asking about negatives
            if 'negative' in user_question.lower():
                negatives = full_dataframe[full_dataframe['sentiment'] == 'negative'].head(3)
                if len(negatives) > 0:
                    context_str += f"\n\nSample Negative Sentiments:\n"
                    context_str += negatives.to_string(index=False)
        
        context_str += "\n" + "="*50 + "\n"
        
        prompt = f"""{context_str}

User Question: {user_question}

Provide a detailed, actionable analysis based on the data above. Include:
- Key insights from the data
- Patterns or trends identified
- Specific recommendations
- Focus on business value and actionable next steps

Keep your response concise but comprehensive."""
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Use more powerful model for analysis
            messages=[
                {"role": "system", "content": "You are an expert data analyst specializing in sentiment analysis and business insights. Provide actionable, data-driven recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"


# ============================================================================
# MAIN UI
# ============================================================================

st.title("🤖 AI Sentiment Automation Engine")
st.markdown("**Self-improving sentiment analysis with local ML + selective LLM usage**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check for existing model
    model, vectorizer, metadata = load_model()
    
    if model is not None:
        st.success("✅ Model loaded")
        st.metric("Trained on", metadata.get('train_size', 'N/A'))
        st.metric("Accuracy", f"{metadata.get('accuracy', 0):.2%}")
        st.metric("Last trained", metadata.get('trained_at', 'Unknown'))
    else:
        st.warning("⚠️ No model found. Train one first!")
    
    st.markdown("---")
    
    # Groq API key for chatbot
    st.subheader("Chatbot (Optional)")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Only needed for chatbot")
    
    if groq_api_key:
        st.info("💬 Chatbot enabled")
    else:
        st.info("ℹ️ Chatbot will use local logic only")


# Main tabs
tab1, tab2, tab3 = st.tabs([
    "📚 A. Model Training",
    "🤖 B. Auto-Labelling",
    "💬 C. Chatbot"
])

# ============================================================================
# TAB 1: MODEL TRAINING
# ============================================================================

with tab1:
    st.header("📚 Model Training Section")
    st.markdown("Train or retrain your local sentiment classifier using labelled data.")
    
    st.subheader("1️⃣ Upload Labelled Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file with labelled data",
        type=['csv', 'xlsx', 'xls'],
        key='training_upload'
    )
    
    if uploaded_file:
        # Load file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("2️⃣ Select Columns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox(
                    "Text column:",
                    options=df.columns.tolist(),
                    key='train_text_col'
                )
            
            with col2:
                label_column = st.selectbox(
                    "Sentiment label column:",
                    options=df.columns.tolist(),
                    key='train_label_col'
                )
            
            # Validate labels
            if text_column and label_column:
                unique_labels = df[label_column].unique()
                st.write(f"**Found labels:** {', '.join(map(str, unique_labels))}")
                
                # Check if labels are valid
                valid_labels = {'positive', 'neutral', 'negative'}
                current_labels = set(df[label_column].str.lower().unique())
                
                if not current_labels.issubset(valid_labels):
                    st.warning(f"⚠️ Labels should be: positive, neutral, negative. Found: {current_labels}")
                    st.info("Normalizing labels to lowercase...")
                
                # Normalize labels
                df[label_column] = df[label_column].str.lower()
                
                # Filter to valid labels only
                df = df[df[label_column].isin(valid_labels)]
                
                st.write(f"**Valid rows after filtering:** {len(df)}")
                
                st.subheader("3️⃣ Train Model")
                
                if st.button("🚀 Train Classifier", type="primary"):
                    if len(df) < 10:
                        st.error("❌ Need at least 10 labelled samples to train")
                    else:
                        with st.spinner("Training model..."):
                            # Prepare data
                            texts = df[text_column].astype(str).tolist()
                            labels = df[label_column].tolist()
                            
                            # Train
                            model, vectorizer, metrics = train_sentiment_model(texts, labels)
                            
                            # Save model
                            metadata = {
                                'accuracy': metrics['accuracy'],
                                'precision': metrics['precision'],
                                'recall': metrics['recall'],
                                'f1_score': metrics['f1_score'],
                                'train_size': metrics['train_size'],
                                'test_size': metrics['test_size'],
                                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            save_model(model, vectorizer, metadata)
                            st.session_state.model_stats = metrics
                            
                            st.success("✅ Model trained and saved successfully!")
                            st.rerun()
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Display model performance if available
    if st.session_state.model_stats:
        st.subheader("📊 Model Performance")
        
        metrics = st.session_state.model_stats
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
        
        st.markdown("#### Classification Report")
        
        # Format classification report
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df.style.format("{:.2%}"), use_container_width=True)
        
        st.markdown("#### Confusion Matrix")
        conf_matrix = np.array(metrics['confusion_matrix'])
        conf_df = pd.DataFrame(
            conf_matrix,
            index=['True Positive', 'True Neutral', 'True Negative'],
            columns=['Pred Positive', 'Pred Neutral', 'Pred Negative']
        )
        st.dataframe(conf_df, use_container_width=True)


# ============================================================================
# TAB 2: AUTO-LABELLING
# ============================================================================

with tab2:
    st.header("🤖 Auto-Labelling Section")
    st.markdown("Automatically label unlabelled data using the trained local model (no API calls).")
    
    # Check if model exists
    model, vectorizer, metadata = load_model()
    
    if model is None:
        st.error("❌ No trained model found. Please train a model first in the 'Model Training' section.")
    else:
        st.info(f"✅ Using model trained on {metadata.get('train_size', 'N/A')} samples with {metadata.get('accuracy', 0):.2%} accuracy")
        
        st.subheader("1️⃣ Upload Unlabelled Data")
        
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file (with or without labels)",
            type=['csv', 'xlsx', 'xls'],
            key='labelling_upload'
        )
        
        if uploaded_file:
            try:
                # Load file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded {len(df)} rows")
                st.dataframe(df.head(), use_container_width=True)
                
                st.subheader("2️⃣ Select Text Column")
                
                text_column = st.selectbox(
                    "Text column to analyze:",
                    options=df.columns.tolist(),
                    key='label_text_col'
                )
                
                # Check if ground truth exists
                has_ground_truth = 'sentiment' in df.columns or any('label' in col.lower() for col in df.columns)
                
                if has_ground_truth:
                    st.info("ℹ️ Ground truth labels detected. Evaluation metrics will be computed.")
                    ground_truth_col = st.selectbox(
                        "Ground truth label column (optional):",
                        options=['None'] + df.columns.tolist(),
                        key='ground_truth_col'
                    )
                else:
                    ground_truth_col = 'None'
                
                st.subheader("3️⃣ Auto-Label")
                
                if st.button("🚀 Predict Sentiments", type="primary"):
                    with st.spinner("Predicting sentiments..."):
                        # Prepare texts
                        texts = df[text_column].astype(str).tolist()
                        
                        # Predict
                        predictions, probabilities = predict_sentiment(model, vectorizer, texts)
                        
                        # Add to dataframe
                        df['sentiment'] = predictions
                        
                        # Add confidence (max probability)
                        df['confidence'] = probabilities.max(axis=1)
                        
                        # Add individual class probabilities
                        classes = model.classes_
                        for i, cls in enumerate(classes):
                            df[f'prob_{cls}'] = probabilities[:, i]
                        
                        st.session_state.labelled_data = df
                        
                        st.success(f"✅ Labelled {len(df)} rows successfully!")
                        
                        # Show results
                        st.subheader("📊 Results Preview (All Columns)")
                        st.info("💡 All original columns are preserved. Scroll right to see sentiment predictions.")
                        
                        # Reorder columns: original columns first, then sentiment columns at the end
                        original_cols = [col for col in df.columns if col not in ['sentiment', 'confidence', 'prob_positive', 'prob_neutral', 'prob_negative']]
                        sentiment_cols = ['sentiment', 'confidence'] + [col for col in df.columns if col.startswith('prob_')]
                        ordered_cols = original_cols + sentiment_cols
                        
                        st.dataframe(df[ordered_cols].head(20), use_container_width=True)
                        
                        # Sentiment distribution
                        st.markdown("#### Sentiment Distribution")
                        sentiment_counts = df['sentiment'].value_counts()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Positive", sentiment_counts.get('positive', 0))
                        with col2:
                            st.metric("Neutral", sentiment_counts.get('neutral', 0))
                        with col3:
                            st.metric("Negative", sentiment_counts.get('negative', 0))
                        
                        # Average confidence
                        st.metric("Average Confidence", f"{df['confidence'].mean():.2%}")
                        
                        # Show breakdown by category/product if exists
                        category_cols = [col for col in df.columns if 'category' in col.lower() or 'product' in col.lower()]
                        if category_cols:
                            st.markdown("#### Sentiment by Category")
                            for cat_col in category_cols[:2]:  # Show first 2 category columns
                                st.write(f"**{cat_col}:**")
                                breakdown = pd.crosstab(df[cat_col], df['sentiment'], margins=True)
                                st.dataframe(breakdown, use_container_width=True)
                        
                        # Evaluation metrics if ground truth exists
                        if ground_truth_col != 'None' and ground_truth_col in df.columns:
                            st.subheader("📈 Evaluation Metrics")
                            
                            # Normalize ground truth
                            df[ground_truth_col] = df[ground_truth_col].str.lower()
                            
                            # Compute metrics
                            accuracy = accuracy_score(df[ground_truth_col], df['sentiment'])
                            precision, recall, f1, _ = precision_recall_fscore_support(
                                df[ground_truth_col], df['sentiment'], 
                                average='weighted', zero_division=0
                            )
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Accuracy", f"{accuracy:.2%}")
                            with col2:
                                st.metric("Precision", f"{precision:.2%}")
                            with col3:
                                st.metric("Recall", f"{recall:.2%}")
                            with col4:
                                st.metric("F1-Score", f"{f1:.2%}")
                            
                            # Confusion matrix
                            st.markdown("#### Confusion Matrix")
                            conf_matrix = confusion_matrix(
                                df[ground_truth_col], df['sentiment'],
                                labels=['positive', 'neutral', 'negative']
                            )
                            conf_df = pd.DataFrame(
                                conf_matrix,
                                index=['True Positive', 'True Neutral', 'True Negative'],
                                columns=['Pred Positive', 'Pred Neutral', 'Pred Negative']
                            )
                            st.dataframe(conf_df, use_container_width=True)
                
                # Download button
                if st.session_state.labelled_data is not None:
                    st.subheader("4️⃣ Download Results")
                    
                    # Prepare download
                    csv = st.session_state.labelled_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Labelled Data (CSV)",
                        data=csv,
                        file_name=f"labelled_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")


# ============================================================================
# TAB 3: CHATBOT
# ============================================================================

with tab3:
    st.header("💬 Advanced Chatbot Section")
    st.markdown("Ask questions about your sentiment analysis results. Performs deep analysis on entire dataset including products, dates, and trends.")
    
    # Gather context data
    context_data = {}
    full_df = None
    
    if metadata:
        context_data.update(metadata)
    
    if st.session_state.labelled_data is not None:
        full_df = st.session_state.labelled_data
        # Determine text column
        text_col = 'text' if 'text' in full_df.columns else full_df.columns[0]
        local_stats = get_local_stats(full_df, text_col)
        context_data.update(local_stats)
    
    if st.session_state.model_stats:
        context_data.update(st.session_state.model_stats)
    
    # Display available context
    with st.expander("📊 Available Data for Analysis"):
        if context_data:
            st.write("**Dataset Overview:**")
            if 'total_rows' in context_data:
                st.write(f"- Total rows: {context_data['total_rows']}")
            if 'all_columns' in context_data:
                st.write(f"- Columns: {', '.join(context_data['all_columns'])}")
            if 'sentiment_distribution' in context_data:
                st.write(f"- Sentiments: {context_data['sentiment_distribution']}")
            
            st.write("\n**Advanced Analysis Available:**")
            st.write("- Product/Category breakdowns")
            st.write("- Date range analysis")
            st.write("- Confidence score insights")
            st.write("- Pattern identification")
            st.write("- Actionable recommendations")
        else:
            st.info("No data loaded yet. Upload and process data first in 'Auto-Labelling' section.")
    
    st.markdown("---")
    
    # Chat interface
    st.subheader("Ask Advanced Questions")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Basic (Local Logic - Free):**
        - How many total rows?
        - What's the accuracy?
        - Show sentiment distribution
        - What columns are available?
        
        **Advanced (LLM Analysis - Requires API):**
        - Which products have the most negative feedback?
        - What date ranges show the worst sentiment?
        - Why are customers unhappy in category X?
        - What actionable steps should I take?
        - Compare sentiment trends across products
        - What patterns do you see in low-confidence predictions?
        """)
    
    user_question = st.text_input(
        "Your question:", 
        placeholder="e.g., Which products have the most negatives? What trends do you see in the data?"
    )
    
    if st.button("Ask", type="primary") and user_question:
        with st.spinner("Analyzing..."):
            # Try local logic first
            local_answer = chat_with_local_logic(user_question, context_data)
            
            if local_answer:
                response = local_answer
                method = "local"
                st.success("✅ Answered using local logic (free)")
            elif groq_api_key:
                response = chat_with_llm(groq_api_key, user_question, context_data, full_df)
                method = "llm"
                st.info("🤖 Used LLM for advanced analysis (~$0.001)")
            else:
                response = "I can't answer that with local logic alone. Please provide a Groq API key in the sidebar to enable advanced analysis with product breakdowns, date trends, and actionable insights."
                method = "none"
                st.warning("⚠️ API key needed for advanced analysis")
            
            # Add to history
            st.session_state.chat_history.append({
                'question': user_question,
                'answer': response,
                'method': method,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            method_emoji = "💻" if chat['method'] == 'local' else "🤖" if chat['method'] == 'llm' else "❓"
            method_label = "Local" if chat['method'] == 'local' else "LLM" if chat['method'] == 'llm' else "No API"
            
            with st.container():
                st.markdown(f"**{method_emoji} [{method_label}] [{chat['timestamp']}]**")
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.markdown("---")
        
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>AI Sentiment Automation Engine</strong> | Local ML + Selective LLM | Zero-cost daily predictions</p>
</div>
""", unsafe_allow_html=True)