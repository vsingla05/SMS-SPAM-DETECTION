import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Input Text Area Styling */
    .stTextArea textarea {
        background-color: #262730;
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid #4B4B4B;
    }
    
    /* Custom Button Styling */
    div.stButton > button {
        background: linear-gradient(45deg, #FF4B4B, #FF914D);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    
    /* Header Styling */
    h1 {
        background: -webkit-linear-gradient(#eee, #999);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Metrics Box */
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #464855;
        padding: 10px;
        border-radius: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# NLTK and Helper Functions
@st.cache_resource
def load_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    return PorterStemmer(), set(stopwords.words('english'))

ps, STOPWORDS = load_resources()

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in STOPWORDS]
    words = [ps.stem(w) for w in words]
    return " ".join(words)

# Load Model
try:
    # Use relative paths or ensure files are in the root
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 System Error: Model files not found. Please upload 'vectorizer.pkl' and 'model.pkl'.")
    st.stop()

# Session State Management
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {'Spam': 0, 'Not Spam': 0}
if 'latest_result' not in st.session_state:
    st.session_state.latest_result = None

# Sidebar Control
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2593/2593491.png", width=80)
    st.title("🛡️ SpamGuard AI")
    st.markdown("---")
    
    # Live Stats
    st.subheader("📊 Live Statistics")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Spam Detected", st.session_state.stats['Spam'])
    with col_s2:
        st.metric("Safe Msgs", st.session_state.stats['Not Spam'])
    
    st.markdown("---")
    
    # History Toggle
    show_history = st.checkbox("Show History Logs", value=False)
    
    st.markdown("---")
    st.info("💡 **Tip:** This model uses Naive Bayes & TF-IDF vectorization to analyze text patterns.")

# Main Layout

# Title Section
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("Message Intelligence System")
    st.markdown("Enter a message below to analyze its safety and sentiment.")

st.markdown("<br>", unsafe_allow_html=True)

# Main Workspace
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Input Message")
    input_sms = st.text_area("Paste your SMS/Email content:", height=250, placeholder="e.g., Congratulations! You've won a $1000 gift card...")
    
    analyze_btn = st.button("🔍 Analyze Message")

# Prediction Logic
with col2:
    st.subheader("🎯 Analysis Result")
    
    if analyze_btn:
        if input_sms.strip() == "":
            st.warning("⚠️ Text area is empty. Please type something.")
        else:
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])
            
            # 3. Predict
            result = model.predict(vector_input)[0]
            
            # 4. Probabilities
            confidence = 0
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vector_input)[0]
                confidence = max(proba) * 100
            
            # 5. Sentiment
            blob = TextBlob(input_sms)
            sentiment_score = blob.sentiment.polarity
            if sentiment_score > 0.1: sent_label = "Positive"
            elif sentiment_score < -0.1: sent_label = "Negative"
            else: sent_label = "Neutral"

            # 6. Update Stats
            if result == 1:
                st.session_state.stats['Spam'] += 1
                pred_label = "SPAM"
                result_color = "#FF4B4B"  # Red
                icon = "🚨"
            else:
                st.session_state.stats['Not Spam'] += 1
                pred_label = "NOT SPAM"
                result_color = "#00CC96"  # Green
                icon = "✅"

            # 7. Store Result
            st.session_state.latest_result = {
                "msg": input_sms,
                "pred": pred_label,
                "conf": confidence,
                "sent": sent_label,
                "score": sentiment_score,
                "color": result_color,
                "icon": icon,
                "ts": datetime.now().strftime('%H:%M:%S')
            }
            
            # Append to history
            st.session_state.prediction_history.insert(0, st.session_state.latest_result) # Add to top

#    Result Display
    if st.session_state.latest_result:
        res = st.session_state.latest_result
        
        # Result Card
        st.markdown(f"""
            <div style="background-color: {res['color']}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h2 style="color: white; margin:0;">{res['icon']} {res['pred']}</h2>
                <p style="color: white; margin:0; opacity: 0.8;">Confidence: {res['conf']:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Sentiment & Details
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Sentiment Analysis")
            st.markdown(f"**{res['sent']}** ({res['score']:.2f})")
            st.progress((res['score'] + 1) / 2) # Normalize -1 to 1 range for progress bar
        with c2:
            st.caption("Word Count")
            st.markdown(f"**{len(input_sms.split())} words**")

        # Feedback Mechanism
        with st.expander("Is this incorrect? Provide Feedback"):
            feedback = st.radio("Correct Label should be:", ["Spam", "Not Spam"], horizontal=True, key="fb_radio")
            if st.button("Submit Feedback"):
                # Save to CSV
                csv_file = "feedback_records.csv"
                new_row = pd.DataFrame([[res['ts'], res['msg'], feedback]], columns=['Timestamp', 'Message', 'User_Label'])
                
                if not os.path.exists(csv_file):
                    new_row.to_csv(csv_file, index=False)
                else:
                    new_row.to_csv(csv_file, mode='a', header=False, index=False)
                st.success("Feedback recorded!")

    else:
        st.info("👈 Waiting for input...")
        st.markdown("### How it works")
        st.markdown("""
        1. **Text Cleaning**: Removes noise (special chars, stopwords).
        2. **Stemming**: Reduces words to root form (e.g., 'running' -> 'run').
        3. **Vectorization**: Converts text to numbers using TF-IDF.
        4. **Prediction**: Uses Naive Bayes to classify.
        """)

# History and Analytics
st.markdown("---")

if show_history:
    st.subheader("📜 Recent Analysis History")
    if len(st.session_state.prediction_history) > 0:
        df_hist = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df_hist[['ts', 'pred', 'conf', 'sent', 'msg']], use_container_width=True)
    else:
        st.write("No history available yet.")
else:
    # Show Visuals instead if history is hidden
    if sum(st.session_state.stats.values()) > 0:
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.subheader("Distribution")
            # Create a nice Donut chart with Plotly
            labels = list(st.session_state.stats.keys())
            values = list(st.session_state.stats.values())
            
            fig = px.pie(names=labels, values=values, hole=0.5, 
                         color_discrete_sequence=['#FF4B4B', '#00CC96'])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
        with col_viz2:
            st.subheader("Model Performance Info")
            st.markdown("""
            - **Model Type:** Multinomial Naive Bayes
            - **Vectorizer:** TF-IDF (3000 features)
            - **Accuracy:** ~97% (on test set)
            - **Last Updated:** Feb 2026
            """)