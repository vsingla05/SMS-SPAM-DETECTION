import streamlit as st
import pickle
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import os
from datetime import datetime

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Function to preprocess the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load the vectorizer
vectorizer_path = r'D:\Projects\Fraud Detection\SMS-Spam-Detection\vectorizer.pkl'
if os.path.exists(vectorizer_path):
    with open(vectorizer_path, 'rb') as file:
        tfidf = pickle.load(file)
else:
    st.error("vectorizer.pkl not found. Please check if it was created correctly.")
    st.stop()

# Load the model
model_path = r'D:\Projects\Fraud Detection\SMS-Spam-Detection\model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    st.error("model.pkl not found. Please check if it was created correctly.")
    st.stop()

# Initialize session state for prediction counts and feedback
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = {'Spam': 0, 'Not Spam': 0}
if 'latest_prediction' not in st.session_state:
    st.session_state.latest_prediction = None
if 'predictions_time' not in st.session_state:
    st.session_state.predictions_time = []
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []  # Store feedback records
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = {}  # Dictionary to store predictions per user (using session_id)

# Feedback file path
feedback_file_path = 'feedback_records.csv'

# Function to log feedback to a CSV file
def log_feedback(message, prediction, feedback, timestamp):
    # Check if the file exists, if not create it with headers
    if not os.path.exists(feedback_file_path):
        with open(feedback_file_path, 'w') as file:
            file.write('Timestamp,Message,Prediction,Feedback\n')

    # Append the feedback to the file
    with open(feedback_file_path, 'a') as file:
        file.write(f'{timestamp},{message},{prediction},{feedback}\n')

# Streamlit interface
st.title("SMS Spam Classifier")
st.markdown("### Enter a message below to check if it's spam or not.")

# Ensure each user has a unique session ID
session_id = st.session_state.get('session_id', None)
if session_id is None:
    session_id = str(datetime.now().timestamp())  # Use timestamp to create a unique session ID
    st.session_state.session_id = session_id

input_sms = st.text_area("Enter the message", height=200)

if st.button('Predict', key='predict_button', help='Click to classify the SMS'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        with st.spinner("Processing..."):
            try:
                # 1. Preprocess
                transformed_sms = transform_text(input_sms)

                # 2. Vectorize
                vector_input = tfidf.transform([transformed_sms])

                # 3. Predict
                result = model.predict(vector_input)[0]

                # Calculate prediction confidence
                prob_spam = model.predict_proba(vector_input)[0][1]
                prob_not_spam = model.predict_proba(vector_input)[0][0]
                
                # 4. Sentiment Analysis
                sentiment = TextBlob(input_sms).sentiment.polarity
                if sentiment > 0:
                    sentiment_label = "Positive"
                elif sentiment < 0:
                    sentiment_label = "Negative"
                else:
                    sentiment_label = "Neutral"

                # Update session state for dynamic graph
                if result == 1:
                    st.session_state.prediction_data['Spam'] += 1
                    st.header(f"🛑 Spam (Confidence: {prob_spam*100:.2f}%)")
                else:
                    st.session_state.prediction_data['Not Spam'] += 1
                    st.header(f"✅ Not Spam (Confidence: {prob_not_spam*100:.2f}%)")

                # Store only the latest prediction
                st.session_state.latest_prediction = (input_sms, "Spam" if result == 1 else "Not Spam")

                # Track time of prediction for trend visualization
                st.session_state.predictions_time.append(datetime.now())

                # Store the prediction in the session state for the specific user
                if session_id not in st.session_state.prediction_history:
                    st.session_state.prediction_history[session_id] = []

                st.session_state.prediction_history[session_id].append({
                    'message': input_sms,
                    'prediction': "Spam" if result == 1 else "Not Spam",
                    'sentiment': sentiment_label,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

                # Display sentiment analysis
                st.write(f"Sentiment: {sentiment_label} (Score: {sentiment:.2f})")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Display the most recent prediction
if st.session_state.latest_prediction:
    sms, pred = st.session_state.latest_prediction
    st.subheader("Latest Prediction")
    st.write(f"Message: {sms} - Prediction: {pred}")

# Display the history of predictions for this user only
st.markdown("### Previous Predictions")

if session_id in st.session_state.prediction_history and len(st.session_state.prediction_history[session_id]) > 0:
    prediction_df = pd.DataFrame(st.session_state.prediction_history[session_id])
    st.dataframe(prediction_df)
else:
    st.write("No previous predictions yet.")

# Feedback section
feedback = st.radio("Was the prediction correct?", ("Yes", "No"))
if st.button("Submit Feedback"):
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Log the feedback to the CSV file
    log_feedback(sms, pred, feedback, timestamp)

    # Store feedback data in session state
    st.session_state.feedback_data.append({'message': sms, 'prediction': pred, 'feedback': feedback, 'timestamp': timestamp})

    st.success("Thank you for your feedback!")

# Visualizations
st.markdown("### Distribution of Spam vs. Not Spam Messages")

# Create a bar plot with custom colors
labels = list(st.session_state.prediction_data.keys())
counts = list(st.session_state.prediction_data.values())

plt.figure(figsize=(8, 4))
sns.barplot(x=labels, y=counts, palette=['red', 'green'])
plt.title('Dynamic Distribution of Spam and Not Spam Messages')
plt.xlabel('Message Type')
plt.ylabel('Count')

st.pyplot(plt)

# Trend Visualization (Time-based)
if len(st.session_state.predictions_time) > 0:
    # Create lists for each prediction's timestamp
    time_labels = [t.strftime('%Y-%m-%d %H:%M') for t in st.session_state.predictions_time]
    spam_counts = [st.session_state.prediction_data['Spam']] * len(time_labels)  # Track spam count over time
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_labels, spam_counts, label="Spam count over time", color='red', marker='o')
    plt.xticks(rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Spam Messages Count')
    plt.title('Trend of Spam Messages Over Time')
    plt.tight_layout()  # To avoid overlap of x-axis labels
    st.pyplot(plt)

st.markdown(""" 
---
*This SMS Spam Classifier uses machine learning techniques to determine whether a message is spam or not. Enter a message and get instant feedback!*
""")
