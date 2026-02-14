📩 SMS Spam Detection using Machine Learning

A Machine Learning web application that classifies SMS messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and is deployed with Streamlit.

🚀 Live Demo

🔗 Add your deployed Streamlit link here

**Problem Statement**

Spam SMS messages are widely used for fraud, promotions, and phishing attacks.
This project builds a machine learning model that automatically detects whether a message is:

✅ Ham – Legitimate message

🚨 Spam – Unwanted or fraudulent message

The application provides an interactive web interface for real-time predictions.

🧠 Model Overview

Text Preprocessing (Lowercasing, Stopword Removal, etc.)

Feature Extraction using TF-IDF Vectorization

Classification using (MultinomialNB)

Evaluation using Accuracy, Precision, Recall, and F1-Score

🛠️ Tech Stack

Python

Pandas

NumPy

Scikit-learn

Streamlit

📂 Project Structure
sms-spam-detection/
│
├── app.py                # Streamlit application
├── model.pkl   sv          # Trained ML model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Project dependencies
├── README.md             # Documentation
└── .gitignore

⚙️ How It Works

User enters an SMS message in the web app.

Text is cleaned and preprocessed.

Message is converted into numerical features using TF-IDF.

The trained ML model predicts Spam or Ham.

Result is displayed instantly.

📊 Example Predictions
SMS Message	Prediction
Congratulations! You have won a free iPhone.	Spam
Hey, are we meeting tomorrow?	Ham
Win cash prizes now!!!	Spam
Call me when you are free.	Ham
💻 Installation & Running Locally
1️⃣ Clone the Repository
git clone https://github.com/vsingla05/sms-spam-detection.git
cd sms-spam-detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app.py


🔐 Security Note

Sensitive information such as API keys and credentials are not included in this repository. Environment variables are used where necessary.

🚀 Future Improvements

Deploy using Docker

Add Deep Learning models (LSTM / BERT)

Add confidence probability score

Model Train using feedbacks

Improve handling of obfuscated spam messages

👨‍💻 Author

Vansh Singla
B.Tech Computer Science (AI)
Aspiring Machine Learning Engineer