from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
import numpy as np
import pickle
from FeatureExtraction import FeatureExtraction
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from langchain_utils import summarize_with_langchain, qa_with_langchain  # Import LangChain functions
from urllib.parse import urlparse

try:
    from langchain_community.document_loaders import UnstructuredFileLoader
except ImportError:
    raise ImportError("The 'unstructured' package is not found. Please install it with 'pip install unstructured'.")

# Set Streamlit page configuration
st.set_page_config(page_title="PhishNet AI", layout="wide")

# Load environment variables
load_dotenv()

genai.configure(api_key="AIzaSyA7Q2_eC2-RXD9sG1rZjMl2FqE0eMQwkB0")  # Your provided API key

# Function to load Gemini Pro model and get responses
@st.cache_resource
def initialize_genai_model():
    model = genai.GenerativeModel("gemini-pro")
    return model.start_chat(history=[])

chat = initialize_genai_model()

def get_gemini_response(question):
    try:
        response = chat.send_message(question, stream=True)
        if not response:
            raise ValueError("No response received from the model.")
        return ''.join([chunk.text for chunk in response if hasattr(chunk, 'text')])
    except Exception as e:
        # Attempt to rewind and retry once
        try:
            chat.rewind()
            response = chat.send_message(question, stream=True)
            if not response:
                raise ValueError("No response received from the model.")
            return ''.join([chunk.text for chunk in response if hasattr(chunk, 'text')])
        except Exception as retry_e:
            # Handle safety ratings and no valid text
            if hasattr(retry_e, 'safety_ratings'):
                safety_ratings = retry_e.safety_ratings
                return f"The response was flagged for safety concerns: {safety_ratings}"
            return f"An error occurred: {retry_e}"

# Train or Load the Phishing Detection Model
@st.cache_resource
def train_and_save_model():
    dataset_path = 'phishing.csv'  # Update this path if necessary
    try:
        data = pd.read_csv(dataset_path)
    except FileNotFoundError:
        st.error(f"Dataset not found at {dataset_path}. Ensure it exists.")
        raise

    # Separate features and labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train a Gradient Boosting Classifier
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Save the trained model
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    return model

# Load existing model or train a new one if not found
try:
    with open('model.pkl', 'rb') as model_file:
        phishing_model = pickle.load(model_file)
except (FileNotFoundError, EOFError):
    phishing_model = train_and_save_model()

# Function to check if a URL is phishing or safe
def check_phishing(url):
    try:
        # Parse the domain from the URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check if the URL is one of the official URLs
        official_urls = ["google.com", "www.google.com", "linkedin.com", "www.linkedin.com"]
        if domain in official_urls:
            reason = classify_safe_reason(url)
            return {
                "is_safe": True,
                "prob_safe": 100,
                "prob_phishing": 0,
                "reason": reason
            }

        # Extract features from the URL
        feature_extractor = FeatureExtraction(url)
        features = np.array(feature_extractor.getFeaturesList()).reshape(1, -1)

        # Adjust features dynamically to match model input size
        required_features = phishing_model.n_features_in_
        if features.shape[1] < required_features:
            features = np.pad(features, ((0, 0), (0, required_features - features.shape[1])), constant_values=0)
        elif features.shape[1] > required_features:
            features = features[:, :required_features]

        # Model prediction
        prediction = phishing_model.predict(features)[0]
        proba_safe = phishing_model.predict_proba(features)[0, 0]
        proba_phishing = phishing_model.predict_proba(features)[0, 1]

        is_safe = proba_safe > 0.6
        reason = classify_safe_reason(url) if is_safe else get_detailed_phishing_reason(url, feature_extractor)

        return {
            "is_safe": is_safe,
            "prob_safe": round(proba_safe * 100, 2),
            "prob_phishing": round(proba_phishing * 100, 2),
            "reason": reason
        }
    except Exception as e:
        return {"error": str(e)}

def classify_phishing_reason(url):
    try:
        # Reset chat history for each new URL
        global chat
        chat = initialize_genai_model()
        
        question = f"Why is the URL '{url}' considered unsafe?"
        response = get_gemini_response(question)
        if "flagged for safety concerns" in response:
            return "The URL is flagged for safety concerns and could not be processed."
        return response
    except Exception as e:
        return f"An error occurred while classifying the reason: {e}"

def classify_safe_reason(url):
    try:
        # Reset chat history for each new URL
        global chat
        chat = initialize_genai_model()
        
        question = f"Why is the URL '{url}' considered safe?"
        response = get_gemini_response(question)
        if "flagged for safety concerns" in response:
            return "The URL is flagged for safety concerns and could not be processed."
        return response
    except Exception as e:
        return f"An error occurred while classifying the reason: {e}"

def get_detailed_phishing_reason(url, feature_extractor):
    reasons = []
    if feature_extractor.shortUrl() == -1:
        reasons.append("The URL is a shortened URL, which is often used for phishing.")
    if feature_extractor.UsingIp() == -1:
        reasons.append("The URL uses an IP address, which is often used for phishing.")
    if feature_extractor.symbol() == -1:
        reasons.append("The URL contains the '@' symbol, which is often used for phishing.")
    if feature_extractor.redirecting() == -1:
        reasons.append("The URL contains multiple redirects, which is often used for phishing.")
    if feature_extractor.prefixSuffix() == -1:
        reasons.append("The URL contains a hyphen in the domain, which is often used for phishing.")
    if feature_extractor.SubDomains() == -1:
        reasons.append("The URL contains multiple subdomains, which is often used for phishing.")
    if feature_extractor.Hppts() == -1:
        reasons.append("The URL does not use HTTPS, which is often used for phishing.")
    if feature_extractor.DomainRegLen() == -1:
        reasons.append("The domain registration length is short, which is often used for phishing.")
    if feature_extractor.Favicon() == -1:
        reasons.append("The favicon is not from the same domain, which is often used for phishing.")
    if feature_extractor.NonStdPort() == -1:
        reasons.append("The URL uses a non-standard port, which is often used for phishing.")
    if feature_extractor.HTTPSDomainURL() == -1:
        reasons.append("The URL contains 'https' in the domain, which is often used for phishing.")
    if feature_extractor.RequestURL() == -1:
        reasons.append("The URL contains suspicious request URLs, which is often used for phishing.")
    if feature_extractor.AnchorURL() == -1:
        reasons.append("The URL contains suspicious anchor URLs, which is often used for phishing.")
    if feature_extractor.LinksInScriptTags() == -1:
        reasons.append("The URL contains suspicious links in script tags, which is often used for phishing.")
    if feature_extractor.ServerFormHandler() == -1:
        reasons.append("The URL contains suspicious server form handlers, which is often used for phishing.")
    if feature_extractor.InfoEmail() == -1:
        reasons.append("The URL contains suspicious email addresses, which is often used for phishing.")
    if feature_extractor.AbnormalURL() == -1:
        reasons.append("The URL is abnormal, which is often used for phishing.")
    if feature_extractor.WebsiteForwarding() == -1:
        reasons.append("The URL contains multiple forwards, which is often used for phishing.")
    if feature_extractor.StatusBarCust() == -1:
        reasons.append("The URL customizes the status bar, which is often used for phishing.")
    if feature_extractor.DisableRightClick() == -1:
        reasons.append("The URL disables right-click, which is often used for phishing.")
    if feature_extractor.UsingPopupWindow() == -1:
        reasons.append("The URL uses popup windows, which is often used for phishing.")
    if feature_extractor.IframeRedirection() == -1:
        reasons.append("The URL uses iframe redirection, which is often used for phishing.")
    if feature_extractor.AgeofDomain() == -1:
        reasons.append("The domain age is short, which is often used for phishing.")
    if feature_extractor.DNSRecording() == -1:
        reasons.append("The DNS recording is suspicious, which is often used for phishing.")
    if feature_extractor.WebsiteTraffic() == -1:
        reasons.append("The website traffic is low, which is often used for phishing.")
    if feature_extractor.PageRank() == -1:
        reasons.append("The page rank is low, which is often used for phishing.")
    if feature_extractor.GoogleIndex() == -1:
        reasons.append("The URL is not indexed by Google, which is often used for phishing.")
    if feature_extractor.LinksPointingToPage() == -1:
        reasons.append("The URL has few links pointing to it, which is often used for phishing.")
    if feature_extractor.StatsReport() == -1:
        reasons.append("The URL has a poor stats report, which is often used for phishing.")
    
    if not reasons:
        reasons.append("The URL contains suspicious patterns or domains, which are often used for phishing.")
    
    return " ".join(reasons)

# Add custom CSS for enhanced UI
def add_custom_css():
    st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #1e1e2f, #121212);
        color: #fff;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background: #222;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 10px;
    }
    .stAlert {
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

# Streamlit UI setup
st.title("💬 PhishNetUI - Chatbot with Phishing Detector")
st.caption("🚀 A Streamlit chatbot powered by Google Gemini AI")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = get_gemini_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

# Tabs for organization
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "🔍 Phishing Detection", "📄 LangChain Features"])

# Chatbot Tab
with tab1:
    st.subheader("Chatbot Section")
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    col1, col2 = st.columns([8, 2])

    with col1:
        input_text = st.text_input("Ask the chatbot something:", key="chat_input")

    with col2:
        file_uploaded = st.file_uploader("", key="document_uploader")
        
        if file_uploaded:
            st.success("Document uploaded successfully!")

    submit_chat = st.button("Ask")

    if submit_chat and input_text:
        response = get_gemini_response(input_text)
        
        st.session_state['chat_history'].append(("You", input_text))
        st.session_state['chat_history'].append(("Bot", response))

    if file_uploaded:
        try:
            summary = summarize_with_langchain(file_uploaded)
            if summary:
                st.subheader("Document Summary")
                st.write(summary)
                
                question = st.text_input("Ask a question about the document:")
                if st.button("Get Answer"):
                    answer = qa_with_langchain(question, summary)
                    if answer:
                        st.subheader("Answer")
                        st.write(answer)
                    else:
                        st.warning("Please upload a document first.")
                    
        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")

    st.subheader("Chat History")
    
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")

# Phishing Detection Tab
with tab2:
    st.subheader("Phishing Detection Section")
    
    url_input = st.text_input("Enter a URL to check if it is safe:", key="phishing_url_input")
    
    submit_url = st.button("Check URL")

    if submit_url and url_input:
        
        result = check_phishing(url_input)
        
        if "error" in result:
            st.error(f"An error occurred: {result['error']}")
            
        elif result["is_safe"]:
            st.success(f"The URL is safe ({result['prob_safe']}% confidence).")
            st.info(f"Reason: {result['reason']}")
            
        else:
            st.error(f"The URL is phishing ({result['prob_phishing']}% confidence).")
            st.info(f"Reason: {result['reason']}")

# LangChain Features Tab
with tab3:
    
    st.subheader("LangChain Features")
    
    file_upload = st.file_uploader("Upload a document for summarization or QA:")
    
    if file_upload:
        
        summary = summarize_with_langchain(file_upload)
        
        if summary:
            st.subheader("Summary")
            st.write(summary)

            question = st.text_input("Ask a question about the document:")
            
            if st.button("Get Answer"):
                answer = qa_with_langchain(question, summary)
                
                if answer:
                    st.subheader("Answer")
                    st.write(answer)
                else:
                    st.warning("Please upload a document first.")
