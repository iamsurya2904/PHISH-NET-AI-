from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
import numpy as np
import pickle
from Feature import FeatureExtraction
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
try:
    from langchain_utils import summarize_text, load_input_file, STYLES, LANGUAGES  # Import LangChain functions
except FileNotFoundError as e:
    raise FileNotFoundError(f"Error importing langchain_utils: {e}")
from urllib.parse import urlparse
import plotly.graph_objs as go
from PIL import Image
import streamlit.components.v1 as components

try:
    from langchain_community.document_loaders import UnstructuredFileLoader
except ImportError:
    raise ImportError("The 'unstructured' package is not found. Please install it with 'pip install unstructured'.")
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'exceptions' module is not found. Please install it or check your environment.")

import requests
import time

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
        valid_text = ''.join([chunk.text for chunk in response if hasattr(chunk, 'text')])
        if not valid_text:
            raise ValueError("No valid text received from the model.")
        return valid_text
    except Exception as e:
        # Attempt to rewind and retry once
        try:
            chat.rewind()
            response = chat.send_message(question, stream=True)
            if not response:
                raise ValueError("No response received from the model.")
            valid_text = ''.join([chunk.text for chunk in response if hasattr(chunk, 'text')])
            if not valid_text:
                raise ValueError("No valid text received from the model.")
            return valid_text
        except Exception as retry_e:
            # Handle safety ratings and no valid text
            if hasattr(retry_e, 'safety_ratings'):
                safety_ratings = retry_e.safety_ratings
                return f"The response was flagged for safety concerns: {safety_ratings}"
            return "An error occurred while processing your request. Please try again later."

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
        # Extract features from the URL
        feature_extractor = FeatureExtraction(url)
        features = np.array(feature_extractor.extract_features()).reshape(1, -1)

        # Adjust features dynamically to match the model input size
        required_features = phishing_model.n_features_in_
        if features.shape[1] < required_features:
            features = np.pad(features, ((0, 0), (0, required_features - features.shape[1])), constant_values=0)
        elif features.shape[1] > required_features:
            features = features[:, :required_features]

        # Model prediction
        prediction = phishing_model.predict(features)[0]
        proba_safe = phishing_model.predict_proba(features)[0, 0]
        proba_phishing = phishing_model.predict_proba(features)[0, 1]

        # Determine if the URL is safe
        is_safe = proba_safe > 0.6  # Confidence threshold for safe classification

        # Generate reasons based on prediction
        if is_safe:
            reason = classify_safe_reason(url)
            message = f"The URL is safe ({round(proba_safe * 100, 2)}% confidence)."
        else:
            reason = classify_phishing_reason(url)
            message = f"The URL is phishing ({round(proba_phishing * 100, 2)}% confidence)."

        # Return result dictionary
        return {
            "is_safe": is_safe,
            "prob_safe": round(proba_safe * 100, 2),
            "prob_phishing": round(proba_phishing * 100, 2),
            "message": message,
            "reason": reason,
        }
    except ValueError as ve:
        return {"error": f"Value error occurred: {ve}"}
    except FileNotFoundError as fnfe:
        return {"error": f"File not found: {fnfe}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}

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

def get_detailed_safe_reason(url, feature_extractor):
    try:
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
    try:
        global chat
        chat = initialize_genai_model()
        
        question = f"Why is the URL '{url}' considered unsafe?"
        response = get_gemini_response(question)
        if "flagged for safety concerns" in response:
            return "The URL is flagged for safety concerns and could not be processed."
        return response
    except Exception as e:
        return f"An error occurred while classifying the reason: {e}"

def check_email_breaches(email):
    api_url = f"https://api.xposedornot.com/v1/check-email/{email}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"Error": str(e)}

def get_breach_analytics(email, retries=3, backoff_factor=1):
    api_url = f"https://api.xposedornot.com/v1/breach-analytics?email={email}"
    for attempt in range(retries):
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if response.status_code == 429 and attempt < retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                time.sleep(sleep_time)
            elif response.status_code == 403:
                return {"Error": "Access forbidden. Please check your API key or permissions."}
            else:
                return {"Error": str(e)}

def summarize_breach_analytics(analytics):
    try:
        summary = get_gemini_response(f"Summarize the following breach analytics: {analytics}")
        return summary
    except Exception as e:
        return f"An error occurred while summarizing the analytics: {e}"

# Add custom CSS for enhanced UI
def add_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
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
        background-color: #2b2b38;
        color: #fff;
    }
    .stAlert {
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

# Add 3D particle background
particles_js = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  #particles-js {
    position: fixed;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: -1; /* Send the animation to the back */
  }
  .content {
    position: relative;
    z-index: 1;
    color: white;
  }
  </style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="content">
    <!-- Placeholder for Streamlit content -->
  </div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {
          "value": 300,
          "density": {
            "enable": true,
            "value_area": 800
          }
        },
        "color": {
          "value": "#ffffff"
        },
        "shape": {
          "type": "circle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          },
          "image": {
            "src": "img/github.svg",
            "width": 100,
            "height": 100
          }
        },
        "opacity": {
          "value": 0.5,
          "random": false,
          "anim": {
            "enable": false,
            "speed": 1,
            "opacity_min": 0.2,
            "sync": false
          }
        },
        "size": {
          "value": 2,
          "random": true,
          "anim": {
            "enable": false,
            "speed": 40,
            "size_min": 0.1,
            "sync": false
          }
        },
        "line_linked": {
          "enable": true,
          "distance": 100,
          "color": "#ffffff",
          "opacity": 0.22,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 0.2,
          "direction": "none",
          "random": false,
          "straight": false,
          "out_mode": "out",
          "bounce": true,
          "attract": {
            "enable": false,
            "rotateX": 600,
            "rotateY": 1200
          }
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": true,
            "mode": "grab"
          },
          "onclick": {
            "enable": true,
            "mode": "repulse"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 100,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 2,
            "duration": 2,
            "opacity": 0.5,
            "speed": 1
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 2
          },
          "remove": {
            "particles_nb": 3
          }
        }
      },
      "retina_detect": true
    });
  </script>
</body>
</html>
"""

# Streamlit UI setup
st.markdown("<h1 style='font-family: VT323, monospace;'>💬 PhishNetUI - Chatbot with Phishing Detector</h1>", unsafe_allow_html=True)
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
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chatbot", "🔍 Phishing Detection", "📄 LangChain Features", "📧 Email Breach Check"])

# Chatbot Tab
with tab1:
    st.subheader("Chatbot Section")
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    col1, col2 = st.columns([8, 2])

    with col1:
        input_text = st.text_input("Ask the chatbot something:", key="chat_input_main")

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
            content = file_uploaded.read().decode('utf-8', errors='ignore')
            summary = summarize_text(content, "Detailed", "English")
            if summary:
                st.subheader("Document Summary")
                st.write(summary)
                
                question = st.text_input("Ask a question about the document:", key="document_question")
                if st.button("Get Answer", key="get_answer_button"):
                    answer = get_gemini_response(question)
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
            st.success(f"{result['message']}")
            st.info(f"Reason: {result['reason']}")
        else:
            st.error(f"{result['message']}")
            st.warning(f"Reason: {result['reason']}")

# LangChain Features Tab
with tab3:
    st.subheader("LangChain Features")
    
    file_upload = st.file_uploader("Upload a document for summarization or QA:")
    
    if file_upload:
        content = load_input_file(file_upload)
        
        if content:
            st.subheader("Summary")
            style = st.selectbox("Select summary style:", options=list(STYLES.keys()))
            language = st.selectbox("Select language:", options=LANGUAGES)
            summary = summarize_text(content, style, language)
            st.write(summary)

            question = st.text_input("Ask a question about the document:")
            
            if st.button("Get Answer"):
                answer = get_gemini_response(question)
                if answer:
                    st.subheader("Answer")
                    st.write(answer)
                else:
                    st.warning("No answer could be generated.")

# Email Breach Check Tab
with tab4:
    st.subheader("Email Breach Check Section")
    
    email_input = st.text_input("Enter an email to check for breaches:", key="email_breach_input")
    
    if st.button("Check Email Breaches"):
        if email_input:
            result = check_email_breaches(email_input)
            if "Error" in result:
                st.error(f"An error occurred: {result['Error']}")
            else:
                st.json(result)
        else:
            st.warning("Please enter an email address.")
    
    if st.button("Get Breach Analytics"):
        if email_input:
            result = get_breach_analytics(email_input)
            if "Error" in result:
                st.error(f"An error occurred: {result['Error']}")
            else:
                st.json(result)
                if st.button("Summarize and Explain Analytics"):
                    summary = summarize_breach_analytics(result)
                    st.subheader("Analytics Summary")
                    st.write(summary)
        else:
            st.warning("Please enter an email address.")

# Add the 3D particle background
components.html(particles_js, height=370, scrolling=False)
