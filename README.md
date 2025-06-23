# 💬 PhishNetAI: AI-Powered Phishing Detection & Email Breach Analysis

PhishNetAI is a comprehensive, AI-driven web application designed to enhance cybersecurity awareness and analysis. Built using Streamlit, this tool integrates Google Gemini AI, LangChain, and ML-based phishing detection to provide users with real-time threat analysis, document summarization, and email breach tracking.

![PhishNetAI Screenshot](https://user-images.githubusercontent.com/example/phishnetai.png)

 🚀 Features

🧠 AI Chatbot (Gemini-Powered)
- Interact with Google Gemini 1.5 for contextual chat.
- Summarize documents or get answers using LLMs.
- Real-time conversation history.

🕵️ URL Phishing Detection
- Gradient Boosting Classifier-based detection.
- Real-time phishing prediction with confidence score.
- Feature extraction via `FeatureExtraction` class.
- Gemini-generated explanations for results.

📄 LangChain Document Q&A
- Upload files and ask contextual questions.
- Generate summaries in various styles and languages.

📧 Email Breach Check
- Integrates with `XposedOrNot` API.
- Scan emails for known data breaches.
- Generate analytics and receive security best practices.

✨ UI Enhancements
- Dark-themed custom CSS.
- 3D particle background using `particles.js`.
- Tabbed navigation and interactive widgets.



🛠️ Tech Stack

| Component | Technology |
|----------|-------------|
| Frontend | Streamlit, Plotly |
| Backend | Python, scikit-learn, LangChain |
| LLM | Google Gemini (gemini-1.5-flash) |
| ML Model | GradientBoostingClassifier |
| Breach Check | `xposedornot.com` API |
| Utilities | dotenv, NumPy, pandas, Pickle |

🔧 Setup Instructions

1. Clone the Repository

```bash
git clone https://github.com/yourusername/phishnetai.git
cd phishnetai
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Environment Variables
Create a .env file and add your Gemini API key:

env
Copy
Edit
GOOGLE_API_KEY=your-gemini-api-key
4. Model & Dataset Setup
Ensure the following paths are set correctly in qachat.py:

Dataset CSV: D:/chatbot/phishing.csv

Trained model: D:/chatbot/model.pkl

Or modify the script to dynamically accept alternate paths.

5. Run the App
bash
Copy
Edit
streamlit run qachat.py
📁 Project Structure
bash
Copy
Edit
phishnetai/
│
├── qachat.py                   # Main Streamlit application
├── Feature.py                  # Feature extraction logic for URLs
├── langchain_utils.py          # Document summarization & Q&A
├── requirements.txt            # Dependencies
└── README.md                   # This file
🧪 Sample Use Cases
🛡️ Verify if a suspicious URL is safe or phishing.

📩 Check whether your email has been involved in a breach.

📚 Ask questions about research papers, resumes, or legal docs.

💬 Have natural conversations with Gemini AI.

🚨 Limitations & Notes
Only supports Gemini 1.5 flash model for now.

Relies on external services (XposedOrNot, Google AI) — rate limits may apply.

The current path for the dataset and model is hardcoded and platform-specific (Windows). Update it for cross-platform use.

🤝 Contributing
Contributions are welcome! To propose a feature or bugfix:

Fork the repo

Create a new branch (feature/your-feature)

Push changes and open a PR

📜 License
Distributed under the MIT License. See LICENSE for details
