# 💬 PhishNetAI: AI-Powered Phishing Detection & Email Breach Analysis

**PhishNetAI** is a comprehensive, AI-driven web application designed to enhance cybersecurity awareness and real-time analysis. Built using **Streamlit**, this tool integrates **Google Gemini AI**, **LangChain**, and **ML-powered phishing detection** to deliver intelligent threat detection, document summarization, and email breach tracking—all in one seamless platform.

![PhishNetAI Screenshot](https://user-images.githubusercontent.com/example/phishnetai.png)

---

## 🏆 Achievements

🚀 **Presented at ImpactX2.0 – IEEE Edition**  
📅 January 2025 | 📍 St. Joseph’s College of Engineering

🚓 **Showcased at Cyberthon 2025**  
🎖️ Hosted by **Tamil Nadu Police**  
📅 March 2025 | 📍 Kumaraguru College of Technology, Coimbatore

> _PhishNetAI was highly appreciated by cybersecurity experts and law enforcement mentors for its practical application in phishing detection, AI-assisted analysis, and public awareness._

---

## 🚀 Features

### 🧠 AI Chatbot (Gemini-Powered)
- Interact with **Google Gemini 1.5** for contextual chats.
- Summarize uploaded documents or get LLM-powered insights.
- Maintains real-time conversational history.

### 🕵️ URL Phishing Detection
- Uses a **Gradient Boosting Classifier** to detect phishing.
- Returns prediction with confidence score.
- Feature engineering via `FeatureExtraction.py`.
- Results are explained by Gemini in human-readable terms.

### 📄 LangChain Document Q&A
- Upload PDFs, DOCX, or text files.
- Ask context-based questions and receive precise answers.
- Get summaries in multiple languages or tones.

### 📧 Email Breach Check
- Integrates with `xposedornot.com` API.
- Verifies if an email is part of a known breach.
- Offers actionable security tips based on exposure.

### ✨ UI Enhancements
- Fully responsive **dark theme**
- 3D **particle effects** using `particles.js`
- Tabbed navigation and interactive widgets via Streamlit & Plotly

---

## 🛠️ Tech Stack

| Component      | Technology                          |
|----------------|--------------------------------------|
| Frontend       | Streamlit, Plotly                    |
| Backend        | Python, scikit-learn, LangChain      |
| LLM            | Google Gemini (gemini-1.5-flash)     |
| ML Model       | GradientBoostingClassifier           |
| API Integration| XposedOrNot (Data breach check)      |
| Utilities      | dotenv, NumPy, pandas, Pickle        |

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/phishnetai.git
cd phishnetai
2. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
3. Environment Variables
Create a .env file:

env
Copy
Edit
GOOGLE_API_KEY=your-gemini-api-key
4. Model & Dataset Setup
Ensure paths are set in qachat.py:

python
Copy
Edit
Dataset CSV: D:/chatbot/phishing.csv
Model File: D:/chatbot/model.pkl
You can refactor the script to dynamically accept paths or use os.path.

5. Run the Application
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
├── qachat.py              # Main Streamlit app
├── Feature.py             # URL feature extraction
├── langchain_utils.py     # LangChain integration
├── requirements.txt       # Python dependencies
└── README.md              # This documentation
🧪 Sample Use Cases
🛡️ Check if a suspicious URL is phishing.

📩 Verify if your email has been leaked.

📚 Ask questions from research papers or policies.

💬 Interact with Google Gemini for real-time answers.

🚨 Limitations & Notes
Only supports Gemini 1.5 Flash model (as of now).

External services like Gemini & XposedOrNot may enforce rate limits.

File paths for dataset/model are Windows-specific – update for cross-platform.

🤝 Contributing
We welcome contributions!

Fork the repo

Create a feature branch: feature/your-feature

Commit your changes

Open a pull request (PR) with proper documentation

📜 License
Distributed under the MIT License. See LICENSE for details.

🙌 Acknowledgements
IEEE ImpactX2.0 mentors for strategic input

Tamil Nadu Police Cyber Crime Division for field insights

The entire PhishNetAI team for collaborative development

“In the era of AI and cyber warfare, staying informed isn’t optional — it’s essential.”
