# 🩺 MEDIBOT - Intelligent Medical Chat Assistant

**MEDIBOT** is an AI-powered medical chatbot that helps users with health-related queries, symptom explanations, treatment insights, preventive care tips, and more. It also supports image-based medical detection (like skin conditions and X-rays).

---

## Features

- 💬 **Medical Chatbot**
  - Ask any medical-related question.
  - Get structured responses including symptoms, causes, prevention, and treatments.
  - Conversational memory to maintain context in a chat session.

- 📷 **Image-Based Medical Detection**
  - Upload medical images (skin, X-ray, MRI, CT scan, etc.)
  - Detect skin conditions or other medical-related images.
  - Receive detailed explanations about detected conditions.

- ⚙️ **Settings & Feedback**
  - Clear chat history.
  - Feedback options (can be expanded in the UI).
  - Informative about MEDIBOT and contact information.

---

## Tech Stack

- **Backend / AI**
  - Python
  - [LangChain](https://www.langchain.com/) for conversational AI
  - Hugging Face Transformers for LLM and image classification
  - Groq LLM (optional)
  
- **Frontend**
  - Streamlit for interactive web interface

- **Other**
  - `.env` for storing API keys securely (never commit to GitHub)Usage

Ask medical questions in the chat tab.

Upload medical images in the image analysis tab.

Clear chat history or view information in the sidebar.

Important Notes

Do not rely solely on MEDIBOT for medical decisions.

Always consult a qualified healthcare professional.

.env contains sensitive API keys — never commit to GitHub.

License

This project is open-source and available under the MIT License.
  - FAISS for vector storage (local indexing)

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Ayushashverma/MEDIBOT.git
cd MEDIBOT
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux / Mac
pip install -r requirements.txt
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
streamlit run medibot.py
