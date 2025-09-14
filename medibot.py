import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PIL import Image

# LangChain + Models
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Choose your LLM provider
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ModuleNotFoundError:
    GROQ_AVAILABLE = False

from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline

# ----------------------------
# 🔹 Medical Text Classification
# ----------------------------
def classify_medical_question(query, llm):
    prompt = f"""
    Classify whether the following question is related to medical topics:
    - medicine, health, human body, diseases, conditions, symptoms
    - treatments, drugs, therapy
    - lifestyle, wellness, nutrition, diet, exercise
    - hair, skin, nails, mental health, medical technology

    Only answer with 'Yes' if it is related; otherwise, 'No'.

    Question: {query}
    """
    try:
        response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        response_lower = str(response).lower().strip()
        return "yes" in response_lower
    except Exception:
        return True  # safer default

# ----------------------------
# 🔹 Generate Medical Response
# ----------------------------
def generate_medical_response(query, llm, memory=None):
    template = """
    You are a helpful and knowledgeable Medical AI assistant. 
    Provide clear, accurate, and structured answers. 
    Include symptoms, causes, preventive measures, and treatment if relevant.
    Avoid repeating information from previous responses.

    Question: {query}
    Answer:
    """
    prompt = PromptTemplate(input_variables=["query"], template=template)
    chain = LLMChain(prompt=prompt, llm=llm, memory=memory, verbose=False)
    response = chain.run({"query": query})
    return response

# ----------------------------
# 🔹 Streamlit UI
# ----------------------------
def main(llm):
    st.set_page_config(page_title="🩺 MEDIMIND", page_icon="💊", layout="wide")

    # Header
    st.markdown(
        "<div style='text-align: center;'><h1>🩺 MEDIMIND</h1><p style='font-size: 18px;'>Your Intelligent Medical Chat Assistant</p></div>",
        unsafe_allow_html=True
    )
    st.divider()

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferMemory(memory_key="history")

    with st.sidebar.expander("ℹ️ About MEDIMIND", expanded=True):
        st.info("""
        MEDIMIND is your intelligent medical chatbot assistant.  
        - 📚 Symptom explanations  
        - 💊 Treatment insights  
        - 🧪 Preventive care tips  
        - ⚠️ Safety-first responses  

        👉 Always consult a doctor before making health decisions.
        """)

    with st.sidebar.expander("📩 Contact Us", expanded=False):
        st.success("""
        Reach us at:  
        - Email: support@medimind.ai  
        - Website: [medimind.ai](https://medimind.ai)  
        - Twitter: [@medimind_ai](https://twitter.com)  
        """)

    # 🆕 Extra Sidebar Features (like mature chatbots)
    with st.sidebar.expander("💡 FAQs", expanded=False):
        st.write("""
        **Q1. Can I rely 100% on this chatbot?**  
        ⚠️ No, always consult a doctor for medical advice.  

        **Q2. What kind of images can I upload?**  
        📷 Skin, X-ray, MRI, CT scans, etc.  

        **Q3. Is my data safe?**  
        🔒 Yes, no personal info is stored.  
        """)

    with st.sidebar.expander("📝 Feedback", expanded=False):
        feedback = st.text_area("Share your feedback here:")
        if st.button("Submit Feedback"):
            if feedback.strip():
                st.success("✅ Thank you for your valuable feedback!")
            else:
                st.warning("⚠️ Please enter some feedback before submitting.")

    with st.sidebar.expander("❓ Help & Support", expanded=False):
        st.write("""
        - 📖 User Guide: Learn how to use MEDIMIND.  
        - 🛠️ Troubleshooting: If chatbot is slow, check your internet.  
        - 📬 For urgent issues, contact support@medimind.ai  
        """)

    with st.sidebar.expander("🌙 Theme", expanded=False):
        theme = st.radio("Choose Theme:", ["Light", "Dark", "System Default"], index=0)
        st.caption("🔧 (Streamlit theme settings can be adjusted in app config)")

    # Initialize memory & chat history
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="history")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Tabs
    tab1, tab2 = st.tabs(["💬 Medical Chatbot", "📷 Image Analysis"])

    # ----------------------------
    # 💬 Medical Chatbot Tab
    # ----------------------------
    with tab1:
        st.markdown("### 💬 Chat with MEDIMIND")
        st.markdown("<hr>", unsafe_allow_html=True)

        # Show chat history
        for role, message in st.session_state.chat_history:
            if role == "user":
                with st.chat_message("user", avatar="🧑"):
                    st.write(message)
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(message)

        # Chat input always at bottom
        query = st.chat_input("💬 Ask your medical question...")
        if query:
            st.session_state.chat_history.append(("user", query))
            if classify_medical_question(query, llm):
                with st.spinner("🤖 Thinking..."):
                    response = generate_medical_response(query, llm, st.session_state.memory)
                st.session_state.chat_history.append(("bot", response))
            else:
                st.session_state.chat_history.append(("bot", "⚠️ Please ask a medical-related question."))
            st.rerun()

    # ----------------------------
    # 📷 Image Analysis Tab
    # ----------------------------
    with tab2:
        st.markdown("### 📷 Image-Based Medical Detection")
        uploaded_file = st.file_uploader("Upload a medical image (skin, X-ray, scan, etc.)", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            try:
                with st.spinner("🤖 Checking if image is medical-related..."):
                    medical_labels = ["skin disease", "x-ray", "MRI", "CT scan", "lesion", "tumor", "medical scan", "infection"]
                    classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
                    results = classifier(image, candidate_labels=medical_labels)
                    top_label = results[0]['label']
                    top_score = results[0]['score']

                if top_score < 0.4:
                    st.warning("⚠️ This image does not appear medical. Please upload a valid medical image.")
                else:
                    st.success(f"✅ Detected medical-related image: **{top_label}** (confidence: {top_score:.2f})")

                    # If skin disease, run fine-grained classifier
                    if top_label.lower() == "skin disease":
                        try:
                            skin_model_id = "Eraly-ml/Skin-AI"
                            skin_classifier = pipeline("image-classification", model=skin_model_id)
                        except Exception:
                            skin_model_id = "akhaliq/dermnet"
                            skin_classifier = pipeline("image-classification", model=skin_model_id)

                        skin_results = skin_classifier(image)
                        predicted_condition = skin_results[0]['label']
                        predicted_score = skin_results[0]['score']
                        st.success(f"Detected skin condition: **{predicted_condition}** (confidence: {predicted_score:.2f})")

                        query = (
                            f"I have uploaded an image showing a skin condition. "
                            f"The detected skin condition is **{predicted_condition}**. "
                            f"Please provide a detailed explanation including: "
                            f"1. What it is, 2. Symptoms, 3. Causes, 4. Prevention methods, "
                            f"and 5. Treatment options if applicable."
                        )
                    else:
                        query = (
                            f"I have uploaded a medical image labeled as **{top_label}**. "
                            f"Please provide a detailed explanation including: "
                            f"1. What it is, 2. Symptoms, 3. Causes, 4. Prevention methods, "
                            f"and 5. Treatment options if applicable."
                        )

                    response = generate_medical_response(query, llm, memory=None)
                    st.chat_message("assistant", avatar="🤖").write(response)

            except Exception as e:
                st.error(f"⚠️ Could not analyze image. Error: {e}")


# ----------------------------
# 🔹 Run App
# ----------------------------
if __name__ == "__main__":
    load_dotenv(find_dotenv())

    groq_api_key = os.environ.get("GROQ_API_KEY")
    use_groq = GROQ_AVAILABLE and groq_api_key is not None

    if use_groq:
        llm = ChatGroq(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.2,
            groq_api_key=groq_api_key,
        )
    else:
        HF_TOKEN = os.environ.get("HF_TOKEN")
        if HF_TOKEN is None:
            st.error("Neither GROQ_API_KEY nor HF_TOKEN is set.")
            st.stop()
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            temperature=0.2,
            model_kwargs={"token": HF_TOKEN, "max_length": 2048},
        )

    main(llm)
