import os
from typing import List

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation

from huggingface_hub import InferenceClient

# --- HuggingFace API ---
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
client = InferenceClient(MODEL_ID, token=HF_TOKEN)

# --- Zephyr wrapper ---
class ZephyrLLM(LLM):
    """Wrapper for Zephyr-7B chat model"""

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return response.choices[0].message["content"]

    def _generate(self, prompts: List[str], stop: List[str] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "zephyr"

# --- Prompt ---
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know.
Do not provide anything beyond the context.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

def set_custom_prompt(template: str):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# --- Load FAISS DB ---
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# --- Create RetrievalQA chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=ZephyrLLM(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
)

# --- Query loop ---
while True:
    user_query = input("Write Query (or 'exit' to quit): ")
    if user_query.lower() == "exit":
        break
    response = qa_chain.invoke({"query": user_query})
    print("\nRESULT:\n", response["result"])
    print("\nSOURCE DOCUMENTS:\n", response["source_documents"])
    print("-" * 50)
