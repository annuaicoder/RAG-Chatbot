# rag_gpt2.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np

# -----------------------------
# 1. Load or define your documents
# -----------------------------
knowledge_chunks = [
    "GPT-2 is a transformer-based language model developed by OpenAI. It is trained to predict the next word in a sentence.",
    "Artificial Intelligence (AI) is the field of study focused on creating machines that can perform tasks that typically require human intelligence.",
    "RAG stands for Retrieval-Augmented Generation. It combines document retrieval with natural language generation.",
    "FAISS is a library developed by Facebook AI that allows for efficient similarity search of dense vectors.",
    "BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query.",
    "Tokenization is the process of breaking down text into smaller units, typically words or subwords.",
    "The context window of GPT-2 is limited to 1024 tokens, which includes both the prompt and the generated response."
]

# -----------------------------
# 2. Build a simple TF-IDF retriever
# -----------------------------
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(knowledge_chunks)

def retrieve_documents(query, top_k=3):
    query_vector = vectorizer.transform([query])
    scores = (doc_vectors @ query_vector.T).toarray().flatten()
    top_indices = scores.argsort()[-top_k:][::-1]
    return [knowledge_chunks[i] for i in top_indices]

# -----------------------------
# 3. Load GPT-2 model and tokenizer
# -----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Optional: speed up with GPU
if torch.cuda.is_available():
    model = model.cuda()

# -----------------------------
# 4. Combine retrieved documents and query into a prompt
# -----------------------------
def build_prompt(context_chunks, query):
    context = "\n".join(context_chunks)
    prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
    return prompt

# -----------------------------
# 5. Generate answer using GPT-2
# -----------------------------
def generate_answer(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + max_length,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.8,
            eos_token_id=tokenizer.eos_token_id
        )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_output.replace(prompt, "").strip()

# -----------------------------
# 6. Full RAG Pipeline
# -----------------------------
def rag_pipeline(query):
    retrieved_docs = retrieve_documents(query)
    prompt = build_prompt(retrieved_docs, query)
    print(f"\n--- Prompt ---\n{prompt}\n")
    answer = generate_answer(prompt)
    return answer

# -----------------------------
# 7. Test the pipeline
# -----------------------------
if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        response = rag_pipeline(user_input)
        print(f"\n🤖 GPT-2 Answer:\n{response}")

