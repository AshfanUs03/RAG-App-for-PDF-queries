from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI   #from langchain_openai import ChatOpenAI, from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings  #from langchain_huggingface import HuggingFaceEmbeddings, from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ✅ Load your PDF resume
loader = PyPDFLoader("resume_ashfan.pdf")
documents = loader.load()

# ✅ Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# ✅ Use HuggingFace embeddings (no OpenAI key needed here)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Create FAISS vector store
vectorstore = FAISS.from_documents(docs, embedding_model)

# ✅ Use OpenRouter to call GPT
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key="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",
    openai_api_base="https://openrouter.ai/api/v1"
)

# ✅ Retrieval Q&A Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# ✅ CLI Q&A
print("🔍 Ask anything about your resume (type 'exit' to quit):")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting...")
        break
    response = qa_chain.invoke(query)   #response = qa_chain.run(query)
    print("GPT:", response)
