from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer


# --- 2.5 Prompt + simple RAG function
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Use ONLY the provided context to answer the user's question.
If the answer is not in the context, say you don't know.

Question: {question}

Context:
{context}

Answer:"""
)

def load_pdf(pdf_path, chunk_size=600, chunk_overlap=80):
    """
    Reads a PDF and splits it into text chunks for embedding.
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Max size of each chunk
        chunk_overlap (int): Overlap between chunks
    
    Returns:
        List[Document]: LangChain Document objects
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = splitter.split_documents(docs)
    return split_docs

def vector_store_init():
    EMBEDDING_MODEL = "BAAI/bge-large-en"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device":"cuda"},
        encode_kwargs={"normalize_embeddings":True}
    )
    pdf_path = "./test.pdf"
    split_docs = load_pdf(pdf_path)

    persist_dir = "./chroma_rag_store"
    vectorestore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="kb",
        persist_directory=persist_dir
    )
    vectorestore.persist()

    retriever = vectorestore.as_retriever(search_kwargs={"k": 4})
    return retriever

def chat_model_init():
    CHAT_MODEL = "mistralai/Mistral-7B-v0.1"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,                 # enable 4-bit
        bnb_4bit_use_double_quant=True,    # nested quantization
        bnb_4bit_quant_type="nf4",         # NormalFloat4, best for LLMs
        bnb_4bit_compute_dtype="float16",  # can also try "bfloat16" on Ampere GPUs
    )
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CHAT_MODEL,
        device_map="auto",
        quantization_config=quant_config
    )

    adapter_path = "./lora-adapter/adapter_model.safetensors" 
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception:
        print("No PEFT adapter found, continuing with base model.")

    gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
)
    return HuggingFacePipeline(pipeline=gen_pipe), tokenizer


def rag_answer():
    llm = chat_model_init()
    question = "Explain quantum computing in simple terms."
    retriever = vector_store_init()
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = RAG_PROMPT.format(question=question, context=context)

    out = llm(prompt.to_string())
    return out, docs

