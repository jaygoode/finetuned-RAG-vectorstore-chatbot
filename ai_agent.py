from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# --- 2.5 Prompt + simple RAG function
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a career advisor. Use ONLY the provided context (the candidate's CV) to answer the user's question.
If the answer is not in the context, say "I don't know".

Question: {question}

Candidate CV:
{context}

Answer:"""
)

def load_pdf(pdf_path, chunk_size=300, chunk_overlap=50):
    """
    Reads a PDF and splits it into text chunks for embedding.
    
    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Max size of each chunk
        chunk_overlap (int): Overlap between chunks
    
    Returns:
        List[Document]: LangChain Document objects
    """
    loader = UnstructuredPDFLoader(pdf_path)
    docs = loader.load()
    for d in docs:
        print(d.page_content[:500])
    breakpoint()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
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

    persist_dir = "./chroma_rag_store"
    collection_name = "kb_store"

    try:
        # Try to load existing collection
        vectorestore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        print("Loaded existing Chroma vectorstore from disk ✅")
    except Exception:
        # If it doesn't exist, embed documents
        split_docs = load_pdf("./johnny_nylund_2025_Resume.pdf")
        vectorestore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        print("Created new Chroma vectorstore from PDF ✅")

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

    adapter_path = "D:\\repos\\finetuned-RAG-vectorstore-chatbot\\lora-adapter\\adapter_model.safetensors" 
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
    return HuggingFacePipeline(pipeline=gen_pipe)


def rag_answer():
    llm = chat_model_init()
    question = "what job roles are recommended for this persons CV(located under context)?"
    retriever = vector_store_init()
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = RAG_PROMPT.format(question=question, context=context)

    out = out = llm.invoke(prompt)
    return out, docs

