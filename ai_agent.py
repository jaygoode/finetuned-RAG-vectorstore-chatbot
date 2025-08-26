from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

# --- Prompt
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a career advisor. Use ONLY the provided context (the candidate's CV) to answer the user's question.
If the answer is not in the context, say "I don't know".

Question: {question}

Candidate CV:
{context}

Answer:"""
)

def load_text_file(file_path, chunk_size=300, chunk_overlap=50):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Clean text
    text = " ".join(text.split())
    print(f"[DEBUG] Loaded text length: {len(text)} characters")
    
    doc = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents([doc])
    print(f"[DEBUG] Split into {len(split_docs)} chunks")
    if split_docs:
        print(f"[DEBUG] First chunk preview:\n{split_docs[0].page_content[:300]}")
    return split_docs

# --- Vectorstore
def vector_store_init():
    EMBEDDING_MODEL = "BAAI/bge-large-en"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device":"cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    persist_dir = "./chroma_rag_store"
    collection_name = "kb_store"
    
    try:
        vectorestore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        print("[DEBUG] Loaded existing Chroma vectorstore from disk ✅")
    except Exception:
        split_docs = load_text_file("D:\\repos\\finetuned-RAG-vectorstore-chatbot\\johnny_nylund_2025_Resume.txt")
        vectorestore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        print("[DEBUG] Created new Chroma vectorstore from PDF ✅")
    
    retriever = vectorestore.as_retriever(search_kwargs={"k": 4})
    print("[DEBUG] Retriever initialized")
    return retriever

# --- Chat model
def chat_model_init():
    CHAT_MODEL = "mistralai/Mistral-7B-v0.1"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
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
        print("[DEBUG] Loaded PEFT adapter")
    except Exception:
        print("[DEBUG] No PEFT adapter found, continuing with base model")
    
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
    )
    
    print("[DEBUG] Generation pipeline initialized")
    return HuggingFacePipeline(pipeline=gen_pipe)

# --- RAG function
def rag_answer():
    llm = chat_model_init()
    question = "what job roles are recommended for this persons CV(located under context)?"
    
    retriever = vector_store_init()
    
    # Retrieve docs
    docs = retriever.invoke(question)
    print(f"[DEBUG] Retrieved {len(docs)} documents")
    
    # Preview retrieved documents
    # for i, d in enumerate(docs):
    #     print(f"[DEBUG] Retrieved doc {i} first 300 chars:\n{d.page_content[:300]}\n---")
    
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = RAG_PROMPT.format(question=question, context=context)
    # print(f"[DEBUG] Prompt:\n{prompt[:500]}...\n---")
    
    out = llm.invoke(prompt)
    # print(f"[DEBUG] LLM output:\n{out}\n---")
    return out, docs