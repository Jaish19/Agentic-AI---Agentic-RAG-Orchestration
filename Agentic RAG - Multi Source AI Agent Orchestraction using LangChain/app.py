# app.py

import os
import streamlit as st
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from langchain import hub
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

from dotenv import load_dotenv
load_dotenv()

# Streamlit UI
st.set_page_config("LangChain Agent App", layout="wide")
st.title("🔎 LangChain Multi-Tool Agent")
st.markdown("Use Wiki, Arxiv, and LangSmith document search via LLM")

# API Keys
openai_key = st.sidebar.text_input("OpenAI Key", type="password")
hf_key = st.sidebar.text_input("HuggingFace Token", type="password")
gemini_key = st.sidebar.text_input("Gemini Key", type="password")
langsmith_key = st.sidebar.text_input("LangSmith Key", type="password")

# Set environment variables
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if hf_key:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key
    os.environ["HUG_FACE_TOKEN"] = hf_key
if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key
if langsmith_key:
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Load Tools (Wiki, Arxiv, Retriever)
@st.cache_resource
def load_tools():
    # Wiki Tool
    wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))

    # Arxiv Tool
    arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))

    # Retriever Tool (LangSmith docs)
    loader = WebBaseLoader("https://docs.smith.langchain.com/")
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever()
    retriever_tool = create_retriever_tool(retriever, "langsmith_search", "Search LangSmith documentation.")

    return [wiki_tool, arxiv_tool, retriever_tool]

tools = load_tools()

# Choose LLM
llm_option = st.sidebar.radio("Choose LLM", ["Gemini", "HuggingFace"])
if llm_option == "Gemini":
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)
else:
    llm = HuggingFacePipeline.from_model_id(
        model_id="5CD-AI/visocial-T5-base",
        task="text2text-generation",
        device=0,
        pipeline_kwargs={"max_new_tokens": 100},
    )

# Prompt template
prompt = hub.pull("hwchase17/openai-functions-agent")

# Agent Executor
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# Query UI
user_query = st.text_input("Enter your question", placeholder="What is LangSmith?")
if user_query:
    with st.spinner("Thinking..."):
        response = agent_executor.invoke({"input": user_query})
        st.success("Agent Response:")
        st.write(response["output"])
