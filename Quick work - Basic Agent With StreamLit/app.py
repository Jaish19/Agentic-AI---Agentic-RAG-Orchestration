import streamlit as st
import os
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from google.colab import userdata

# Load GROQ key securely
os.environ["GROQ_API_KEY"] = "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"

# UI Setup
st.set_page_config(page_title="Groq QA Agent", layout="wide")
st.title("🤖 World Cup Q&A with Groq + DuckDuckGo")

question = st.text_input("Enter your question:", "Who lifted the WC2025?")

if st.button("Ask"):
    with st.spinner("Fetching answer..."):
        agent = Agent(
            model=Groq(id="qwen-qwq-32b"),
            description="You are an assistant. Please reply based on the question.",
            tools=[DuckDuckGoTools()],
            markdown=True,
        )

        # Run the agent
        result = agent.run(
            question,
            stream=False,
            show_full_reasoning=True,
            stream_intermediate_steps=True,
        )

        
        st.markdown(result.content, unsafe_allow_html=True)
