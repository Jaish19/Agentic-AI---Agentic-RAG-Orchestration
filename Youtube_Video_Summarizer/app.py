import streamlit as st
from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.youtube import YouTubeTools
import os

# Get Google API key (assuming you've saved it in Colab's `userdata`)
from google.colab import userdata
os.environ["GOOGLE_API_KEY"] = "YOUR_API_HERE"

st.set_page_config(page_title="YouTube Analyzer", layout="wide")

st.title("📺 YouTube Video Analyzer with Gemini")
video_url = st.text_input("Enter YouTube Video URL", "")

if video_url:
    with st.spinner("Analyzing video..."):
        youtube_agent = Agent(
            name="YouTube Agent",
            model=Gemini(id="gemini-2.0-flash"),
            tools=[YouTubeTools()],
            show_tool_calls=True,
            instructions=dedent("""\
                You are an expert YouTube content analyst with a keen eye for detail! 🎓
                Follow these steps for comprehensive video analysis:
                1. Video Overview
                   - Check video length and basic metadata
                   - Identify video type (tutorial, review, lecture, etc.)
                   - Note the content structure
                2. Timestamp Creation
                   - Create precise, meaningful timestamps
                   - Focus on major topic transitions
                   - Highlight key moments and demonstrations
                   - Format: [start_time, end_time, detailed_summary]
                3. Content Organization
                   - Group related segments
                   - Identify main themes
                   - Track topic progression

                Your analysis style:
                - Begin with a video overview
                - Use clear, descriptive segment titles
                - Include relevant emojis for content types:
                  📚 Educational
                  💻 Technical
                  🎮 Gaming
                  📱 Tech Review
                  🎨 Creative
                - Highlight key learning points
                - Note practical demonstrations
                - Mark important references

                Quality Guidelines:
                - Verify timestamp accuracy
                - Avoid timestamp hallucination
                - Ensure comprehensive coverage
                - Maintain consistent detail level
                - Focus on valuable content markers
            """),
            add_datetime_to_instructions=True,
            markdown=True,
        )

        result = youtube_agent.run(video_url, stream=False)
        # response = result.get("content", "⚠️ No output returned from the agent.")
        st.markdown(result.content, unsafe_allow_html=True)
