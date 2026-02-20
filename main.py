import os
from dotenv import load_dotenv
import requests
from crewai import Agent, Task, Crew, LLM
from crewai_tools import TavilySearchTool
from crewai.tools import tool
from datetime import datetime

load_dotenv()
@tool("post_to_devto")
def post_to_devto(title: str, markdown_content: str, tags: list):
    """
    Posts a technical article to Dev.to.
    Args:
        title: The headline of the post.
        markdown_content: The full article body in Markdown.
        tags: A list of up to 4 strings (e.g., ["python", "ai"]).
    """
    api_key = os.getenv("DEVTO_API_KEY")
    
    # 1. Check for the API Key immediately
    if not api_key:
        return "Error: DEVTO_API_KEY is missing from environment variables."

    url = "https://dev.to/api/articles"
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }
    
    data = {
        "article": {
            "title": title,
            "body_markdown": markdown_content,
            "published": True,
            "tags": tags[:4] if tags else []
        }
    }
    
    try:
        # 2. Set a timeout so the agent doesn't hang forever
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        # 3. This raises an exception for 4xx or 5xx status codes
        response.raise_for_status()
        
        return f"Success! Draft created at: {response.json().get('url')}"

    except requests.exceptions.HTTPError as http_err:
        # 4. Handle specific API rejections (e.g., 422 Unprocessable Entity)
        return f"HTTP error occurred: {http_err} - Details: {response.text}"
    except Exception as err:
        # 5. Catch-all for network issues or timeouts
        return f"An unexpected error occurred: {err}"

#Initialize the search tool
search_tool = TavilySearchTool(
    search_tool = TavilySearchTool(
    topic="news",           # ⬅️ Focuses on news sites rather than general web pages
    time_range="day",       # ⬅️ Restricts results to the last 24 hours
    search_depth="advanced" # ⬅️ Higher quality/more recent results
)
)

# Get today's date as a string
current_date = datetime.now().strftime("%Y-%m-%d")

#Initialize the Brain
llm = LLM(model="gemini/gemini-3-flash-preview")

#Define the researcher agent
researcher = Agent(
    role="Senior Tech Researcher",
    goal=f'Find the top 3 tech stories published on {current_date}.',
    backstory="""You are a meticulous fact-checker. 
    When you search, you verify that the articles were published TODAY. 
    Ignore any articles from 2024 or 2025 that talk about 'future predictions.'
    Only report on events that have actually happened in the last 24 hours.""",
  tools=[search_tool], #The agent now has eyes
  llm=llm,
  verbose=True, #This lets us see the agent's thought process in the console
  inject_date=True,      # ⬅️ Adds the current date to every task
  date_format="%B %d, %Y" # ⬅️ Optional: Formats it as "February 20, 2026"
)

# The Social Media / Tech Blogger Agent
writer = Agent(
    role='Senior Tech Editor (Ex-Verge/CNET)',
    goal='Write a data-driven analysis of the single most important tech story today.',
    backstory="""You have 10 years of experience at CNET. You despise fluff. 
    You are an expert at identifying the 'So What?' of a story. 
    Your writing is grounded in hard numbers: market caps, investment totals, 
    and court ruling counts (e.g., 6-3 decisions).""",
    llm=llm,
    verbose=True,
    tools = [post_to_devto]
)

# 1. Define the Task
research_task = Task(
    description="""Find the top 3 tech stories from today ({today}). 
    CRITICAL: You must only report on events that HAVE HAPPENED. 
    Do not report on rumors, upcoming releases, or predictions as if they are current facts.""",
    expected_output="A list of 3 verified facts from today's news with source URLs.",
    agent=researcher
)

# Task 2: Writing the content
writing_task = Task(
    description="""R1. Identify the SINGLE most dominant theme from the research.
    2. Write a 500-word blog post. Use concrete details only (no 'recent updates').
    3. Include at least one hard data point (e.g., '$30 Billion', '6-3 ruling') per section.
    4. Headline must be punchy and focus ONLY on that lead theme.""",
    expected_output="A professional, data-backed tech blog and a LinkedIn post.",
    agent=writer,
    context=[research_task] # ⬅️ This tells the writer to look at the researcher's work!
)

# 2. Form the Crew
tech_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True # This ensures the Crew manager also reports progress
)

# 3. Kick it off!
result = tech_crew.kickoff(inputs={'today': current_date})


print("\n\n########################")
print("## FINAL RESULT ##")
print("########################\n")
print(result)