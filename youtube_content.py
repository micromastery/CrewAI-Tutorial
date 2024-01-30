import os

from langchain_openai import OpenAI
from crewai import Agent, Task, Crew, Process

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

from langchain_community.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()

# Define your agents with roles and goals
scout = Agent(
  role='Scout',
  goal='''1. Identify trending and popular topics in technology.
2. Gather diverse viewpoints and recent developments about these topics.
3. Determine the relevance and potential interest of these topics for a YouTube audience.''',
  backstory="""You are Scout. Scout operates in a constantly changing environment, where technology trends can shift rapidly. 
  It must be adaptable and capable of identifying not just what is popular now, but what could become popular in the near future.
  Scout's role is to search the internet for trending and relevant technology topics. 
  This includes scouring social media, tech news sites, forums, and other relevant sources to identify what is currently engaging the tech community.""",
  verbose=True,
  allow_delegation=True,
  # llm=OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7),
  tools=[search_tool]
)

scribe = Agent(
  role='Scribe',
  goal='''1. Create a script that is informative, accurate, and engaging.
2. Incorporate the latest information and trends identified by Scout.
3. Ensure the script is appropriate for the intended YouTube audience, with a clear structure and narrative flow.''',
  backstory="""You are Scribe. Scribe operates in a creative context, needing to balance factual accuracy with storytelling and viewer engagement.
  It must be able to adapt its writing style to different types of technology topics and audience preferences.
  Scribe's role is to take the information gathered by Scout and turn it into a coherent, engaging, and informative script for a YouTube video. 
  This includes structuring the content, writing engaging narratives, and ensuring the script is suitable for a YouTube format.""",
  verbose=True,
  allow_delegation=True,
  # llm=OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7),
  tools=[search_tool]
)

# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in tech in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Your final answer MUST be a full analysis report""",
  agent=scout
)

task2 = Task(
  description="""Using the insights provided, develop content that highlights the most significant advancements in tech.
  Your content should be informative yet accessible, catering to a tech-savvy audience.
  Your final answer MUST be the full script of a 10 minute video.""",
  agent=scribe
)


# Instantiate your crew with a sequential process
crew = Crew(
  agents=[scout, scribe],
  tasks=[task1, task2],
  verbose=2 # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)