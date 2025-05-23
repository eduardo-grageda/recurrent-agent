import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.llms.base import BaseLLM
from openai import OpenAI
load_dotenv()

# Choose the LLM that will drive the agent
# llm = AzureChatOpenAI(azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), temperature=0, streaming=True)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'lmstudio')
LLM_MODEL = os.environ.get('LLM_MODEL', 'lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF')
SERPER_API_KEY= os.environ.get('SERPER_API_KEY', 'apikey')
# llm = ChatOpenAI(
#     model=LLM_MODEL
#     )

# client = OpenAI(api_key=OPENAI_API_KEY)


import os
from crewai import Agent, Task, Crew
# Importing crewAI tools
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

# Set up API keys
os.environ["SERPER_API_KEY"] = SERPER_API_KEY # serper.dev API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Instantiate tools
docs_tool = DirectoryReadTool(directory='./text')
file_tool = FileReadTool()
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# Create agents
researcher = Agent(
    role='Market Research Analyst',
    goal='Provide up-to-date market analysis of the AI industry',
    backstory='An expert analyst with a keen eye for market trends.',
    tools=[search_tool, web_rag_tool],
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Craft engaging blog posts about the AI industry',
    backstory='A skilled writer with a passion for technology.',
    tools=[docs_tool, file_tool],
    verbose=True
)

# Define tasks
research = Task(
    description='Research the latest trends in the AI industry and provide a summary.',
    expected_output='A summary of the top 3 trending developments in the AI industry with a unique perspective on their significance.',
    agent=researcher
)

write = Task(
    description='Write an engaging blog post about the AI industry, based on the research analyst\'s summary. Draw inspiration from the latest blog posts in the directory.',
    expected_output='A 4-paragraph blog post formatted in markdown with engaging, informative, and accessible content, avoiding complex jargon.',
    agent=writer,
    output_file='blog-posts/new_post.md'  # The final blog post will be saved here
)

# Assemble a crew with planning enabled
crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=True,
    planning=True,  # Enable planning feature
)

# Execute tasks
crew.kickoff()