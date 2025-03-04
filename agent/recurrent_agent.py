from langchain_core.prompts import PromptTemplate, ChatPromptTemplate  # Import PromptTemplate
from langchain_openai import ChatOpenAI
import dotenv
import os
from typing import Optional
from typing_extensions import Annotated, TypedDict

dotenv.load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'lm-studio')
LLM_MODEL = os.environ.get('LLM_MODEL', 'lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF')
llm = ChatOpenAI(model=LLM_MODEL)
CHUNK_SIZE = os.environ.get('CHUNK_SIZE', 20)

class Topic(TypedDict):
    """Topic by line."""
    lines: Annotated[tuple, ..., "from line to line (1,15)"]
    description: Annotated[Optional[str], ..., "Topic description"]
    keywords: Annotated[Optional[str], ..., "Keywords separated by comma"]
    relevance: Annotated[Optional[int], ..., "How relevant the topic is, from 1 to 5"]

class Summary(TypedDict):
    topics: Annotated[list[Topic], ..., "List of topics"]

class RecurrentAgent:
    def __init__(self, file_path, system_prompt, user_prompt, reviewer_system_prompt, reviewer_user_prompt):
        """
        Initialize the RecurrentAgent with necessary parameters.
        
        :param file_path: Path to the text file to be processed.
        :param chunk_size: Size of each chunk to be read from the file.
        :param system_prompt: System prompt for the agent.
        :param user_prompt: User prompt for the agent.
        :param reviewer_system_prompt: System prompt for the summary reviewer.
        :param reviewer_user_prompt: User prompt for the summary reviewer.
        """
        self.file_path = file_path
        self.chunk_size = CHUNK_SIZE
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.reviewer_system_prompt = reviewer_system_prompt
        self.reviewer_user_prompt = reviewer_user_prompt
        self.summary = ""
        self.lines=0

    def process_file(self):
        """
        Read the file in specified chunk sizes and update the summary.
        """
        with open(self.file_path, 'r') as file:
            self.lines = sum(1 for _ in file)
            file.seek(0)  # Reset file pointer to the beginning after counting lines
            while chunk := file.read(self.chunk_size):
                summary = self.invoke_agent(self.summary, chunk)
                self.update_summary(summary)

    def update_summary(self, summary):
        """
        Update the summary based on the current chunk and predefined conditions.
        
        :param summary: The current chunk of text to be evaluated.
        """

        self.summary = summary

    def evaluate_conditions(self, chunk):
        """
        Evaluate conditions to determine if the chunk should be added to the summary.
        
        :param chunk: The current chunk of text to be evaluated.
        :return: Boolean indicating if the chunk should be added.
        """
        # Placeholder for actual condition evaluation logic
        return True

    def review(self, response):
        """
        Review the summary to ensure it contains the correct cumulative information.
        """
        return response

    def invoke_agent(self, summary, chunk):
        """
        Send requests to the chat/completions API.
        
        :param prompt: The prompt to be sent to the API.
        :return: The API's response.
        """
        structured_llm = llm.with_structured_output(Summary)
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("user", self.user_prompt)]
        )
        prompt = prompt_template.invoke({"summary": summary, "lines": self.lines, "chunk":chunk})
        response = structured_llm.invoke(prompt)

        reviewed_response = self.review(response)

        return reviewed_response