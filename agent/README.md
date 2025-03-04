# recurrent-agent - agent/README.md

## Class description

The RecurrentAgent class is designed to process text iteratively, updating a cumulative summary based on predefined conditions. Here's a detailed description of the class and its methods:
Class: RecurrentAgent

### Purpose

The RecurrentAgent is responsible for reading a text file in chunks (default chunk size is an env variable) and updating a cumulative summary. It uses the system and user prompts to interact with a chat/completions API, which helps in determining whether additional context should be added to the summary.

### Key Components

1. Initialization
Parameters: The class is initialized with a text file, chunk size, system and user prompts for the agent and system and user prompts for the summary reviewer.
Attributes: It sets up necessary attributes such as the file path, chunk size, conditions, and an initial empty summary.
2. Iterative Processing
Method: A method to read the file in specified chunk sizes.
Functionality: This method iterates over the file, processing each chunk and updating the summary accordingly.
3. Cumulative Summary
Method: A method to update the summary.
Functionality: This method takes the current chunk and evaluates it against predefined conditions to decide if it should be added to the summary.
4. Reviewer
Method: A method to evaluate the response.
Functionality: This method reviews that the new response actually contains the exact same cumulative summary as the previous response, plus the new parts of the summary. This revision is also calling the API and the prompts can be constants.
5. API Interaction
Method: A method to send requests to the chat/completions API.
Functionality: This method constructs requests using the system and user parameters and processes the API's response to guide the summary update.
