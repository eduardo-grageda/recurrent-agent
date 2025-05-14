# RecurrentAgent: Enhanced Iterative Text Processing Tool

## Overview

RecurrentAgent is a powerful Python utility for processing large text files through an LLM-based agent pipeline. The tool operates by:

Parsing a configuration file that defines agent behavior
Dividing large text files into manageable chunks
Sending each chunk to an LLM with predefined system and user prompts
Collecting structured outputs from each chunk processing operation
Aggregating all responses into a comprehensive output collection

This approach enables efficient analysis of text too large to process in a single LLM call, while maintaining context and ensuring consistent structured output across all chunks.

## RecurrentAgent Development Specification

### Overview
Develop a Python tool called RecurrentAgent that processes large text files by breaking them down into manageable chunks, sending each chunk to an LLM with predefined prompts, and collecting structured results. This is particularly useful for tasks like entity extraction, summarization, concept detection, or any text processing that requires maintaining consistent output format across large documents.

### Core Requirements

The tool should:

Accept a configuration file (JSON or YAML) with all processing parameters
Read large text files in chunks of configurable size
Process each chunk through an LLM agent with defined system and user prompts
Collect structured responses from each chunk processing operation
Return a consolidated list of all responses
Handle errors gracefully and provide meaningful feedback
Support different LLM providers through a configurable interface

### Configuration File Structure

The configuration file should include:

- system_prompt: The system prompt that defines the agent's behavior and capabilities
- user_prompt: (Optional) Additional instructions for how the agent should process each chunk
- output_schema: The expected structure of the output object (used for validation)
- file_path: Path to the large text file to be processed
- chunk_size: Number of characters or tokens per chunk
- chunk_overlap: (Optional) Number of characters/tokens to overlap between chunks for context preservation
- llm_provider: Configuration for the LLM provider to use
- max_retries: (Optional) Number of retry attempts for failed API calls
- output_file: (Optional) Path to save the consolidated results
- logs_path: (Optional) Path to save all the prompts and responses from openai

### Expected Behavior

The tool should validate the configuration file for all required fields
It should check if the specified file exists and is readable
It should process the file in chunks, sending each to the LLM
For each chunk, it should validate that the response matches the expected schema
After processing all chunks, it should return a list of all valid responses
If specified, it should save the results to the output file

###  Example Usage

```python
pythonfrom recurrent_agent import RecurrentAgent

# Initialize the agent with a configuration file
agent = RecurrentAgent('config.json')

# Process the file and get results
results = agent.process()

# Access the results
for result in results:
    print(result)
```

### Implementation Guidelines

Modularity: Design the tool with clear separation of concerns
Error Handling: Implement robust error handling for file I/O and API calls
Logging: Add comprehensive logging for debugging and monitoring
Documentation: Provide clear documentation for all functions and classes
Testing: Include unit tests for core functionality
Extensibility: Design the tool to be easily extended for different LLM providers
Rate Limiting: Implement rate limiting to respect API provider limits
Progress Tracking: Add progress indicators for long-running processes

### Sample Configuration File

```json
json{
  "system_prompt": "You are a text analysis agent that identifies key people mentioned in text. For each person, extract their name and role if mentioned.",
  "user_prompt": "Analyze the following text and extract all people mentioned. Return an array of objects with 'name' and 'role' fields. If no people are mentioned, return an empty array.",
  "output_schema": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "role": {"type": "string"}
      },
      "required": ["name"]
    }
  },
  "file_path": "large_document.txt",
  "chunk_size": 4000,
  "chunk_overlap": 200,
  "llm_provider": {
    "type": "openai",
    "model": "gpt-4",
    "api_key": "env:OPENAI_API_KEY"
  },
  "max_retries": 3,
  "output_file": "extracted_people.json"
}
```

### Deliverables

- Python package with the RecurrentAgent implementation
- Documentation on usage and configuration options
- Example configuration files for common use cases
- Unit tests for core functionality
- Sample scripts demonstrating typical usage patterns

### Extra Credit Features

- Add support for maintaining state between chunks (for tasks that require cumulative context)
Implement parallel processing of chunks to improve performance
- Add support for different chunking strategies (by character, token, paragraph, etc.)
Implement progress visualization for long-running processes
- Add a CLI interface for running the tool directly from the command line