# RecurrentAgent

A Python tool for processing large text files with LLM-based agents. This tool breaks down large text files into manageable chunks, sends each chunk to an LLM with predefined prompts, and collects structured results.

## Features

- Process large text files that exceed LLM context windows
- Configure system and user prompts for the LLM agent
- Define expected output schema for validation
- Support for different LLM providers (OpenAI, Anthropic)
- Configurable chunk size and overlap
- Error handling with automatic retries
- Progress tracking for long-running processes
- Optional output file saving

## Installation

```bash
pip install recurrent-agent
```

## Quick Start

1. Create a configuration file in JSON or YAML format:

```json
{
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

2. Run the agent:

```python
from recurrent_agent import RecurrentAgent

# Initialize the agent with a configuration file
agent = RecurrentAgent('config.json')

# Process the file and get results
results = agent.process()

# Access the results
for result in results:
    print(result)
```

3. Or use the command line:

```bash
python -m recurrent_agent config.json
```

## Configuration Options

| Option | Description | Required | Default |
|--------|-------------|----------|---------|
| `system_prompt` | The system prompt that defines the agent's behavior | Yes | - |
| `user_prompt` | Additional instructions for processing each chunk | No | "" |
| `output_schema` | JSON Schema definition for validating responses | No | - |
| `file_path` | Path to the large text file to process | Yes | - |
| `chunk_size` | Number of characters per chunk | Yes | - |
| `chunk_overlap` | Number of characters to overlap between chunks | No | 0 |
| `llm_provider` | Configuration for the LLM provider | Yes | - |
| `max_retries` | Number of retry attempts for failed API calls | No | 3 |
| `output_file` | Path to save the consolidated results | No | - |

### LLM Provider Configuration

#### OpenAI

```json
"llm_provider": {
  "type": "openai",
  "model": "gpt-4",
  "api_key": "sk-...",  // Or "env:OPENAI_API_KEY"
  "temperature": 0.7,
  "max_tokens": 1024
}
```

#### Anthropic (Claude)

```json
"llm_provider": {
  "type": "anthropic",
  "model": "claude-3-opus-20240229",
  "api_key": "sk-ant-...",  // Or "env:ANTHROPIC_API_KEY"
  "temperature": 0.7,
  "max_tokens": 1024
}
```

## Common Use Cases

- **Entity Extraction**: Extract people, organizations, locations from large documents
- **Summarization**: Generate summaries of key sections in large documents
- **Content Analysis**: Identify themes, sentiments, or key concepts
- **Data Extraction**: Pull structured data from unstructured text
- **Document Classification**: Categorize sections of text based on content

## Example: Processing a Large Research Paper

```json
{
  "system_prompt": "You are a research assistant that identifies key findings, methodologies, and conclusions from scientific papers.",
  "user_prompt": "Analyze the following section of a research paper and extract key findings, methodology details, and limitations if mentioned. Structure your response as JSON with 'findings', 'methodology', and 'limitations' fields, each containing an array of strings.",
  "output_schema": {
    "type": "object",
    "properties": {
      "findings": {
        "type": "array",
        "items": {"type": "string"}
      },
      "methodology": {
        "type": "array",
        "items": {"type": "string"}
      },
      "limitations": {
        "type": "array",
        "items": {"type": "string"}
      }
    },
    "required": ["findings", "methodology", "limitations"]
  },
  "file_path": "research_paper.txt",
  "chunk_size": 6000,
  "chunk_overlap": 500,
  "llm_provider": {
    "type": "anthropic",
    "model": "claude-3-opus-20240229",
    "api_key": "env:ANTHROPIC_API_KEY"
  },
  "output_file": "paper_analysis.json"
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.