# recurrent-agent - /README.md

A recurring approach for an agent with a cumulative summary

## Module Overview

The module will consist of a class RecurrentAgent that processes text in chunks. Each iteration will update a cumulative summary based on the current chunk and predefined conditions. The agent will decide whether to add more context to the summary.
Key Components.

1. Initialization: The agent will be initialized with a text file and parameters for chunk size and conditions for adding context.
2. Iterative Processing: The agent will read the file in chunks, updating the summary with each iteration.
3. Cumulative Summary: The summary will be updated based on the current chunk and conditions.
4. Condition Evaluation: Conditions will determine if additional context is needed in the summary.

The current file structure is this:

```bash
├── LICENSE
├── README.md
├── agent
│   ├── README.md
│   ├── __init__.py
│   └── recurrent_agent.py
├── examples
│   ├── large_text.txt
│   ├── system.md
│   └── user.md
└── main.py
```
