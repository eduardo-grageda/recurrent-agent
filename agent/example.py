import os
from typing import List, Dict, Any
from crewai import LLM, Crew
import json

# Import the RecursiveAgent class
from recursive_agent import RecursiveAgent

def create_custom_agent(
    task_type: str,
    api_key: str,
    model: str = "gpt-4",
    temperature: float = 0.5
) -> RecursiveAgent:
    """
    Create a custom recursive agent based on the specified task type.
    
    Args:
        task_type: The type of task ("topic_detection", "sentiment_analysis", etc.)
        api_key: The API key for the language model
        model: The model to use
        temperature: The temperature setting
        
    Returns:
        A configured RecursiveAgent
    """
    llm = LLM(model=model, temperature=temperature, api_key=api_key)
    
    # Configure based on task type
    if task_type == "summarizer":
        system_prompt = """
        You are an expert at detecting topics in numbered conversation transcripts.
        
        Analyze the provided fragment of conversation transcript and determine:
        1. What topics are being discussed in this fragment
        2. If the topic is a continuation from the previous fragment
        3. In case there are topics starting in this fragment, use the number of the lines where the topic starts to respond with a JSON
        
        Be precise with the line numbers and focused in your analysis.
        """
        
        output_structure = {
            "topics": [{"topic_description":"description", "start_line":100}],
        }
        
        role = "Topic detector"
        
    elif task_type == "sentiment_analysis":
        system_prompt = """
        You are an expert at analyzing sentiment in conversation transcripts.
        
        Analyze the provided fragment of conversation transcript and determine:
        1. The overall sentiment of this fragment (positive, negative, neutral, mixed)
        2. Any sentiment shifts within this fragment
        3. How the sentiment compares to the previous fragments (if any)
        
        Focus on emotional cues, tone, and language that indicates the speakers' feelings.
        """
        
        output_structure = {
            "overall_sentiment": "positive/negative/neutral/mixed",
            "sentiment_by_speaker": {
                "speaker1": "sentiment",
                "speaker2": "sentiment"
            },
            "sentiment_shifts": [
                {
                    "from": "previous sentiment",
                    "to": "new sentiment",
                    "trigger": "what caused the shift",
                    "approximate_position": "beginning/middle/end of fragment"
                }
            ],
            "emotional_markers": ["list", "of", "notable", "emotional", "expressions"],
            "summary": "Brief summary of the sentiment patterns"
        }
        
        role = "Sentiment Analyzer"
        
    elif task_type == "action_item_extraction":
        system_prompt = """
        You are an expert at identifying action items and commitments in conversation transcripts.
        
        Analyze the provided fragment of conversation transcript and extract:
        1. Any tasks, commitments, or action items mentioned
        2. Who is responsible for each item
        3. Any mentioned deadlines or timeframes
        4. How these relate to action items from previous fragments (if any)
        
        Be thorough in identifying both explicit and implicit commitments.
        """
        
        output_structure = {
            "action_items": [
                {
                    "description": "what needs to be done",
                    "assignee": "who is responsible",
                    "deadline": "when it needs to be done (if specified)",
                    "context": "additional relevant details"
                }
            ],
            "follow_ups": ["list", "of", "items", "requiring", "follow-up"],
            "completed_items": ["previously", "mentioned", "items", "now", "marked", "as", "done"],
            "summary": "Brief summary of the action items and their status"
        }
        
        role = "Action Item Extractor"
        
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return RecursiveAgent(
        llm=llm,
        system_prompt=system_prompt,
        output_structure=output_structure,
        role=role,
        verbose=True
    )


# Example usage
if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.environ.get("")
    
    # Path to transcript file
    transcript_path = "../examples/large_text.txt"
    
    # Create the agent
    agent = create_custom_agent(
        task_type="summarizer",
        api_key=api_key
    )
#     crew = Crew(
#     agents=[agent.agent],
#     verbose=1
# )
    
    # Process the file directly with 50 lines per fragment
    results = agent.process_file(transcript_path, lines_per_fragment=50)
    
    # Save results to file
    with open("summary_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete! Results saved to summary_results.json")