import os
from typing import List, Dict, Any, Optional, Union, Callable
from crewai import Agent, Task, Crew, Process
import json
from langchain.llms.base import BaseLLM

class RecursiveAgent:
    """
    A recursive agent that processes fragments of content sequentially,
    maintaining context between iterations.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: str,
        output_structure: Dict[str, Any],
        tools: List[Any] = None,
        role: str = "Analyzer",
        verbose: bool = False
    ):
        """
        Initialize the recursive agent.
        
        Args:
            llm: The language model to use
            system_prompt: The system prompt that defines the agent's task
            output_structure: The expected structure of the agent's output
            tools: Optional list of tools for the agent
            role: Role description for the agent
            verbose: Whether to print detailed outputs
        """
        self.llm = llm
        self.system_prompt = system_prompt
        self.output_structure = output_structure
        self.tools = tools or []
        self.role = role
        self.verbose = verbose
        self.agent = self._create_agent()
        
    def _create_agent(self) -> Agent:
        """Create the CrewAI agent with the given configuration."""
        return Agent(
            role=self.role,
            goal=f"Process content fragments and provide structured analysis according to instructions",
            backstory=f"You are an expert at analyzing content and maintaining context across multiple iterations.",
            tools=self.tools,
            llm=self.llm,
            verbose=self.verbose
        )
    
    def _create_task(self, fragment: str, previous_context: Optional[Dict[str, Any]] = None) -> Task:
        """
        Create a task for the agent with the current fragment and previous context.
        
        Args:
            fragment: The current text fragment to analyze
            previous_context: Results from previous iterations
            
        Returns:
            A Task object for the agent to execute
        """
        context_str = json.dumps(previous_context) if previous_context else "null"
        structure_str = json.dumps(self.output_structure)
        task_description = f"""
        {self.system_prompt}
        
        # Current Fragment
        ```
        {fragment}
        ```
        
        # Previous Context
        ```json
        {context_str}
        ```
        
        # Expected Output Structure
        ```json
        {structure_str}
        ```
        
        Analyze the current fragment while considering the previous context.
        Return your analysis in valid JSON matching the expected output structure.
        """
        print(f'Creating task: {task_description}')
        
        return Task(
            description=task_description,
            agent=self.agent,
            context=context_str,
            expected_output=f'a JSON whit this structure {structure_str}'
        )
    
    def process_fragment(self, fragment: str, previous_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a single fragment, taking into account previous context.
        
        Args:
            fragment: The current text fragment to analyze
            previous_context: Results from previous iterations
            
        Returns:
            Structured output from the agent
        """
        task = self._create_task(fragment, previous_context)
        print(f'Task: {task}')
        result = task.execute()
        
        try:
            # Try to parse the result as JSON
            parsed_result = json.loads(result)
            return parsed_result
        except json.JSONDecodeError:
            # If the agent didn't return valid JSON, try to extract JSON from the text
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', result)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
                    
            # If we still don't have valid JSON, return a formatted error
            return {
                "error": "Failed to parse agent response as JSON",
                "raw_response": result
            }
    
    def process_all_fragments(self, fragments: List[str]) -> List[Dict[str, Any]]:
        """
        Process a list of fragments sequentially, passing context between iterations.
        
        Args:
            fragments: List of text fragments to process
            
        Returns:
            List of results from each iteration
        """
        results = []
        context = None
        
        for i, fragment in enumerate(fragments):
            if self.verbose:
                print(f"Processing fragment {i+1}/{len(fragments)}")
                
            result = self.process_fragment(fragment, context)
            results.append(result)
            context = result
            
        return results
        
    def process_file(self, file_path: str, lines_per_fragment: int = 50) -> List[Dict[str, Any]]:
        """
        Process a large text file by splitting it into fragments of specified number of lines.
        
        Args:
            file_path: Path to the text file to process
            lines_per_fragment: Number of lines to include in each fragment
            
        Returns:
            List of results from processing each fragment
        """
        # Read the file and split into fragments
        fragments = self.split_file_by_lines(file_path, lines_per_fragment)
        
        if self.verbose:
            print(f"Split file into {len(fragments)} fragments")
            
        # Process all fragments
        return self.process_all_fragments(fragments)
    
    @staticmethod
    def split_file_by_lines(file_path: str, lines_per_fragment: int) -> List[str]:
        """
        Split a file into fragments based on number of lines.
        
        Args:
            file_path: Path to the file to split
            lines_per_fragment: Number of lines per fragment
            
        Returns:
            List of text fragments
        """
        fragments = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            current_fragment = []
            line_count = 0
            
            for line in file:
                current_fragment.append(line)
                line_count += 1
                
                if line_count >= lines_per_fragment:
                    fragments.append(''.join(current_fragment))
                    current_fragment = []
                    line_count = 0
            
            # Add any remaining lines as the last fragment
            if current_fragment:
                fragments.append(''.join(current_fragment))
        
        return fragments