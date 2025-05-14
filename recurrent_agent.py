"""
RecurrentAgent: A Python tool for processing large text files with LLM-based agents

This module provides the RecurrentAgent class and associated utilities for
processing large text files by breaking them into manageable chunks, sending
each chunk to an LLM with predefined prompts, and collecting structured results.
"""

import json
import yaml
import os
import logging
import time
import jsonschema
from typing import List, Dict, Any, Optional, Union, Callable
import requests
from tqdm import tqdm
import openai
import anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RecurrentAgent")

class LLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM provider with configuration"""
        self.config = config
        self.setup()
    
    def setup(self):
        """Set up the LLM provider (override in subclasses)"""
        pass
    
    def process_chunk(self, system_prompt: str, user_prompt: str, chunk: str) -> Dict[str, Any]:
        """Process a chunk using the LLM provider (override in subclasses)"""
        raise NotImplementedError("LLM providers must implement process_chunk")

    def save_log(self, provider: str, system_prompt: str, user_prompt: str, chunk: str, response: dict):
        import os
        import json
        from datetime import datetime
        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        log_path = os.path.join(logs_dir, f"{provider}_{timestamp}.json")
        log_entry = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "chunk": chunk,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def setup(self):
        """Set up the OpenAI API client"""
        api_key = self.config.get("api_key", "")
        
        # Handle environment variable references
        if api_key.startswith("env:"):
            env_var = api_key.split(":", 1)[1]
            api_key = os.environ.get(env_var, "")
            if not api_key:
                raise ValueError(f"Environment variable {env_var} not set or empty")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = self.config.get("model", "gpt-4")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 1024)
    
    def process_chunk(self, system_prompt: str, user_prompt: str, chunk: str) -> Dict[str, Any]:
        """Process a chunk using the OpenAI API"""
        import os
        import json
        from datetime import datetime
        try:
            # Instruct the model to always respond with a JSON object
            json_instruction = "Respond ONLY with a valid JSON object. Do not include anything else."
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_prompt}\n\n{chunk}\n\n{json_instruction}"}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            result = json.loads(response.choices[0].message.content)
            self.save_log('openai', system_prompt, user_prompt, chunk, result)
            return result
        except Exception as e:
            logger.error(f"Error processing chunk with OpenAI: {e}")
            raise

class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""
    
    def setup(self):
        """Set up the Anthropic API client"""
        api_key = self.config.get("api_key", "")
        
        # Handle environment variable references
        if api_key.startswith("env:"):
            env_var = api_key.split(":", 1)[1]
            api_key = os.environ.get(env_var, "")
            if not api_key:
                raise ValueError(f"Environment variable {env_var} not set or empty")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = self.config.get("model", "claude-3-opus-20240229")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 1024)
    
    def process_chunk(self, system_prompt: str, user_prompt: str, chunk: str) -> Dict[str, Any]:
        """Process a chunk using the Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": f"{user_prompt}\n\n{chunk}"}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract JSON from the response
            content = response.content[0].text
            # Try to find JSON in the response
            try:
                # Look for JSON between triple backticks if formatted that way
                if "```json" in content and "```" in content.split("```json", 1)[1]:
                    json_str = content.split("```json", 1)[1].split("```", 1)[0].strip()
                    result = json.loads(json_str)
                else:
                    # Try to parse the entire response as JSON
                    result = json.loads(content)
                self.save_log('anthropic', system_prompt, user_prompt, chunk, result)
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response from Anthropic")
                raise ValueError(f"Invalid JSON response from Anthropic: {content}")
        except Exception as e:
            logger.error(f"Error processing chunk with Anthropic: {e}")
            raise

# Factory for creating LLM providers based on configuration
def create_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """Create an LLM provider based on configuration"""
    provider_type = config.get("type", "").lower()
    
    if provider_type == "openai":
        return OpenAIProvider(config)
    elif provider_type == "anthropic":
        return AnthropicProvider(config)
    else:
        raise ValueError(f"Unsupported LLM provider type: {provider_type}")

class RecurrentAgent:
    """
    A tool for processing large text files with LLM-based agents.
    
    The RecurrentAgent breaks down large text files into manageable chunks,
    processes each chunk using an LLM with predefined prompts, and collects
    structured results.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the RecurrentAgent with a configuration file.
        
        Args:
            config_path: Path to the configuration file (JSON or YAML)
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self._validate_config()
        
        # Initialize LLM provider
        self.llm_provider = create_llm_provider(self.config["llm_provider"])
        
        # Initialize results list
        self.results = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load the configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            The configuration as a dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.lower().endswith(".json"):
                return json.load(f)
            elif config_path.lower().endswith((".yaml", ".yml")):
                return yaml.safe_load(f)
            else:
                raise ValueError("Configuration file must be JSON or YAML")
    
    def _validate_config(self):
        """Validate the configuration"""
        required_fields = ["system_prompt", "file_path", "chunk_size", "llm_provider"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        # Validate file path
        if not os.path.exists(self.config["file_path"]):
            raise FileNotFoundError(f"Text file not found: {self.config['file_path']}")
        
        # Validate LLM provider config
        llm_provider_config = self.config["llm_provider"]
        if "type" not in llm_provider_config:
            raise ValueError("LLM provider configuration must include 'type'")
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split the text into chunks.
        
        Args:
            text: The text to chunk
            
        Returns:
            A list of text chunks
        """
        chunk_size = self.config["chunk_size"]
        chunk_overlap = self.config.get("chunk_overlap", 0)
        
        chunks = []
        i = 0
        text_length = len(text)
        
        while i < text_length:
            # Calculate end position for this chunk
            logger.info(f"Chunk {i}")
            
            end = min(i + chunk_size, text_length)
            
            # Get the chunk
            chunk = text[i:end]
            chunks.append(chunk)
            
            # Move to next chunk position, considering overlap
            next_i = end - chunk_overlap if end < text_length and chunk_overlap > 0 else end
            
            # Break the loop if we're not making progress
            if next_i <= i:
                break
                
            i = next_i
        logger.info(chunks)
        
        return chunks
    
    def _validate_response(self, response: Any) -> bool:
        """
        Validate that the response matches the expected schema.
        
        Args:
            response: The response to validate
            
        Returns:
            True if the response is valid, False otherwise
        """
        if "output_schema" not in self.config:
            # No schema defined, assume valid
            return True
        
        try:
            jsonschema.validate(instance=response, schema=self.config["output_schema"])
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.warning(f"Response validation failed: {e}")
            return False
    
    def _process_chunk(self, chunk: str, retry_count: int = 0) -> Optional[Dict[str, Any]]:
        """
        Process a single chunk using the LLM provider.
        
        Args:
            chunk: The chunk to process
            retry_count: Current retry attempt
            
        Returns:
            The processed response as a dictionary, or None if processing failed
        """
        system_prompt = self.config["system_prompt"]
        user_prompt = self.config.get("user_prompt", "")
        max_retries = self.config.get("max_retries", 3)
        
        try:
            response = self.llm_provider.process_chunk(system_prompt, user_prompt, chunk)
            
            # Validate response
            if self._validate_response(response):
                return response
            else:
                logger.warning("Response validation failed")
                if retry_count < max_retries:
                    logger.info(f"Retrying chunk (attempt {retry_count + 1}/{max_retries})")
                    return self._process_chunk(chunk, retry_count + 1)
                else:
                    logger.error(f"Max retries reached for chunk")
                    return None
                
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
            if retry_count < max_retries:
                # Exponential backoff
                wait_time = 2 ** retry_count
                logger.info(f"Retrying in {wait_time} seconds (attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self._process_chunk(chunk, retry_count + 1)
            else:
                logger.error(f"Max retries reached for chunk")
                return None
    
    def process(self) -> List[Dict[str, Any]]:
        """
        Process the text file and return the results.
        
        Returns:
            A list of processed responses
        """
        # Read the text file
        file_path = self.config["file_path"]
        logger.info(f"Reading file {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Split into chunks
        logger.info(f"Splitting...")
        chunks = self._chunk_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Process each chunk
        results = []
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                response = self._process_chunk(chunk)
                
                if response is not None:
                    # Handle the case where the result is an empty list or dict
                    if (isinstance(response, list) or isinstance(response, dict)) and not response:
                        logger.info(f"Chunk {i+1} returned empty result")
                    else:
                        results.append(response)
                        logger.info(f"Successfully processed chunk {i+1}")
                
                pbar.update(1)
        
        # Save results to file if specified
        output_file = self.config.get("output_file")
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved results to {output_file}")
        
        # Store results in instance
        self.results = results
        return results

def main():
    """Command-line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process a large text file with an LLM agent")
    parser.add_argument("config", help="Path to the configuration file")
    args = parser.parse_args()
    
    try:
        agent = RecurrentAgent(args.config)
        results = agent.process()
        print(f"Processed {len(results)} chunks with valid results")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
    
    # localhost run with lm studio running in windows and agent running in wsl:
#     curl http://172.30.240.1:1234/v1/chat/completions   -H "Content-Type: application/json"   -d '{
#     "model": "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
#     "messages": [
#       { "role": "system", "content": "Always answer in rhymes." },
#       { "role": "user", "content": "Introduce yourself." }
#     ],
#     "temperature": 0.7,
#     "max_tokens": -1,
#     "stream": true
# }'