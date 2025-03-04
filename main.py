from agent import recurrent_agent as ra


# Example parameters
file_path = "examples/large_text.txt"
chunk_size = 100  # Size of each chunk in lines
system_prompt = "System prompt for the agent"
user_prompt = "User prompt for the agent"
reviewer_system_prompt = "System prompt for the summary reviewer"
reviewer_user_prompt = "User prompt for the summary reviewer"

# Instantiate the RecurrentAgent
agent = ra.RecurrentAgent(
    file_path=file_path,
    chunk_size=chunk_size,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    reviewer_system_prompt=reviewer_system_prompt,
    reviewer_user_prompt=reviewer_user_prompt
)

# Now you can use the agent to process the file
agent.process_file()

# Access the summary
print(agent.summary)
