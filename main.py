from agent import recurrent_agent as ra


# Example parameters
file_path = "examples/large_text.txt"
system_prompt = "You are a recurrent AI agent who summarizes extensive text files, filtering out only the key information. In each iteration, you get the summary done by the previous iterations you have to identify the line numbers only of the most important topics in a new set of lines. Do not explain any procedure, just respond with a json."
user_prompt = "This is the summary from the previous iteration: {summary} of a file with {lines} lines, review the next lines and determine if there is important lines to add to the summary. {chunk}. Respond with the same json structure as it will be used for the same purpose."
reviewer_system_prompt = "System prompt for the summary reviewer"
reviewer_user_prompt = "User prompt for the summary reviewer"

# Instantiate the RecurrentAgent
agent = ra.RecurrentAgent(
    file_path=file_path,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    reviewer_system_prompt=reviewer_system_prompt,
    reviewer_user_prompt=reviewer_user_prompt
)

# Now you can use the agent to process the file
agent.process_file()

# Access the summary
print(agent.summary)
