Final datapoint structure:

repo_id: int
repo_name: str
project_context (dict - file_path: file_content)
file_context (list of content to be completed)
ground_truth (list of completion lines)
completion types (think, whether do we need it as a list. I prefer to have several datasets, each for completion type. Or even take only project-level completion)
context_size: str (small, medium, large, huge)
