import yaml
import os
from langchain_core.prompts import ChatPromptTemplate

def load_prompt(yaml_path: str, task_name: str) -> ChatPromptTemplate:
    """Loads a prompt from a YAML file and returns a ChatPromptTemplate."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Prompt file not found at {yaml_path}")
        
    with open(yaml_path, 'r') as f:
        prompts = yaml.safe_load(f)
        
    if task_name not in prompts:
        raise ValueError(f"Task '{task_name}' not found in {yaml_path}")
        
    task_config = prompts[task_name]
    system_prompt = task_config.get("system_prompt", "")
    human_template = task_config.get("human_template", "")
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])

if __name__ == "__main__":
    # Test loading
    try:
        p = load_prompt("prompts/qa_prompts.yaml", "qa_task")
        print("Prompt loaded successfully.")
        print(p.messages)
    except Exception as e:
        print(f"Error: {e}")
