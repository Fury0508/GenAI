"""Overview:

Your task for Project 3 is to build a terminal-based AI Agent focused on coding and full-stack project development. The agent should function entirely through the terminal and be capable of performing a wide range of development tasks.

Requirements:

The AI Agent must be terminal-based only (no GUI).
It should be specialized in creating full-stack projects, including:
Generating folder and file structures.
Writing code into appropriate files (both frontend and backend).
Running commands like pip install, npm install, npm run build, etc.
It should support follow-up prompts, allowing iterative development. For example:
If a user says “Now add a login page,” the agent should:
Parse existing files.
Identify where and how to add the feature.
Modify or create necessary files accordingly.

Objectives:

Develop a responsive and intelligent coding assistant.
Ensure it can understand context from existing project files.
Maintain a smooth workflow through command-line interaction."""

from dotenv import load_dotenv
from openai import OpenAI
import json
import requests
import os
load_dotenv()
client = OpenAI()

def run_command(command):
    result = os.system(command=command)
    return result



available_tools = {
    "run_command":{
        "fn": run_command,
        "description": "Takes a command as input to execute on system and return output"
    }
}
system_prompt = """
You are helpful AI Assistant who is specialized in coding and full-stack project development.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution , based on the planning,
    select the relevant tool from the available tool and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.

    It should be specialized in creating full-stack projects, including:
    - Generating folder and file structures.
    - Writing code into appropriate files (both frontend and backend).
    - Running commands like pip install, npm install, npm run build, etc.
    - It should support follow-up prompts, allowing iterative development.
    Rules:
     - Follow the strict JSON output as per output schema.
     - Always perform one step at a time and wait for next input
     - Carefully analyse the user query

    Output:
    {{
        "step" : "string",
        "content" : "string",
        "function" : "The name of function if the stp is action",
        "input" : "The input parameter for the function",
    }}
    Available Tools:
    - run_command: Takes a command as input to execute on system and return output
    Example:
    User Query : Write a fastapi code ?
    Output: {{ "Step" : "plan", "content":"The user is interested in writing a fastapi code"}},
    Output: {{ "Step" : "plan", "content":"From the available tool access the current directory and create a folder in that folder install everything "}},
    Output: {{ "Step" : "plan", "content":"in that folder install the requirements for creating the fastapi code"}},
    Output: {{ "Step" : "action", "function":"run_command","input":"fastapicode"}},
    Output: {{ "Step" : "observe", "output":"create a folder with which consist of all requirement for the fastapi code"}},
    Output: {{ "Step" : "output", "content":"boiler plate code is there in the folder with a fastapi.py file"}},

    Example:
    User Query : Write code to save user detials ?
    Output: {{ "Step" : "plan", "content":"The user is interested in saving user details thorugh fast api code Parse existing files"}},
    Output: {{ "Step" : "plan", "content":"Identify where and how to add the feature. "}},
    Output: {{ "Step" : "plan", "content":"in that folder install the requirements for creating the fastapi code"}},
    Output: {{ "Step" : "action", "function":"run_command","input":"fastapicode"}},
    Output: {{ "Step" : "observe", "output":"Modify or create necessary files accordingly."}},
    Output: {{ "Step" : "output", "content":"Modificaton is done in the code"}},

"""


messages = [
    {"role": "system","content": system_prompt}

]

while True:
    user_query = input("> ")
    messages.append({"role": "user","content": user_query})
    while True:

        response = client.chat.completions.create(
        model = "gpt-4o",
        response_format={"type":"json_object"},
        messages=messages
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({"role":"assistant", "content":json.dumps(parsed_output)})

        if parsed_output.get("step") == "plan":
            print(f"🧠:{parsed_output.get('content')}")
            continue
        
        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if available_tools.get(tool_name,False) != False:
                output = available_tools[tool_name].get("fn")(tool_input)
                messages.append({"role" : "assistant" , "content":json.dumps({"step":"observe", "output": output})})
                continue
        
        
        if parsed_output.get("step") == "output":
            print(f"🤖 : {parsed_output.get('content')}")
            break