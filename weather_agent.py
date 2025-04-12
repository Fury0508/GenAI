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
# print(run_command("ls"))
def get_weather(city:str):
    print("🔨 Tool called: get_weather", city)
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"The weather is {city} is {response.text}"
    return "Something went wrong"

def add(x,y):
    print("🔨 Tool called: add", x,y)
    return x+y

available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description" : "Takes a city name as input and return the current weather of the city"
    },
    "add":{
        "fn": add,
        "description": "Takes input two numbers and add them"
    },
    "run_command":{
        "fn": run_command,
        "description": "Takes a command as input to execute on system and return output"
    }
}
system_prompt = """
    You are helpful AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution , based on the planning,
    select the relevant tool from the available tool and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.
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
    - get_weather: Takes a city name as input and return the current weather of the city
    - run_command: Takes a command as input to execute on system and return output
    Example:
    User Query: What is weather of new york?
    Output: {{ "Step" : "plan", "content":"The user in intreseted in weather data of the new york"}},
    Output: {{ "Step" : "plan", "content":"From the available tools I should call get_weather"}},
    Output: {{ "Step" : "action", "function":"get_weather","input":"new york"}},
    Output: {{ "Step" : "observe", "output":"12 Degree Cel"}},
    Output: {{ "Step" : "output", "content":"The weather for new york seems to be 12 degree"}},
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