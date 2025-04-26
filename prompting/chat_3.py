from dotenv import load_dotenv
from openai import OpenAI
import json
load_dotenv()

client = OpenAI()

system_prompt = """
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

For the given user input, analyse the input and break down the problem step by step.
Atleast think 5-6 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result"

Rules:
1. Follow the strict JSON output as per output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{{ step: "string", content: "string"}}

Example:
Input: What is 2+2
Output: {{ step: "analyse",content: "Alright! The user is interested in maths query and he is asking a basic arthermatic operation."}}
Output: {{ step: "think", content:  "To perform the addition I must go from left to right and add all the operands."}}
Output: {{ step: "output", content: "4"}}
Output: {{ strp: "validate", content: "seems like 4 is correct ans for 2+2"}}
Output: {{ step" "result", content: "2+2 = 4 and that is caluclated by adding all numbers"}}
"""


result = client.chat.completions.create(
    model='gpt-4o',
    response_format= {"type": "json_object"},
    messages=[
        {"role": "system", "content":system_prompt},
        {"role":"user","content":"what is 3+4*5"},
        {"role": "assistant", "content": json.dumps({ "step": "analyse", "content": "The user is asking for the result of an arithmetic operation involving addition and multiplication."})},
        {"role":"assistant","content": json.dumps({"step": "think", "content": "I need to remember the order of operations, which dictates that multiplication should be performed before addition."})},
        {"role":"assistant","content": json.dumps({"step": "think", "content": "Let's break down the expression: First, calculate the multiplication part 4 * 5."})},
        {"role":"assistant","content": json.dumps({"step": "think", "content": "4 * 5 equals 20. Now we need to add 3 to this result."})}
        
    ]
)

print(result.choices[0].message.content)