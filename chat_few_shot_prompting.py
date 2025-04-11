from dotenv import load_dotenv
from openai import OpenAI
# import os
# os.kill(os.getpid(), 9)
load_dotenv()

client = OpenAI()

"""
System prompt kya karta ke voh ek inital context set karta hai 
like you only solve math questions
"""

system_prompt = """
You are an AI Assistant who is specialized in maths.
You should not answer any query that is not related to maths.

For a given query help user to solve that along with explanation.

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2.

Input : 3 * 10
Output: 3 * 3 is 30 which is calculated by multiplying 3 by 10 , Funfact you can even multiply vica versa.

Input:  Why is sky  blue?
Output: Bruh? You alright? Is it maths query?
"""

result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        # {"role":"user", "content":"Hey There"}
        {"role":"system", "content": system_prompt},
        {"role":"user", "content":"what is chaicode?"}
    ]
)
print(result.choices[0].message.content)