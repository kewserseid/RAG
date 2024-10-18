
from prompt import SYSTEM_PROMPT,RETRIEVE_PROMPT
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import openai

load_dotenv()

api = os.getenv('OPENAI_KEY')

from prompt import RETRIEVE_PROMPT,SYSTEM_PROMPT
def get_result_from_llm_background(query,retrieved_content,model: str = "gpt-4o") -> str:
        """
        Generate an answer to a user question based on the provided content.
        """
        try:
            print("chat completion")
           
            client =  OpenAI(api_key=api)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user", 
                        "content":f'user asked {query} and similiar result filtered are {retrieved_content}'
                                f"{RETRIEVE_PROMPT}"
                    },
                ]
            )
            result =  response.choices[0].message.content
            return result
        except Exception as e:
            print(e)
            return ""
    