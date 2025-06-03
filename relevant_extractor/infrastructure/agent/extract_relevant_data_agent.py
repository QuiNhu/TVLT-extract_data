from typing import Text, Dict, List
from google import genai
from google.genai import types

from relevant_extractor.domain import (
    get_relevant_data_function_calling,
    get_relevant_data_prompt,
    rewrite_question_function_calling,
    rewrite_question_prompt
)
from relevant_extractor.configs import GEMINI_API_KEY


# Configure the client and tools
client = genai.Client(api_key=GEMINI_API_KEY)

def get_relevant_data_gemini(
        question: Text,
        articles: Text,
        prompt: Text = get_relevant_data_prompt,
        function_calling: Dict = get_relevant_data_function_calling
) -> Dict:
    context = f"## USER'S QUESTION:\n{question}\n\n---\n## LAW ARTICLES: {articles}"
    tools = types.Tool(function_declarations=[function_calling])
    config = types.GenerateContentConfig(
        temperature=0.3,
        top_k=2,
        max_output_tokens=4096,
        tools=[tools]
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{prompt} \n{context}",
        config=config,
    )
    print(f"{prompt} \n{context}")
    print(response)
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        return function_call.args
    else:
        return {
            "reasoning": response.text
        }


async def rewrite_question(
        question: Text,
        prompt: Text = rewrite_question_prompt,
        function_calling: Dict = rewrite_question_function_calling
) -> Dict:
    context = f"## Câu hỏi người dùng:{question}"
    tools = types.Tool(function_declarations=[function_calling])
    config = types.GenerateContentConfig(
        temperature=0.5,
        top_k=2,
        max_output_tokens=256,
        tools=[tools]
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{prompt} \n{context}",
        config=config,
    )
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        response = function_call.args
    else:
        response = {
            "rewrote_question": None,
            "reasoning": response.text
        }

    return response


# if __name__ == "__main__":
#     question = "Người sử dụng lao động được sa thải người lao động nữ đang mang thai không?"
#     articles = read_txt(txt_path="/Users/cerris/PycharmProjects/TVPL-test/dataset/45_2019_QH14_333670.txt")
#     relevant_articles = rewrite_question(question)
#     print(relevant_articles)