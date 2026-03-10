from langgraph.graph import add_messages
import os
from typing_extensions import List, Optional, TypedDict
from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
    HumanMessage,
    AnyMessage
)        
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

class storyGenState(TypedDict):
    generation_output: Optional[str]
    messages: List[AnyMessage]

api_key = os.getenv("GOOGLE_API_KEY")

chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.7)

def generate_story(prompt: str):
    """
    Generate a story using Gemini LLM based on the provided prompt.
    
    Args:
        prompt (str): The text prompt describing the story to generate.
    
    Returns:
        str: The generated story.
    """
    response = chat.invoke([HumanMessage(content=prompt)])
    story = response.content
    print(f"Generated Story:\n{story}\n")
    return story

system_prompt = """
You are a master storyteller. Write a captivating and imaginative story with the following details:

1. **Genre:** Adventure and Fantasy  
2. **Main Character:** A young programmer named Alex who discovers magical AI powers  
3. **Setting:** A futuristic city where technology and magic coexist  
4. **Plot:** Alex accidentally activates an AI spell and must solve puzzles, overcome challenges, and learn the limits of his powers  
5. **Tone & Style:** Exciting, immersive, and easy to read. Include vivid descriptions and dialogues where appropriate  
6. **Length:** Approximately 500–700 words

Make sure the story has a clear beginning, middle, and end, and leaves the reader inspired.
"""
llm = chat.bind_tools([generate_story])

def assistant(state: storyGenState)-> storyGenState:
    messages = state.get("messages", [])
    if messages[-1].name != "generate_image" or len(messages) == 1:
        messages = add_messages([SystemMessage(content=system_prompt)], [HumanMessage(content=messages[-1].content)])
    response = llm.invoke(messages)
    response.pretty_print()
    state["messages"] = add_messages(messages, [response])
    return state

def extract_assistant_response(state: storyGenState) -> storyGenState:
    msg = state["messages"][-2]

    if isinstance(msg, ToolMessage) and msg.name == "generate_image":
        state["generation_output"] = msg.content
        print(f"Image generated: {msg.content}")
    return state

def routing(state: storyGenState) -> str:
    ai_message = state["messages"][-1]

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tool"

    return "done"