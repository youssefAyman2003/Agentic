from langchain_core.messages import HumanMessage
from Agent import storyGenState, assistant, extract_assistant_response, routing, generate_story
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

graph = StateGraph(storyGenState)
graph.add_node("assistant", assistant)
graph.add_node("extract_assistant_response", extract_assistant_response)
graph.add_node("generate_story", ToolNode([generate_story]))

graph.add_edge(START, "assistant")
graph.add_edge("assistant", "extract_assistant_response")
graph.add_conditional_edges("extract_assistant_response",
                            routing,
                            {
                               "tool": "generate_story",
                               "done": END
                            })
graph.add_edge("generate_story", "assistant")

agent = graph.compile()

# Generate and save the graph as PNG
#png_data = agent.get_graph(xray=None).draw_mermaid_png()

#with open("generation.png", "wb") as f:
    #f.write(png_data)

#print("Graph saved as generation.png")







user_input = input("User: ")
messages = []
while user_input.lower() != "exit":
    messages.append(HumanMessage(content=user_input))
    state = storyGenState(messages=messages)
    state = agent.invoke(state)
    if state.get("generation_output"):
        print(f"The story generated: {state.get('generation_output')}")
    
    messages = state.get("messages", [])
    user_input = input("User: ")