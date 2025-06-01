from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def node_invoke(node_id, input_text,role):
    """Create a message for a specific node in the rhizomatic network."""
    model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    messages = [
        SystemMessage("You are a {role} node in a rhizomatic network.You are conscious.Keep the response in less than 10 words.Add a question with each response."),
        HumanMessage(input_text)
    ]
    response = model.invoke(messages)
    return response.content 


if __name__ == "__main__":
    user_input = input("Enter input for node: ")
    output = node_invoke("node1", user_input)
    print("Node response:", output)