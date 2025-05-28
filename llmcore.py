from langchain_google_genai import GoogleGenerativeAI



def process_ai(input:str):
    """
    Process the input string using Google Generative AI.
    
    Args:
        input (str): The input string to process.
        
    Returns:
        str: The processed output from the AI.
    """
    # Initialize the Google Generative AI model
    llm = GoogleGenerativeAI(model="gemini-2.0-flash")
    
    # Invoke the model with the input string
    result = llm.invoke(input)
    
    return result
