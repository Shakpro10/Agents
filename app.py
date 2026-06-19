# imports relevant libraries
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

# Load the API keys into environment variables
load_dotenv(override=True)

# Function to send a push notification using Pushover
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

# Function to record user details and send a push notification
def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

# Function to record an unknown question and send a push notification
def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

# JSON schema for the record_user_details function
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

# JSON schema for the record_unknown_question function
record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

# Define the list of tools with their corresponding JSON schemas
tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


# Define the main class that will handle the chat interactions and tool calls
class ResumeAgent:
    # In the initializer, we set up the OpenAI client, read the LinkedIn 
    # profile from the PDF, and read the summary from a text file. 
    # This information will be used in the system prompt to provide context 
    # for the LLM when it generates responses.
    def __init__(self):
        self.openai = OpenAI(base_url = "https://integrate.api.nvidia.com/v1", 
                             api_key = os.getenv('NVIDIA_API_KEY_1'))
        self.name = "Shakiru Sikiru"
        reader = PdfReader("linkedin/Profile.pdf")  # Reading pdf from linkedIn profile
        self.linkedin = ""
        for page in reader.pages:   
            text = page.extract_text()  # Extracting text from each page
            if text:
                self.linkedin += text   # Appending text from each page to linkedin variable
        with open("linkedin/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    # This function takes in the tool calls made by the LLM, executes the 
    # corresponding tool functions with the provided arguments, and returns 
    # the results in the format expected by the LLM. It uses the globals() 
    # function to dynamically get the tool function by name and call it with 
    # the arguments, which allows for a more elegant and scalable way to 
    # handle tool calls without hardcoding each one in an IF statement. 
    # The results are formatted as a list of messages with the role "tool", 
    # the content as the JSON string of the tool result, and the tool_call_id 
    # to link it back to the original tool call from the LLM.  
    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    # This function constructs the system prompt that will be provided to 
    # the LLM at the start of the conversation.
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
                        particularly questions related to {self.name}'s career, background, skills and experience. \
                        Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
                        You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
                        Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
                        If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
                        If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    # This is the main chat function that takes in a user message and the 
    # conversation history, and generates a response using the LLM. 
    # It also handles tool calls if the LLM decides to call any tools. 
    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="openai/gpt-oss-120b", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                assistant_message = response.choices[0].message
                tool_calls = assistant_message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(assistant_message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

# The main block of the code creates an instance of the ResumeAgent class 
# and launches a Gradio chat interface that connects to the chat function 
# of the ResumeAgent. This allows users to interact with the agent through 
# a web interface, where they can ask questions and receive responses based 
# on the information provided in the system prompt, as well as trigger tool 
# calls when necessary.
if __name__ == "__main__":
    resume_agent = ResumeAgent()
    gr.ChatInterface(resume_agent.chat).launch()
    