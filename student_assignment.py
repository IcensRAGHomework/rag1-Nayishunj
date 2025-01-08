import json
import traceback
import requests
import os
from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import tool
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import base64
from mimetypes import guess_type
from langchain_core.output_parsers import JsonOutputParser

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
store = {}
session_id = "holiday-history"
_llm = None

def get_llm():
    global _llm
    if not _llm:
        _llm = AzureChatOpenAI(
                model=gpt_config['model_name'],
                deployment_name=gpt_config['deployment_name'],
                openai_api_key=gpt_config['api_key'],
                openai_api_version=gpt_config['api_version'],
                azure_endpoint=gpt_config['api_base'],
                temperature=gpt_config['temperature']
        )
    return _llm

def generate_hw01(question):
    llm = get_llm()

    prompt_template = """
    你是台灣人，請回答台灣特定月份的紀念日有哪些，請按以下json格式呈現，且只保留一項(不要加```json):
     {
         "Result":
             {
                 "date": "2024-09-05",
                 "name": "测试日"
             }
     }
    """
    messages = [
        SystemMessage(content=prompt_template),
        HumanMessage(content=[
                {"type": "text", "text": question},
            ])
    ]

    response = llm.invoke(messages)
    res = str(response.content)
    print(res)
    return res

@tool
def get_memorial_days(month):
    """Check the memorial days in a month."""
    try:
        api_key = 'MICUoVyQcOugpnhPK4fnlDL5526qHJGV'
        url = "https://calendarific.com/api/v2/holidays?&api_key={}&country=TW&year=2024&month={}".format(api_key, month)
        resp = requests.get(url)
        print(resp.content.decode())
        return resp.content.decode()
    except Exception:
        return "fail to get memorial days for {}".format(month)
    
def generate_hw02(question):
    llm = get_llm()
    tools = [get_memorial_days, ]
    llm_with_tools = llm.bind_tools(tools)
    prompt_template = """
        你是一個厲害的助手, 但你不知道台灣的紀念日。請用中文並按以下json格式呈現結果(不要加```json):
        {
            "Result": [
                {
                    "date": "2024-10-10",
                    "name": "國慶日"
                },
                {
                    "date": "2024-10-09",
                    "name": "重陽節"
                },
                {
                    "date": "2024-10-21",
                    "name": "華僑節"
                },
                {
                    "date": "2024-10-25",
                    "name": "台灣光復節"
                },
                {
                    "date": "2024-10-31",
                    "name": "萬聖節"
                }
            ]
        }
        """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=prompt_template),
        ("user", """{input}"""),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    response = agent_with_chat_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    res = str(response['output'])
    print(res)
    return res

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def generate_hw03(question2, question3):
    generate_hw02(question2)
    llm = get_llm()
    prompt_template = """
    你是一個厲害的助手, 請按以下json格式呈現(不要加```json):
    - add : 這是一個布林值，表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false。
    - reason : 描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。
     {
         "Result":
             {
                 "add": true,
                 "reason": "蔣中正誕辰紀念日並未包含在十月的節日清單中。目前十月的現有節日包括國慶日、重陽節、華僑節、台灣光復節和萬聖節。因此，如果該日被認定為節日，應該將其新增至清單中。"
             }
     }
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=prompt_template),
        ("user", """{input}"""),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True, return_intermediate_steps=True)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    response = agent_with_chat_history.invoke(
            {'input': question3},
            config={"configurable": {"session_id": session_id}},
        )
    res = str(response['output'])
    print(res)
    return res

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def generate_hw04(question):
    image_path = 'baseball.png'
    data_url = local_image_to_data_url(image_path)
    prompt_template = """ 你是一個厲害的助手, 請按以下json格式呈現(不要加```json):
     {
         "Result":
             {
                 "score": 5498
             }
     }
    """
    messages=[
        { "role": "system", "content": prompt_template },
        { "role": "user", "content": [
            {
                "type": "text",
                "text": question
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            }
        ] }
    ]

    llm = get_llm()

    parser = JsonOutputParser()
    response = llm.invoke(messages)
    #res = parser.invoke(str(response.content))
    res = str(response.content)
    print(res)
    return res
