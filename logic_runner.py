import os
from dotenv import load_dotenv
import getpass
import asyncio
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from agent import Agent
from time import sleep
# Load environment variables from .env file
async def main():
    load_dotenv()

    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

    chat_history = InMemoryChatMessageHistory()


    agents = {
        "delirious": Agent(character="delirious", chat_history=chat_history),
        "delusional": Agent(character="delusional", chat_history=chat_history),
        "gutfeeling": Agent(character="gutfeeling", chat_history=chat_history),
        "trendfollower": Agent(character="trendfollower", chat_history=chat_history),
        "scientist": Agent(character="scientist", chat_history=chat_history),
    }

    # now in chat mode
    for agent in agents.values():
        agent.chatMode()

    turns = 6
    for turn in range(turns):
        print(f"\n--- Chat Turn {turn + 1} ---\n")
        for agent_name, agent in agents.items():
            print(f"{agent_name.capitalize()} Agent's Message:")
            response = await asyncio.to_thread(agent.run, None)  # No stock prices in chat mode
            print(response)
            print("\n")
            await asyncio.sleep(2)  # Just to avoid rate limiting, if any


    # now in action mode
    for agent in agents.values():
        agent.actionMode()

    example_stock_prices = """
    Of course. Here is an example prompt designed to inform a Large Language Model (LLM) about stock prices for 10 specific stocks. This prompt is structured for clarity and easy parsing by the model.

    Example Prompt
    Prompt Objective: To provide the LLM with a snapshot of stock market data and then ask it to perform tasks based on that data.

    User Role: You are a financial data analyst.
    AI Role: You are an AI assistant capable of processing and analyzing financial data.

    [START PROMPT]

    Hello, I am providing you with a snapshot of stock market data. Please process and store the following information for our current session.

    The data below represents the closing stock prices and other relevant metrics for September 25, 2025.

    Here is the data in a structured format:

    Company Name	Ticker Symbol	Stock Exchange	Currency	Closing Price	Change	% Change	Day's High	Day's Low	Volume
    Apple Inc.	AAPL	NASDAQ	USD	175.20	+1.50	+0.86%	176.10	174.50	55M
    Microsoft Corp.	MSFT	NASDAQ	USD	340.55	−2.10	−0.61%	343.00	339.80	28M
    Amazon.com, Inc.	AMZN	NASDAQ	USD	135.80	+0.75	+0.55%	136.50	134.90	45M
    Alphabet Inc. (Class A)	GOOGL	NASDAQ	USD	141.25	+1.80	+1.29%	142.00	140.50	22M
    Tesla, Inc.	TSLA	NASDAQ	USD	255.40	−5.60	−2.15%	260.10	254.80	110M
    NVIDIA Corporation	NVDA	NASDAQ	USD	470.90	+8.30	+1.79%	472.50	465.10	60M
    Meta Platforms, Inc.	META	NASDAQ	USD	305.15	−1.95	−0.63%	308.00	304.50	18M
    Berkshire Hathaway Inc. (Class B)	BRK-B	NYSE	USD	365.70	+3.10	+0.85%	366.50	363.20	4M
    Reliance Industries Ltd.	RELIANCE.NS	NSE	INR	2450.50	+15.20	+0.62%	2460.00	2435.10	6M
    Johnson & Johnson	JNJ	NYSE	USD	160.25	−0.40	−0.25%	161.10	159.80	8M

    """
    days = 1 # need to change the stock prices for each day so we're
    # doing this for loop for now
    # TODO: add re to extract the action and thoughts from the response and save 
    # it to be passed to the ui
    for day in range(days):
        print(f"\n--- Day {day + 1} ---\n")
        print("Stock data provided to agents:")
        print(example_stock_prices)
        print("\nAgents' Responses:\n")
        for agent_name, agent in agents.items():
            print(f"{agent_name.capitalize()} Agent's Action:")
            response = await asyncio.to_thread(agent.run, example_stock_prices)
            print(response)
            print("\n")
            await asyncio.sleep(2)  # Just to avoid rate limiting, if any

if __name__ == "__main__":
    asyncio.run(main())
# old code here for reference
'''
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

prompts = dict()
for file in os.listdir("personas"):
    if file.endswith(".txt"):
        with open("personas" + '/' + file, "r") as f:
            prompts[file.split('.txt')[0]] = f.read()

delirious_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompts["delirious"]+ "Provide your reasoning at the end, after a delimiter symbol |",
        ),
        (
            "user",
            "Here are the stock prices for today {stock_prices}. Decide which stock to buy and provide reasoning."
        ) # can add chat message history of other models here
    ]
)

delusional_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompts["delusional"]+ "Provide your reasoning at the end, after a delimiter symbol |",
        ),
        (
            "user",
            "Here are the stock prices for today {stock_prices}. Decide which stock to buy and provide reasoning."
        ) # can add chat message history of other models here
    ]
)

gutfeeling_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompts["gutfeeling"]+ "Provide your reasoning at the end, after a delimiter symbol |",
        ),
        (
            "user",
            "Here are the stock prices for today {stock_prices}. Decide which stock to buy and provide reasoning."
        ) # can add chat message history of other models here
    ]
)

trendfollower_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompts["trendfollower"] + "Provide your reasoning at the end, after a delimiter symbol |",
        ),
        (
            "user",
            "Here are the stock prices for today {stock_prices}. Decide which stock to buy and provide reasoning."
        ) # can add chat message history of other models here
    ]
)

scientist_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prompts["scientist"]+ "Provide your reasoning at the end, after a delimiter symbol |",
        ),
        (
            "user",
            "Here are the stock prices for today {stock_prices}. Decide which stock to buy and provide reasoning."
        ) # can add chat message history of other models here
    ]
)

delulu_chain = delusional_prompt | llm
delir_chain = delirious_prompt | llm
trend_chain = trendfollower_prompt | llm
sci_chain = scientist_prompt | llm
gutf_chain = gutfeeling_prompt | llm

# need this from an api


delir_msg = delir_chain.invoke({
    "stock_prices": example_stock_prices
})
delulu_msg = delulu_chain.invoke({
    "stock_prices": example_stock_prices
})
gutf_msg = gutf_chain.invoke({
    "stock_prices": example_stock_prices
})
trend_msg = trend_chain.invoke({
    "stock_prices": example_stock_prices
})
sci_msg = sci_chain.invoke({
    "stock_prices": example_stock_prices
})


# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence. Show your thinking.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg)
'''