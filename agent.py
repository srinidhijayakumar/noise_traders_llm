from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


class Agent:
    def __init__(self, character=None): # character must be name of the .txt file that has the prompt
        self.character = character
        if character.lower() == "rational" or character.lower() == "scientist":
            self.temperature = 0.0
        else:
            self.temperature = 0.7

        self.llm = llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash",
                            temperature=self.temperature,
                            max_tokens=None,
                            timeout=None,
                            max_retries=2,
                            # other params...
                        )
        self.prompt_messages = dict()
        self.chat_history = InMemoryChatMessageHistory()
        self.mode = None
        self.llm_chain = None


    def initialize_prompts(self):
        file = open("personas/" + self.character + ".txt")
        character = (
                "system",
                file.read()
            )
        file.close()

        self.prompt_messages["action"] = ChatPromptTemplate.from_messages(
        [
            character,
            (
                "system",
                "You will now respond in the following format: In the beginning, you must clearly state your action in \
                the format: ACTION: BUY/SELL/HOLD TICKER QUANTITY. It must be followed by your reasoning as described.\
                Provide your reasoning at the end, after a delimiter symbol |"
            ),
            (
                "human",
                "Here are the stock prices for today {stock_prices}. Decide which stock to buy and provide reasoning."
            )
        ]
        )

        self.prompt_messages["chat"] = ChatPromptTemplate.from_messages(
            [
                character,
                (
                    "system",
                    f"You are now part of a multi-agent chat. You will provide responses as directed. You will converse \
                     with other agents solely on the topic of the stocks mentioned, and which one you choose to buy.\
                     Your response should begin with your role, which is {self.character}, like so: `{self.character}: [Rest of the message]`."
                ),
                (
                    "agents",
                    "{chat_messages}" # chat history here
                )
            ]
        )

        def actionMode(self):
            self.mode = "action"
            self.llm_chain = self.prompt_messages["action"] | self.llm

        def action(self, stock_prices):
            if self.mode != "action":
                self.actionMode()
            return self.llm_chain.invoke({
                "stock_prices": stock_prices,
            })

        def chatMode(self):
            self.mode = "chat"
            self.llm_chain = self.prompt_messages["chat"] | self.llm