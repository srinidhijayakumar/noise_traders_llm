from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import threading
from time import monotonic, sleep


class Agent:
    # Class-wide rate limiter so all Agent instances share the same throttle
    _rate_lock = threading.Lock()
    _last_call_ts: float = 0.0
    _min_interval: float = 60.0 / float(os.getenv("GEMINI_RPM", "15"))  # seconds between calls

    def __init__(self, character=None, chat_history=None, allowed_tickers=None): # character must be name of the .txt file that has the prompt
        self.character = character
        self.chat_history = chat_history
        # Allowed tickers list for this simulation (used in prompts)
        self.allowed_tickers = allowed_tickers or ["TECH", "HEALTH", "ENERGY", "FINANCE"]
        if character.lower() == "rational" or character.lower() == "scientist":
            self.temperature = 0.0
        else:
            self.temperature = 0.7

        self.llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash-lite",
                            temperature=self.temperature,
                            max_tokens=None,
                            timeout=None,
                            max_retries=2,
                            convert_system_message_to_human=True,
                            # other params...
                        )
        self.prompt_messages = dict()
        self.initialize_prompts()
        self.chat_history = chat_history
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
                    "You must choose an action ONLY for one of these tickers: {allowed_tickers}. \n"
                    "Rules:\n"
                    "- Your FIRST LINE must be exactly: ACTION: BUY|SELL|HOLD <TICKER> <QUANTITY> (choose <TICKER> strictly from {allowed_tickers}).\n"
                    "- QUANTITY must be a positive integer (0 is only allowed with HOLD).\n"
                    "- Do NOT mention or use any other tickers than {allowed_tickers}.\n"
                    "- After the action line, include a single '|' character followed by a short reasoning paragraph.\n"
                    "- Do NOT repeat or quote the prompt, persona, or chat history in your output."
                ),
                (
                    "system",
                    "Chat so far (may be empty):\n{chat_history}"
                ),
                (
                    "human",
                    "Here are the stock prices for today: {stock_prices}. Decide which stock to buy and provide reasoning following the exact format."
                ),
            ]
        )

        self.prompt_messages["chat"] = ChatPromptTemplate.from_messages(
            [
                character,
                (
                    "system",
                    f"You are now part of a multi-agent chat. Stay strictly on the topic of which ONE of the allowed tickers you are buying: {{allowed_tickers}}.\n"
                    f"Start your message with `{self.character}:` and then your argument.\n"
                    f"Do NOT include an ACTION line in chat. Do NOT mention or use any tickers outside of {{allowed_tickers}}.\n"
                    f"If there is no prior chat, you may start by stating which one of {{allowed_tickers}} you favor and why."
                ),
                (
                    "system",
                    "Chat history so far (if any):\n{chat_messages}"
                ),
                (
                    "human",
                    f"Respond now as {self.character}. Keep it on-topic about the allowed tickers {{allowed_tickers}} and your choice. Start with `{self.character}:` and one concise message."
                ),
            ]
        )

    def actionMode(self):
        self.mode = "action"
        self.llm_chain = self.prompt_messages["action"] | self.llm

    def run(self, stock_prices):
        if self.mode == "action":
            self._throttle()
            return self.llm_chain.invoke({
            "stock_prices": stock_prices,
            "chat_history": self._history_text(),
            "allowed_tickers": ", ".join(self.allowed_tickers),
            }).content
        elif self.mode == "chat":
            self._throttle()
            chat_msg =  self.llm_chain.invoke({
            "chat_messages": self._history_text(),
            "allowed_tickers": ", ".join(self.allowed_tickers),
            }).content
            # best-effort record of AI message into history if provided
            try:
                if isinstance(self.chat_history, InMemoryChatMessageHistory):
                    self.chat_history.add_ai_message(chat_msg)
            except Exception:
                pass
            return chat_msg
        else:
            raise NotImplementedError("Chat mode not implemented yet.")

    def chatMode(self):
        self.mode = "chat"
        self.llm_chain = self.prompt_messages["chat"] | self.llm

    def _history_text(self) -> str:
        """Return chat history as a simple string for prompt variables."""
        if not self.chat_history:
            return ""
        try:
            # Prefer synchronous API if available
            if hasattr(self.chat_history, "get_messages"):
                msgs = self.chat_history.get_messages()
                lines = []
                for m in msgs:
                    role = getattr(m, "type", m.__class__.__name__)
                    content = getattr(m, "content", str(m))
                    # Skip any seeded persona/system content if present
                    if isinstance(content, str) and content.strip().lower().startswith("persona:"):
                        continue
                    lines.append(f"{role}: {content}")
                return "\n".join(lines)
        except Exception:
            pass
        # Fallback: string representation
        try:
            return str(self.chat_history)
        except Exception:
            return ""

    @classmethod
    def _throttle(cls):
        """Ensure a minimum interval between Gemini API calls across all agents."""
        with cls._rate_lock:
            now = monotonic()
            wait = cls._min_interval - (now - cls._last_call_ts)
            if wait > 0:
                sleep(wait)
                now = monotonic()
            cls._last_call_ts = now

    