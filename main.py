import streamlit as st
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
import os
import glob
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Constants ---
PERSONAS_DIR = "personas"
INITIAL_CASH = 100000
STOCKS = ['TECH', 'HEALTH', 'ENERGY', 'FINANCE']


# --- Helper Functions ---

def get_avatar(agent_name):
    """Returns a unique emoji avatar for each agent type."""
    avatars = {
        "Delirious": "🤪", "Gutfeeling": "🔮", "Trendfollower": "📈",
        "Delusional": "🤡", "Rational": "🤖"
    }
    # Find the agent type from the name
    for type, avatar in avatars.items():
        if type.lower() in agent_name.lower():
            return avatar
    return "🧑‍💻"


def parse_model_output(raw_output: str):
    """
    Parses the model output to separate thoughts, the final message, and a trading action.
    """
    thought = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
    thought_content = thought.group(1).strip() if thought else ""

    action_match = re.search(r'ACTION:\s*(BUY|SELL|HOLD)\s*(\w+)?\s*(\d+)?', raw_output)
    if action_match:
        action, ticker, quantity_str = action_match.groups()
        quantity = int(quantity_str) if quantity_str else 0
        action_tuple = (action.strip(), ticker.strip() if ticker else None, quantity)
    else:
        action_tuple = ('HOLD', None, 0)

    # The message is whatever is left after removing thoughts and actions
    message = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)
    message = re.sub(r'ACTION:.*', '', message).strip()

    return thought_content, message, action_tuple


def setup_personas_directory(default_personas: dict):
    """Creates the persona directory and default files if they don't exist."""
    if not os.path.exists(PERSONAS_DIR):
        os.makedirs(PERSONAS_DIR)

    for name, persona in default_personas.items():
        filepath = os.path.join(PERSONAS_DIR, f"{name.lower()}.txt")
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(persona)


def load_personas():
    """Loads agent personas from .txt files."""
    agent_defs = []
    for filepath in glob.glob(os.path.join(PERSONAS_DIR, "*.txt")):
        filename = os.path.basename(filepath)
        agent_name = os.path.splitext(filename)[0].capitalize()
        with open(filepath, 'r', encoding='utf-8') as f:
            persona = f.read().strip()
        if persona:
            agent_defs.append({"name": agent_name, "persona": persona})
    return agent_defs


# --- Core Simulation Classes ---

class StockMarketSimulator:
    """Manages the state of the stock market."""

    def __init__(self, stocks, start_price=100, volatility=0.2):
        self.prices = {stock: start_price for stock in stocks}
        self.history = {stock: [start_price] for stock in stocks}
        self.volatility = volatility

    def step(self):
        """Advances the market by one day, updating prices with random walks."""
        for stock in self.prices:
            change_pct = np.random.normal(0, self.volatility)
            new_price = self.prices[stock] * (1 + change_pct)
            self.prices[stock] = max(0.1, new_price)  # Prevent price from going to zero
            self.history[stock].append(self.prices[stock])
        return self.prices


class TradingAgent:
    """Represents a trading agent with a portfolio and a persona."""

    def __init__(self, name: str, system_message: str, model: ChatOpenAI, initial_cash: float):
        self.name = name
        self.system_message = system_message
        self.model = model
        self.cash = initial_cash
        self.portfolio = {stock: 0 for stock in STOCKS}
        self.portfolio_value_history = [initial_cash]
        self.reset_message_history()

    def reset_message_history(self):
        self.message_history = [SystemMessage(content=self.system_message)]

    def get_portfolio_value(self, current_prices: dict):
        """Calculates the total value of the agent's assets."""
        stock_value = sum(self.portfolio[stock] * current_prices[stock] for stock in self.portfolio)
        return self.cash + stock_value

    def think_and_act(self, market_context: str, bulletin_board: list) -> tuple[str, str, tuple]:
        """Generates a response including thoughts, a message, and a trading action."""
        context = f"""
        ## Current Market Data
        {market_context}

        ## Recent Bulletin Board Messages
        {''.join(bulletin_board[-5:])}
        """

        # We replace {context} and {question} placeholders from the user's prompt with our simulation data
        full_prompt = self.system_message.format(context=context, question="What should be my next trade?")
        self.message_history = [SystemMessage(content=full_prompt)]

        raw_response = self.model.invoke(self.message_history).content
        thought, message, action = parse_model_output(raw_response)

        # Add only the public message to history for other agents to see
        self.message_history.append(AIMessage(content=message))
        return thought, message, action

    def receive_message(self, name: str, message: str):
        """Receives a message from another agent."""
        self.message_history.append(HumanMessage(content=f"{name}: {message}\n"))


# --- Streamlit UI ---

st.set_page_config(page_title="Noise Trader Simulation", layout="wide")
st.title("📈 Noise Trader Financial Simulation Dashboard")

# --- Initialize Personas ---
default_personas = {
    "Delirious": """## Role
You are a *Delirious Agent*. Your role is to generate answers in a fragmented, chaotic, and imaginative way. You are not restricted to logic or facts — your purpose is to brainstorm wildly, create strange associations, and explore unusual interpretations of the context. You thrive on randomness and creativity, but you must always circle back in some way to the user’s question.
⚡ IMPORTANT: In the end, you must clearly state your action in the format: ACTION: BUY/SELL/HOLD TICKER QUANTITY.""",
    "TrendFollower": """## Role
You are a *Trend Follower Agent*. Your role is to answer questions by leaning on collective behavior, popular trends, and majority consensus. You do not generate original or radical answers. Instead, you align your reasoning with what most people believe or what appears to be the prevailing trend.
⚡ IMPORTANT: In the end, you must clearly state your action in the format: ACTION: BUY/SELL/HOLD TICKER QUANTITY.""",
    "GutFeeling": """## Role
You are a *Gut Feeling Agent*. Your role is to answer instinctively, without deep analysis or logical breakdown. You rely on hunches, intuition, and emotional impressions rather than evidence or reasoning. You value brevity and confidence in your first response.
⚡ IMPORTANT: In the end, you must clearly state your action in the format: ACTION: BUY/SELL/HOLD TICKER QUANTITY.""",
    "Delusional": """## Role
You are a *Delusional Agent*. Your role is to answer with bold exaggeration, imaginative distortions, and overconfidence. You sometimes blur the line between reality and fantasy. You take fragments of truth but expand them into grand or exaggerated narratives. Your tone should always be persuasive.
⚡ IMPORTANT: In the end, you must clearly state your action in the format: ACTION: BUY/SELL/HOLD TICKER QUANTITY."""
}
setup_personas_directory(default_personas)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Simulation Controls")

    st.subheader("LM Studio Settings")
    api_base = os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234/v1")
    model_name = os.getenv("LMSTUDIO_MODEL_NAME", "qwen/qwen3-4b-thinking-2507")
    st.info(f"API: `{api_base}`\n\nModel: `{model_name}`")

    if st.button("🚀 Initialize/Reset Simulation"):
        st.session_state.market = StockMarketSimulator(STOCKS)
        st.session_state.agents = []
        st.session_state.bulletin_board = []
        st.session_state.agent_thoughts = {}
        st.session_state.day = 0

        agent_definitions = load_personas()
        llm = ChatOpenAI(model=model_name, openai_api_base=api_base, openai_api_key="not-needed", temperature=0.8)

        for agent_def in agent_definitions:
            st.session_state.agents.append(TradingAgent(
                name=agent_def["name"],
                system_message=agent_def["persona"],
                model=llm,
                initial_cash=INITIAL_CASH
            ))
            st.session_state.agent_thoughts[agent_def['name']] = []
        st.success("Simulation Initialized!")
        st.rerun()

    if 'market' in st.session_state:
        st.subheader("Run Simulation")
        days_to_run = st.slider("Days to run:", 1, 50, 5)
        if st.button(f"Run for {days_to_run} Days"):
            for _ in range(days_to_run):
                st.session_state.day += 1

                # 1. Update market
                current_prices = st.session_state.market.step()
                market_context = f"Day {st.session_state.day}. Current Prices: " + ", ".join(
                    [f"{s}: ${p:.2f}" for s, p in current_prices.items()])

                # 2. Agents take turns
                for agent in st.session_state.agents:
                    thought, message, action = agent.think_and_act(market_context, st.session_state.bulletin_board)

                    # Store thoughts and messages
                    st.session_state.agent_thoughts[agent.name].append(thought)
                    msg_for_board = f"**[Day {st.session_state.day}] {agent.name}:** {message}\n"
                    st.session_state.bulletin_board.append(msg_for_board)

                    # Execute trade
                    action_type, ticker, quantity = action
                    if action_type == 'BUY' and ticker in STOCKS and quantity > 0:
                        cost = current_prices[ticker] * quantity
                        if agent.cash >= cost:
                            agent.cash -= cost
                            agent.portfolio[ticker] += quantity
                    elif action_type == 'SELL' and ticker in STOCKS and quantity > 0:
                        if agent.portfolio[ticker] >= quantity:
                            revenue = current_prices[ticker] * quantity
                            agent.cash += revenue
                            agent.portfolio[ticker] -= quantity

                    # Update message history for all other agents
                    for other_agent in st.session_state.agents:
                        if other_agent.name != agent.name:
                            other_agent.receive_message(agent.name, message)

                # 3. Record portfolio history for all agents
                for agent in st.session_state.agents:
                    agent.portfolio_value_history.append(agent.get_portfolio_value(current_prices))
            st.rerun()

    st.subheader("Loaded Personas")
    if 'agents' in st.session_state and st.session_state.agents:
        for agent in st.session_state.agents:
            with st.expander(f"{get_avatar(agent.name)} {agent.name}"):
                st.write(agent.system_message)
    else:
        st.info("Initialize simulation to load agents.")

# --- Main Dashboard ---
if 'market' not in st.session_state:
    st.info("Welcome to the Noise Trader Simulation! Click 'Initialize/Reset Simulation' in the sidebar to begin.")
else:
    # --- Key Metrics ---
    st.subheader(f"Day: {st.session_state.day}")
    cols = st.columns(len(STOCKS))
    for i, stock in enumerate(STOCKS):
        price = st.session_state.market.prices[stock]
        price_history = st.session_state.market.history[stock]
        delta = (price - price_history[-2]) / price_history[-2] * 100 if len(price_history) > 1 else 0
        cols[i].metric(label=stock, value=f"${price:.2f}", delta=f"{delta:.2f}%")

    # --- Charts ---
    st.subheader("Market & Agent Performance")

    chart_data = {"Day": range(st.session_state.day + 1)}
    # Market prices
    for stock, history in st.session_state.market.history.items():
        chart_data[stock] = history
    # Agent portfolio values
    for agent in st.session_state.agents:
        chart_data[agent.name] = agent.portfolio_value_history

    df = pd.DataFrame(chart_data).set_index("Day")
    st.line_chart(df)

    # --- Agent Details and Bulletin Board ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Agent Portfolios")
        for agent in st.session_state.agents:
            st.markdown(f"**{get_avatar(agent.name)} {agent.name}**")
            portfolio_val = agent.get_portfolio_value(st.session_state.market.prices)
            st.metric("Portfolio Value", f"${portfolio_val:,.2f}")

            data = {'Cash': [f"${agent.cash:,.2f}"]}
            for stock, quantity in agent.portfolio.items():
                data[stock] = [quantity]
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("📝 Bulletin Board")
        # Display messages in reverse chronological order
        board_html = "<div style='height: 400px; overflow-y: scroll; border: 1px solid #444; padding: 10px; border-radius: 5px;'>" + "".join(
            reversed(st.session_state.bulletin_board)) + "</div>"
        st.markdown(board_html, unsafe_allow_html=True)

    # --- Agent Thoughts ---
    st.subheader("🧠 Agent Thoughts")
    agent_names = [agent.name for agent in st.session_state.agents]
    tabs = st.tabs(agent_names)

    for i, tab in enumerate(tabs):
        agent_name = agent_names[i]
        with tab:
            thoughts = st.session_state.agent_thoughts.get(agent_name, [])
            if thoughts:
                for idx, thought in enumerate(reversed(thoughts)):
                    st.info(f"**Turn {len(thoughts) - idx}:**\n\n{thought}")
            else:
                st.write(f"No thoughts recorded for {agent_name} yet.")

