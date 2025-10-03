import streamlit as st
import pandas as pd
import numpy as np
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
import os
import glob
import re
from dotenv import load_dotenv
import getpass
import json
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from agent import Agent
from langchain_core.chat_history import InMemoryChatMessageHistory

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

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
    Assumptions (no <think> tags):
    - Thoughts are the text after the first '|' delimiter, if present.
    - Action appears as: ACTION: [BUY,SELL,HOLD] TICKER QUANTITY
    """
    raw_output = (raw_output or "").strip()

    # Extract thought via '|' delimiter
    thought_content = ""
    if '|' in raw_output:
        pre, post = raw_output.split('|', 1)
        thought_content = post.strip()
        message_part = pre
    else:
        message_part = raw_output

    # Parse ACTION from the full output for robustness
    action_match = re.search(r'ACTION:\s*(BUY|SELL|HOLD)\s*([A-Z\-\.]+)?\s*(\d+)?', raw_output, re.IGNORECASE)
    if action_match:
        action, ticker, quantity_str = action_match.groups()
        quantity = int(quantity_str) if quantity_str else 0
        action_tuple = (action.strip().upper(), (ticker or '').strip().upper() or None, quantity)
    else:
        action_tuple = ('HOLD', None, 0)

    # Message: remove any ACTION line from the message_part
    message = re.sub(r'ACTION:.*', '', message_part, flags=re.IGNORECASE).strip()

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


# --- Persistence (Save/Load Simulation) ---

SAVES_DIR = Path("saves")

def ensure_saves_dir():
    SAVES_DIR.mkdir(parents=True, exist_ok=True)

def simulation_to_dict() -> dict:
    """Serialize current simulation in session_state to a plain dict for JSON."""
    market = getattr(st.session_state, 'market', None)
    agents = getattr(st.session_state, 'agents', [])
    return {
        "version": 1,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stocks": list(STOCKS),
        "day": st.session_state.get('day', 0),
        "market": {
            "prices": getattr(market, 'prices', {}),
            "history": getattr(market, 'history', {}),
            "volatility": getattr(market, 'volatility', 0.2),
        },
        "bulletin_board": st.session_state.get('bulletin_board', []),
        "agent_thoughts": st.session_state.get('agent_thoughts', {}),
        "agents": [
            {
                "name": a.name,
                "system_message": getattr(a, 'system_message', ''),
                "cash": float(a.cash),
                "portfolio": {k: int(v) for k, v in a.portfolio.items()},
                "portfolio_value_history": [float(x) for x in a.portfolio_value_history],
            }
            for a in agents
        ],
    }

def save_simulation(filename: str) -> Path:
    ensure_saves_dir()
    # sanitize filename
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", filename.strip()) or f"sim_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    path = SAVES_DIR / f"{name}.json"
    data = simulation_to_dict()
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def list_saves() -> list[Path]:
    ensure_saves_dir()
    return sorted(SAVES_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)

def load_simulation(path: Path):
    global STOCKS
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    # Restore stocks constant (used throughout UI)
    if 'stocks' in data:
        STOCKS = data['stocks']
    # Rebuild market
    st.session_state.market = StockMarketSimulator(STOCKS)
    st.session_state.market.prices = data.get('market', {}).get('prices', {})
    st.session_state.market.history = data.get('market', {}).get('history', {})
    st.session_state.market.volatility = data.get('market', {}).get('volatility', 0.2)
    # Rebuild agents without triggering LLM calls
    st.session_state.agents = []
    for a in data.get('agents', []):
        agent = TradingAgent(name=a['name'], system_message=a.get('system_message', ''), initial_cash=a.get('cash', INITIAL_CASH))
        agent.cash = a.get('cash', agent.cash)
        agent.portfolio = {k: int(v) for k, v in a.get('portfolio', {}).items()}
        agent.portfolio_value_history = [float(x) for x in a.get('portfolio_value_history', [agent.cash])]
        st.session_state.agents.append(agent)
    # Restore other state
    st.session_state.day = int(data.get('day', 0))
    st.session_state.bulletin_board = list(data.get('bulletin_board', []))
    st.session_state.agent_thoughts = dict(data.get('agent_thoughts', {}))



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
    """Wrapper around our Agent to fit the previous interface for the UI."""

    def __init__(self, name: str, system_message: str, initial_cash: float):
        self.name = name
        self.system_message = system_message
        self.cash = initial_cash
        self.portfolio = {stock: 0 for stock in STOCKS}
        self.portfolio_value_history = [initial_cash]
        # Chat history per agent for LM prompting
        self.chat_history = InMemoryChatMessageHistory()
        # Underlying LLM agent
        self.agent = Agent(character=name.lower(), chat_history=self.chat_history, allowed_tickers=STOCKS)

    def get_portfolio_value(self, current_prices: dict):
        stock_value = sum(self.portfolio[stock] * current_prices[stock] for stock in self.portfolio)
        return self.cash + stock_value

    def chat_turn(self) -> str:
        self.agent.chatMode()
        return self.agent.run(None)

    def action_turn(self, stock_text: str) -> str:
        self.agent.actionMode()
        return self.agent.run(stock_text)


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

    st.subheader("Model Settings")
    st.caption("Using Google Gemini via langchain_google_genai for Agent prompts.")

    if st.button("🚀 Initialize/Reset Simulation"):
        st.session_state.market = StockMarketSimulator(STOCKS)
        st.session_state.agents = []
        st.session_state.bulletin_board = []
        st.session_state.agent_thoughts = {}
        st.session_state.day = 0

        agent_definitions = load_personas()
        for agent_def in agent_definitions:
            st.session_state.agents.append(
                TradingAgent(
                    name=agent_def["name"],
                    system_message=agent_def["persona"],
                    initial_cash=INITIAL_CASH,
                )
            )
            st.session_state.agent_thoughts[agent_def['name']] = []
        st.success("Simulation Initialized!")
        st.rerun()

    st.divider()
    st.subheader("💾 Save / Load")
    colA, colB = st.columns([1,1])
    with colA:
        save_name = st.text_input("Save name", value=f"sim_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}" )
        if st.button("Save Snapshot", use_container_width=True, type="secondary"):
            if 'market' in st.session_state and st.session_state.get('agents'):
                path = save_simulation(save_name)
                st.success(f"Saved to {path}")
            else:
                st.warning("Initialize and run the simulation before saving.")
    with colB:
        saves = list_saves()
        options = [s.name for s in saves]
        selected = st.selectbox("Available saves", options)
        if st.button("Load Snapshot", use_container_width=True):
            if selected:
                path = next((p for p in saves if p.name == selected), None)
                if path:
                    load_simulation(path)
                    st.success(f"Loaded {path.name}")
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

                # 2a. Chat phase
                for agent in st.session_state.agents:
                    chat_message = agent.chat_turn()
                    # Sanitize bulletin: ensure message starts with "<role>:" and remove any ACTION lines that leak into chat
                    if isinstance(chat_message, str):
                        chat_message = re.sub(r"^ACTION:.*$", "", chat_message, flags=re.IGNORECASE | re.MULTILINE).strip()
                    # Bulletin board should contain only the chat message text (agent message already includes role)
                    st.session_state.bulletin_board.append(f"**[Day {st.session_state.day}]** {chat_message}\n")

                # Aggregate chat history text for the day if needed
                board_text = "\n".join(st.session_state.bulletin_board[-len(st.session_state.agents):])

                # 2b. Action phase: provide prices as a text block
                price_lines = [f"{s}: ${p:.2f}" for s, p in current_prices.items()]
                stock_text = "\n".join([f"Day {st.session_state.day} Prices:"] + price_lines)

                for agent in st.session_state.agents:
                    action_response = agent.action_turn(stock_text)
                    thought, message, action = parse_model_output(action_response)

                    # Store only the thought under Agent Thoughts for this day
                    if thought:
                        st.session_state.agent_thoughts[agent.name].append(thought)
                    else:
                        # In case no explicit thought, keep the action_response sans ACTION as the thought fallback
                        st.session_state.agent_thoughts[agent.name].append(message)

                    # Do NOT add action-phase message to the bulletin board; keep it chat-only

                    # Execute trade
                    action_type, ticker, quantity = action
                    # Guard against invalid or out-of-universe tickers by mapping to HOLD
                    if ticker not in STOCKS:
                        action_type, ticker, quantity = ('HOLD', None, 0)

                    if action_type == 'BUY' and ticker in STOCKS and quantity > 0:
                        cost = current_prices[ticker] * quantity
                        if agent.cash >= cost:
                            agent.cash -= cost
                            agent.portfolio[ticker] += quantity
                    elif action_type == 'SELL' and ticker in STOCKS and quantity > 0:
                        if agent.portfolio.get(ticker, 0) >= quantity:
                            revenue = current_prices[ticker] * quantity
                            agent.cash += revenue
                            agent.portfolio[ticker] -= quantity

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

    chart_data = {
        "Day": pd.Series(range(st.session_state.day + 1))
    }

    st.subheader("Market & Agent Performance")

    # --- Market Prices ---
    prices_df = pd.DataFrame(st.session_state.market.history)
    prices_df.index.name = "Day"
    st.caption("📈 Market Prices")
    st.line_chart(prices_df)

    # --- Agent Portfolios ---
    portfolios_df = pd.DataFrame(
        {agent.name: pd.Series(agent.portfolio_value_history)
        for agent in st.session_state.agents}
    )
    portfolios_df.index.name = "Day"
    st.caption("🤖 Agent Portfolio Values")
    st.line_chart(portfolios_df)

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

