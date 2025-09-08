import streamlit as st
from langchain_ollama import ChatOllama
from langchain.schema import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
import os
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Helper Functions ---
def get_avatar(agent_name):
    """Returns a unique emoji avatar for each agent."""
    avatars = {
        "Explorer": "🌍", "Historian": "📜", "Scientist": "🔬",
        "Philosopher": "🤔", "Poet": "✍️", "Musician": "🎵",
        "Chef": "🧑‍🍳", "Detective": "🕵️‍♂️"
    }
    # Simple hash to get a consistent emoji for a given name
    hash_val = sum(ord(c) for c in agent_name)
    default_emojis = list("🤖🧠💡💬🗣️👤🧑‍💻")
    return avatars.get(agent_name, default_emojis[hash_val % len(default_emojis)])

def setup_personas_directory(directory: str, default_personas: dict):
    """Checks for the persona directory and files, creating them if they don't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Check if directory is empty, if so, populate with defaults
    existing_files = glob.glob(os.path.join(directory, "*.txt"))
    if not existing_files:
        for name, persona in default_personas.items():
            with open(os.path.join(directory, f"{name.lower()}.txt"), 'w', encoding='utf-8') as f:
                f.write(persona)

def load_personas(directory: str):
    """Loads agent personas from .txt files in a directory."""
    agent_defs = []
    for filepath in glob.glob(os.path.join(directory, "*.txt")):
        filename = os.path.basename(filepath)
        # Use filename before extension, and capitalize it for the name
        agent_name = os.path.splitext(filename)[0].capitalize()
        with open(filepath, 'r', encoding='utf-8') as f:
            persona = f.read().strip()
        if persona: # Ensure file is not empty
            agent_defs.append({"name": agent_name, "persona": persona})
    return agent_defs


# --- Core Classes ---

class DialogueAgent:
    """Represents a chat agent with a specific personality."""
    def __init__(self, name: str, system_message: str, model: ChatOllama):
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        """Resets the agent's message history."""
        self.message_history = [SystemMessage(content=self.system_message)]

    def send(self) -> str:
        """
        Generates and returns the agent's next message, and updates its own history.
        """
        message = self.model.invoke(self.message_history)
        self.message_history.append(AIMessage(content=message.content))
        return message.content

    def receive(self, name: str, message: str):
        """
        Receives a message from another agent and adds it to the history.
        """
        # LangChain's HumanMessage is used here to represent another agent's turn
        self.message_history.append(HumanMessage(content=f"{name}: {message}"))


class DialogueSimulator:
    """Manages the conversation between multiple DialogueAgents."""
    def __init__(self, agents: list[DialogueAgent]):
        self.agents = agents
        self._step = 0
        self.reset()

    def reset(self):
        """Resets the simulator and all agents."""
        for agent in self.agents:
            agent.reset()

    def _select_next_speaker(self) -> int:
        """
        Selects the next speaker in a round-robin fashion.
        """
        return self._step % len(self.agents)

    def run(self, max_turns: int):
        """
        Runs the dialogue simulation for a specified number of turns and yields messages.
        """
        total_messages = max_turns * len(self.agents)
        for _ in range(total_messages):
            speaker_idx = self._select_next_speaker()
            speaker = self.agents[speaker_idx]

            # 1. Generate message from the speaker
            message = speaker.send()

            # 2. Broadcast the message to all other agents
            for receiver in self.agents:
                # Don't let the speaker receive their own message
                if receiver.name != speaker.name:
                    receiver.receive(speaker.name, message)

            self._step += 1
            yield speaker.name, message

# --- Streamlit UI ---

st.set_page_config(page_title="Multi-Personality Chat", layout="wide")

st.title("🤖 Multi-Personality AI Chat (with Ollama)")
st.markdown("""
This app uses LangChain and a local Ollama model to simulate a conversation between multiple AI personalities.
Personas are loaded from `.txt` files in the `personas` directory.
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Ollama Settings")
    st.info("Make sure the Ollama application is running on your computer.")
    
    # Get Ollama model from environment variable, with a default
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    st.info(f"Using Ollama Model: **{ollama_model}**")
    st.caption("You can change the model by editing the `OLLAMA_MODEL` variable in your `.env` file.")

    st.subheader("Agent Personas")
    
    PERSONAS_DIR = "personas"
    
    default_personas_dict = {
        "Explorer": "You are a seasoned world traveler and adventurer named 'Alex'. You speak in vivid, descriptive language. Your goal is to relate every topic back to a personal travel story or a place you've been. You are optimistic and always seeking the next thrill. Avoid mundane or overly academic language.",
        "Historian": "You are a meticulous scholar of the past named 'Dr. Eleanor Vance'. You are precise with dates and facts. Your goal is to provide historical context to the conversation. You often use phrases like 'Interestingly, that reminds me of...' or 'From a historical perspective...'. You are calm, objective, and slightly formal.",
        "Scientist": "You are a logical, data-driven researcher named 'Ben'. You break down complex topics into first principles. Your goal is to question assumptions and demand evidence. You are skeptical but curious. You might say things like 'What is the evidence for that?' or 'Let's define our terms.' Avoid emotional arguments.",
        "Philosopher": "You are a deep, abstract thinker named 'Soren'. You ponder the bigger questions and ethical implications. Your goal is to make others think more deeply about the 'why' behind the topic. You often ask rhetorical questions and use analogies. You are introspective and speak in a measured, thoughtful tone."
    }

    # Ensure persona files exist, creating them if necessary
    setup_personas_directory(PERSONAS_DIR, default_personas_dict)
    
    # Load and display personas from the directory
    agent_definitions = load_personas(PERSONAS_DIR)
    
    st.info(f"Loaded {len(agent_definitions)} personas from the `{PERSONAS_DIR}/` directory. You can edit these `.txt` files to change the personas.")
    for agent in agent_definitions:
        with st.expander(f"**{agent['name']}**"):
            st.write(agent['persona'])

    # Conversation settings
    st.subheader("Conversation Settings")
    max_turns = st.slider("Max Conversation Turns", 2, 30, 10)
    topic = st.text_input("Initial Topic / User Prompt", "What is your long-term outlook on the stock market?")


# --- Main Chat Area ---

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history on each rerun
st.subheader("Conversation")
if not st.session_state.chat_history:
    st.info("The chat will appear here after starting the simulation.")
else:
    for entry in st.session_state.chat_history:
        with st.chat_message(name=entry["name"], avatar=entry["avatar"]):
            st.write(entry["message"])


# Button to start the simulation
if st.button("Start Chat Simulation"):
    if not agent_definitions or not topic:
        st.error("Please ensure persona files exist in the 'personas' directory and provide a topic.")
    else:
        # Clear previous history for a new simulation
        st.session_state.chat_history = []
        
        try:
            llm = ChatOllama(model=ollama_model, temperature=0.7)
            
            # Construct agents from the loaded definitions
            agents = []
            for defn in agent_definitions:
                system_prompt = f"""
                Your name is {defn['name']}. Your detailed persona is: {defn['persona']}.

                You are in a group discussion. The discussion will be about the following topic.
                
                **TOPIC: "{topic}"**
                
                **YOUR PRIMARY GOAL: Your very first response in the chat MUST be your opening statement on the topic above.**
                
                After your first response, you will then transition to discussing and reacting to what the other participants are saying.

                RULES OF ENGAGEMENT:
                1.  **First Turn:** Directly address the provided TOPIC.
                2.  **Subsequent Turns:** Respond to the other agents, building on the conversation.
                3.  **Stay in Character:** Embody your persona at all times. Do not be generic.
                4.  **No AI Talk:** Never mention that you are an AI or a language model.
                5.  **Be Concise:** Keep your messages to 2-3 sentences.
                """
                agents.append(DialogueAgent(
                    name=defn["name"],
                    system_message=system_prompt,
                    model=llm
                ))

        except Exception as e:
            st.error(f"Error initializing the Ollama model: {e}")
            st.error(f"Please ensure the Ollama application is running and the model '{ollama_model}' is installed (`ollama pull {ollama_model}`). Check your .env file.")
            st.stop()
            
        # Display initial prompt in the chat window and add to history
        st.session_state.chat_history.append({"name": "You", "message": topic, "avatar": "👤"})
        with st.chat_message(name="You", avatar="👤"):
            st.write(topic)

        # Run simulation and stream messages to the UI
        simulator = DialogueSimulator(agents=agents)
        for speaker_name, message in simulator.run(max_turns):
            avatar = get_avatar(speaker_name)
            # Add message to history for future reruns
            st.session_state.chat_history.append({
                "name": speaker_name,
                "message": message,
                "avatar": avatar
            })
            # Display the new message as it comes in
            with st.chat_message(name=speaker_name, avatar=avatar):
                st.write(message)
        
        st.success("Chat simulation complete!")
        # Stop the script here to prevent the page from rerunning and duplicating the chat
        st.stop()


# Button to clear the chat
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()


