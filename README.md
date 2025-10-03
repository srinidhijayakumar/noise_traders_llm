# Setup
Create virtual environment(optional):
```sh
python3 -m venv venv
```
Install packages:
```sh
pip install -r requirements.txt
```
Run Ollama server :
```sh
brew install ollama
ollama serve
ollama pull gemma3:1b
```
Run the app:
```sh
streamlit run main.py
```

