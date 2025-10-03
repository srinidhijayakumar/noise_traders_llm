# Setup
Create virtual environment(optional):
```sh
python3 -m venv venv
```
Install packages:
```sh
pip install -r requirements.txt
```
Set up Google API Key:
Put it in the root in a .env file or enter manually while running the app.
```
sh
# in .env file
GOOGLE_API_KEY='you_api_key'
```
Run the app:
```sh
streamlit run main.py
```
