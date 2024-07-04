# RAG Chatbot Streamlit

The app simplifies interaction with a language model via a user-friendly web interface. Powered by retrieval augmented generation (RAG), it ensures responses are relevant and accurate. For testing, it utilizes a dataset containing technology news. The project includes two files: an ipynb file with setup instructions for Pinecone and OpenAI, and app.py, the Streamlit web application code.

## Usage

To start the web interface locally, run the following command from the terminal:

```bash
streamlit run app.py
```

Alternatively, access the deployed application [here](https://rag-chatbot-app-jzcyawtbugtqhkvexawpho.streamlit.app/). User needs to specify valid API keys for both OpenAI and Pinecone.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/tipharez-allmighty/rag-chatbot-streamlit.git
cd rag-chatbot-streamlit
```

2. Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Alternatively, you can run the provided Jupyter Notebook to install dependencies.
