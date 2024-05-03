RAG Chatbot Streamlit

RAG Chatbot Streamlit is a Python application that offers a web-based interface for users to interact with a language model through queries and receive responses. The application utilizes retrieval augmented generation (RAG) to enhance the relevance and accuracy of responses by leveraging both generative and retrieval-based methods.
Features

    Web Interface: Accessible through a user-friendly web interface, allowing easy input of queries and viewing of responses.
    Retrieval Augmented Generation: Integrates retrieval-based techniques with generative language models to provide more relevant and accurate responses.
    OpenAI Integration: Utilizes the OpenAI API for language model inference, enabling powerful natural language understanding and generation capabilities.
    Pinecone Integration: Leverages Pinecone as a vector database for efficient retrieval of relevant information based on user queries.
    Customization: Easily customizable to incorporate additional data sources or fine-tune model parameters.

Usage

To start the web interface locally, run the following command from the terminal:

bash

streamlit run app.py

Alternatively, access the deployed application here, ensuring you have specified valid API keys for both OpenAI and Pinecone.
Installation

    Clone this repository:

bash

git clone https://github.com/tipharez-allmighty/rag-chatbot-streamlit.git
cd rag-chatbot-streamlit

    Install the required packages from requirements.txt:

bash

pip install -r requirements.txt

Alternatively, you can run the provided Jupyter Notebook to install dependencies.
