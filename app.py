# Importing necessary modules
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Function to construct the prompt for the RAG model
def get_prompt(instruction, examples, new_system_prompt):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  SYSTEM_PROMPT + instruction  + "\n" + examples
    return prompt_template


B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# System prompt defining the behavior of the assistant
sys_prompt = """\
You are a helpful, respectful and honest assistant designed to assist with. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# Template for the prompt including context and examples
instruction = """CONTEXT:/n/n {context}/n"""
examples = """
Q: {question}
A: """
template = get_prompt(instruction, examples, sys_prompt)


st.title('RAG test bot')

# Sidebar to input API keys
with st.sidebar:
    openai_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
    pinecone_key = st.text_input("Pinecone API Key", key="pinecone_api_key", type="password")

# If both API keys are provided, initialize the vector store and the RAG model
if pinecone_key and openai_key:
    vector_store = PineconeVectorStore(index_name='rag-db',
                                  embedding=OpenAIEmbeddings(
                                  openai_api_key=openai_key,
                                  model='text-embedding-3-small'),
                                  pinecone_api_key=pinecone_key,
                                  namespace="knowledge_base") 

    template = get_prompt(instruction, examples, sys_prompt) 

    # Prompt template for the RAG model
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    # Initialize the OpenAI Chat model
    llm = ChatOpenAI(model_name='gpt-4', max_tokens=488,
                    temperature=0,
                    model_kwargs={"stop": ["\nQ:", "\nA:"]},api_key=openai_key)

    # Initialize the RetrievalQA model
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
else:
    st.warning('Please add your API keys.')

# Initialize or retrieve chat messages from session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Input for user messages
if prompt:= st.chat_input('Whats up?'):
    # Check if both API keys are provided before proceeding
    if not openai_key or not pinecone_key:
       st.stop()

    # Add user message to session state
    st.session_state.messages.append({'role': 'user','content': prompt})   
    st.chat_message('user').write(prompt)       

    # Generate response using the RAG model
    response = qa_chain.invoke(prompt)['result']
    # Add assistant response to session state
    st.session_state.messages.append({'role': 'assistant','content': response})
    st.chat_message('assistant').write(response) 
