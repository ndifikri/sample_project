import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

# Mengakses variabel dari file .env
QDRANT_URL = st.text_input("QDRANT_URL", type="password")
QDRANT_API_KEY = st.text_input("QDRANT_API_KEY", type="password")
OPENAI_API_KEY = st.text_input("OPENAI_API_KEY", type="password")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key = OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key = OPENAI_API_KEY)

collection_name = "small_recipes"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

@tool
def get_relevant_docs(question):
  """Use this tools for get relevant documents about recipes."""
  results = qdrant.similarity_search(
      question,
      k=5
  )
  return results

tools = [get_relevant_docs]

def chat_chef(question, history):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt= f'''You are a master of any recipes. Answer only question about recipes and use given tools for get recipes details.'''
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0

    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
        elif "token_usage" in message.response_metadata:
            # Fallback for older or different structures
            total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

    price = 17_000*(total_input_tokens*0.15 + total_output_tokens*0.6)/1_000_000
    response = {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens
    }
    return response

st.title("Chatbot Recipes Master")
st.image("header_img.png")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me recipes question"):
    messages_history = st.session_state.get("messages", [])[-20:]
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    # Display user message in chat message container
    with st.chat_message("Human"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("AI"):
        response = chat_chef(prompt, history)
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "AI", "content": answer})

    with st.expander("**History Chat:**"):
        st.code(history)

    with st.expander("**Usage Details:**"):
        st.code(f'input token : {response["total_input_tokens"]}\noutput token : {response["total_output_tokens"]}')