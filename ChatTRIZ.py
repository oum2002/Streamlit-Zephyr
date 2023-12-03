import streamlit as st
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


# Set up the language model using the Hugging Face Hub repository
repo_id = "mistralai/Mistral-7B-v0.1"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.3, "max_new_tokens": 2000})

# Set up the prompt template
template = """Question: {question}

Answer: After careful consideration..."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)


# Create the Streamlit app
def main():
    st.title("ChatTRIZ")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
                st.markdown(prompt)
        response = llm_chain.run(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()