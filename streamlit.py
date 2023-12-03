#pip install langchain
#pip install huggingface_hub
#pip install streamlit 

import streamlit as st
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


# Set up the language model using the Hugging Face Hub repository
repo_id = "HuggingFaceH4/zephyr-7b-beta"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.3, "max_new_tokens": 2000})

# Set up the prompt template
template = """
You are an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's question
In the context of the TRIZ methodology,  find in this text the two contradictory parameters from the 39 parameters:
1	Weight of moving object
2	Weight of stationary object
3	Length of moving object
4	Length of stationary object
5	Area of moving object
6	Area of stationary object
7	Volume of  moving object
8	Volume of stationary object
9	Speed
10	Force (Intensity)
11	Stress or pressure
12	Shape
13	Stability of  the object's composition
14	Strength
15	Duration of action of moving object
16	Duration of action by stationary object
17	Temperature
18	Illumination intensity
19	Use of energy by moving object
20	Use of energy by stationary object
21	Power
22	Loss of Energy
23	Loss of substance
24	Loss of Information
25	Loss of Time
26	Quantity of substance/the matter
27	Reliability
28	Measurement accuracy
29	Manufacturing precision
30	Object-affected harmful factors
31	Object-generated harmful factors
32	Ease of manufacture
33	Ease of operation
34	Ease of repair
35	Adaptability or versatility
36	Device complexity
37	Difficulty of detecting and measuring
38	Extent of automation
39	Productivity
Question: {question}\n\nAnswer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Create the Streamlit app
def main():
    st.title("ChatSST")

    # Get user input
    question = st.text_input("Enter your question")

    # Generate the response
    if st.button("Get Answer"):
        with st.spinner("Generating Answer..."):
            response = llm_chain.run(question)
        st.success(response)

if __name__ == "__main__":
    main()