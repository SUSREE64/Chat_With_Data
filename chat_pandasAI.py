import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_ollama import OllamaLLM
import time

local_model = "llama3.2:latest"
data = None
@st.cache_resource
def get_llm():
    return OllamaLLM(model=local_model)

llm = get_llm()

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
# Use the below code to see if the GPU is available and Cuda tool kit, torch are working compatible to each other
# import torch
# print(torch.cuda.is_available())  # True indicates the GPU is there False indicate only CPU

def get_csv(file):
    config_dict = {"llm":llm,
                   "save_charts": True,
                   "save_charts_path": "exports/charts",
                   "custom_whitelisted_dependencies": ["scikit-learn", "plotly"]}
    df = pd.read_csv(file)
    df = SmartDataframe(df,config=config_dict)
    return df

def show_data(d = None):
    return d
# Main Application UI ##########
st.title("Data Analysis with PandasAI")
with st.sidebar:
    uploader_file = st.file_uploader("Upload a CSV file", type= ["csv"])
    if uploader_file is not None:
        try:
            data = get_csv(uploader_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            data = None

st.write("Data head sample")
if show_data(data) is not None:
    st.write(data.head(5))
else:
    st.write("No data file to show preview")

prompt = st.text_area("Enter your prompt:")
if st.button("Generate") and prompt:
    with st.spinner("Generating response..."):
        start_time = time.time()
        response = data.chat(prompt)
        st.write(response)
        # Append to chat history
        st.session_state['chat_history'].append({"user": prompt, "ollama": response})
        st.write("Time taken for response :", abs(start_time-time.time()), "sec")

else:
    st.write("please enter your query")

# Display chat history
st.write("#### Chat History")
st.write("Model Selected:", llm.model)
for chat in st.session_state['chat_history']:
    st.write(f"**User**: {chat['user']}")
    st.write(f"**Ollama**:")
    st.write(chat['ollama'])  # Use st.write() to render tables or rich output
    st.write("------")
