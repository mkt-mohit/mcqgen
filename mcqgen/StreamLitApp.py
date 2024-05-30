import os,json
import openai
import pandas as pd
import traceback
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_community.callbacks.manager import get_openai_callback
#from langchain.callbacks import get_openai_callback
import PyPDF2
from dotenv import load_dotenv
import logging
from datetime import datetime
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging
import streamlit as st
from src.mcqgenerator.MCQgenerator import generate_evaluate_chain


with open('C:\\Users\\kumarm\\mcqgen\mcqgen\\Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

    # creating a title for the app

    st.title("MCQ Creater app app with langchain")

    # create a form using st.form
    with st.form("user_inputs"):
        uploaded_file=st.file_uploader("Upload PDF or text file")
        mcq_count=st.number_input("No of MCQs",min_value=3,max_value=50)
        #subject
        subject=st.text_input("Insert subject",max_chars=50)
        #Quiz tone
        tone=st.text_input("Complexity level of questions",max_chars=20,placeholder="simple")
        
        #Add button
        button=st.form_submit_button("Cretae MCQs")

        #check
        if button and uploaded_file is not None and mcq_count and subject and tone:
            with st.spinner("loading......"):
                try:
                    text=read_file(uploaded_file)
                    with get_openai_callback() as cb:
                        response=generate_evaluate_chain(
                            {
                                "text": text,
                                "number": mcq_count,
                                "subject": subject,
                                "tone": tone,
                                "response_json": json.dumps(RESPONSE_JSON)
                            }

                    )
                except Exception as e:
                    traceback.print_exception(type(e),e,e.__traceback__)
                    st.error("Error")
                else:
                    print(f"Total tokens:{cb.total_tokens}")
                    if isinstance(response, dict):
                        quiz=response.get("quiz",None)
                        if quiz is not None:
                            table_data=get_table_data(quiz)
                            if table_data is not None:
                                df=pd.DataFrame(table_data)
                                df.index=df.index+1
                                st.table(df)

                                st.text_area(label="Review",value=response["review"])
                            else:
                                st.error("Error in the table data")
                    else:
                        st.write(response)
                





                            







