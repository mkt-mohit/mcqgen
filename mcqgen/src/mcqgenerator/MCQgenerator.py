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

load_dotenv()
key=os.getenv("OPENAI_API_KEY")

llm=ChatOpenAI(api_key=key,model="gpt-3.5-turbo",temperature=0.5)
print(llm)

TEMPLATE="""
Text={text}
Your are an expert MCQ maker. Given the above text, it is your job to \
create quiz of {number} multiple choice questions for  {subject} students in {tone} tone.
Make sure the questions are not repeated and check all questions to be confirming the text as well.
Make sure to format your response like RESPONSE_JSON and use it as guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt= PromptTemplate(

input_variables=["text","number","subject","tone","response_json"],
template=TEMPLATE

)

quiz_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz",verbose=True)

TEMPLATE2="""
You are an expert english grammarian and writer. Given a MUltiple choice quiz for {subject} students.\
you need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for xomplexity.\
if the quiz is not as per woth the cognitive and anlyatical abilities of the stidents,\
update the quiz questions whoch needs to be changed and change the tone such that it perfectly fits the students abilities
Quiz_MCQS:
{quiz}

check from an expert English Writer of the above quiz.
"""

quiz_evalutation_prompt=PromptTemplate(input_variables=["subject","quiz"],template=TEMPLATE2)

review_chain=LLMChain(llm=llm,prompt=quiz_evalutation_prompt,output_key="review",verbose=True)

generate_evaluate_chain=SequentialChain(chains=[quiz_chain,review_chain],input_variables=["text","number","subject","tone","response_json"],output_variables=["quiz","review"],verbose=True,)

