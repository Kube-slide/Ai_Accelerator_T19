import math
from statistics import correlation

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2.filters import do_round
from scipy.cluster.vq import kmeans
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sympy.polys.rootisolation import dup_root_upper_bound
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

loader = WebBaseLoader("https://johnabbott.qc.ca/career-programs/computer-science-technology/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
              documents=split_documents,
              collection_name="rag-chroma",
              embedding=embeddings
              )

retriever = vectorstore.as_retriever()

template = """You are an school guidance counselor who's purpose is to help
high school students select a program they would enjoy to do in CEGEP later on.
Use the provided documentation and the user prompt to assign the top three most
relatable CEGEP programs that the user should consider applying to. Rank the majors from
most relatable (1) to least relatable (3). print out the generated response in the following format :
According to the information you've given me, here are the programs you should consider applying to
1. FIRST_MAJOR
2. SECOND_MAJOR
3. THIRD_MAJOR
                                                            
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("Using the url, what are some hobbies / interests one should have if they wish to pursue a career in computer science at john abbott college?")
print(result)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

income = pd.read_csv('Data/Income_Industry.csv', delimiter=',')
jobPrediction = pd.read_csv('Data/Occupation_Projections.csv', delimiter=',')
medianIncome = pd.read_csv('Data/Median_Income.csv', delimiter=',')
tuitionCost = pd.read_csv('Data/Tuition_Cost.csv', delimiter=',')
enrollment = pd.read_csv('Data/Enrollment.csv', delimiter=',')
graduates = pd.read_csv('Data/Graduates.csv', delimiter=',')

merged_data = pd.merge(enrollment, graduates, on=['Field of study'], how='left')
merged_data = merged_data.merge(medianIncome, on=['Field of study'], how='left')
merged_data['MedianIncome'] = pd.to_numeric(merged_data['MedianIncome'], errors='coerce')
medIncome = merged_data['MedianIncome']

years = [col.split('_')[0] for col in merged_data.columns if '_x' in col]

for col in merged_data.columns:
    if '_x' in col or '_y' in col:
        merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')



for year in years:
    enroll_col = f"{year}_x"
    grad_col = f"{year}_y"
    dropout_col = f"{year}_dropout"

    if enroll_col in merged_data.columns and grad_col in merged_data.columns:
        merged_data[dropout_col] = ((merged_data[enroll_col] - merged_data[grad_col]) / merged_data[enroll_col]) * 100

merged_data.fillna(0, inplace=True)

correlation = merged_data[['MedianIncome', '2020_dropout']].corr()

X = merged_data[['MedianIncome']].values
Y = merged_data[['2020_dropout']].values

model = linear_model.LinearRegression()
model.fit(X, Y)

grouped = merged_data.groupby('2020_dropout')['MedianIncome'].mean()



desired_difficulty = int(input("Enter your desired program difficulty level [1-5]: "))
desired_difficulty = desired_difficulty * 5 + 60
desired_median_income = int(input("Enter your desired median income [$]: "))

merged_data['similarity_score'] = np.sqrt(
    (merged_data['2020_dropout'] - desired_difficulty) ** 2 +
    (merged_data['MedianIncome'] - desired_median_income) ** 2)

top_matches = merged_data.sort_values(by='similarity_score').head(5)
