import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

income = pd.read_csv('Data/Income_Industry.csv', delimiter=',')
jobPrediction = pd.read_csv('Data/Occupation_Projections.csv', delimiter=',')
medianIncome = pd.read_csv('Data/Median_Income.csv', delimiter=',')
tuitionCost = pd.read_csv('Data/Tuition_Cost.csv', delimiter=',')
enrollment = pd.read_csv('Data/Enrollment.csv', delimiter=',')
graduates = pd.read_csv('Data/Graduates.csv', delimiter=',')