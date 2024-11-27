import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2.filters import do_round
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sympy.polys.rootisolation import dup_root_upper_bound

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

income = pd.read_csv('Data/Income_Industry.csv', delimiter=',')
jobPrediction = pd.read_csv('Data/Occupation_Projections.csv', delimiter=',')
medianIncome = pd.read_csv('Data/Median_Income.csv', delimiter=',')
tuitionCost = pd.read_csv('Data/Tuition_Cost.csv', delimiter=',')
enrollment = pd.read_csv('Data/Enrollment.csv', delimiter=',')
graduates = pd.read_csv('Data/Graduates.csv', delimiter=',')

merged_data = pd.merge(enrollment, graduates, on=['Field of study'], how='inner')

years = [col.split('_')[0] for col in merged_data.columns if '_x' in col]

for year in years:
    enroll_col = f"{year}_x"
    grad_col = f"{year}_y"
    dropout_col = f"{year}_dropout"

    for col in merged_data.columns:
        if '_x' in col or '_y' in col:
            merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

    if enroll_col in merged_data.columns and grad_col in merged_data.columns:
        merged_data[dropout_col] = ((merged_data[enroll_col] - merged_data[grad_col]) / merged_data[enroll_col]) * 100

print(merged_data[['Field of study'] + [f"{year}_dropout" for year in years]])

