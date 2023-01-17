# -*- coding: utf-8 -*-
"""
@author: user - Sanket Jadhav
"""
 
# Importing Libraries
import numpy as np 
from scipy import stats 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from chart_studio import plotly

# Standard plotly imports
import chart_studio.plotly as py
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks
import cufflinks as cf
import plotly.figure_factory as ff

import os

# Importing the auxiliar and preprocessing librarys 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split, KFold, cross_validate

# Models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomTreesEmbedding

def binary_ploting_distributions(df, cat_col):
    from plotly import tools
    
    fig = tools.make_subplots(rows=1,
                              cols=2,
                              print_grid=True,
                              horizontal_spacing=0.15, 
                              subplot_titles=("Distribution of and % Churn", 
                                              f'Mean Monthly Charges of {cat_col}') 
                             )

    tmp_churn = df[df['Churn'] == 1]
    tmp_no_churn = df[df['Churn'] == 0]
    tmp_attr = round(tmp_churn[cat_col].value_counts().sort_index() / df_train[cat_col].value_counts().sort_index(),2)*100

    trace1 = go.Bar(
        x=tmp_churn[cat_col].value_counts().sort_index().index,
        y=tmp_churn[cat_col].value_counts().sort_index().values,
        name='Yes_Churn',opacity = 0.8, marker=dict(
            color='seagreen',
            line=dict(color='#000000',width=1)))

    trace2 = go.Bar(
        x=tmp_no_churn[cat_col].value_counts().sort_index().index,
        y=tmp_no_churn[cat_col].value_counts().sort_index().values,
        name='No_Churn', opacity = 0.8, 
        marker=dict(
            color='indianred',
            line=dict(color='#000000',
                      width=1)
        )
    )

    trace3 =  go.Scatter(   
        x=tmp_attr.sort_index().index,
        y=tmp_attr.sort_index().values,
        yaxis = 'y2',
        name='% Churn', opacity = 0.6, 
        marker=dict(
            color='black',
            line=dict(color='#000000',
                      width=2 )
        )
    )

    df_tmp = (df_train.groupby(['Churn', cat_col])['MonthlyCharges'].mean().reset_index())

    tmp_churn = df_tmp[df_tmp['Churn'] == 1]
    tmp_no_churn = df_tmp[df_tmp['Churn'] == 0]

    df_tmp = (df_train.groupby(['Churn', cat_col])['MonthlyCharges'].mean()).unstack('Churn').reset_index()
    df_tmp['diff_rate'] = round((df_tmp[1] / df_tmp[0]) - 1,2) * 100

    trace4 = go.Bar(
        x=tmp_churn[cat_col],
        y=tmp_churn['MonthlyCharges'], showlegend=False,
        name='Mean Charge Churn',opacity = 0.8, marker=dict(
            color='seagreen',
            line=dict(color='#000000',width=1)))

    trace5 = go.Bar(
        x=tmp_no_churn[cat_col],
        y=tmp_no_churn['MonthlyCharges'],showlegend=False,
        name='Mean Charge NoChurn', opacity = 0.8, 
        marker=dict(
            color='indianred',
            line=dict(color='#000000',
                      width=1)
        )
    )

    trace6 =  go.Scatter(   
        x=df_tmp[cat_col],
        y=df_tmp['diff_rate'],
        yaxis = 'y2',
        name='% Diff Churn', opacity = 0.6, 
        marker=dict(
            color='black',
            line=dict(color='#000000',
                      width=5 )
        )
    )

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1) 
    fig.append_trace(trace3, 1, 1)
    fig.append_trace(trace4, 1, 2)
    fig.append_trace(trace5, 1, 2)
    fig.append_trace(trace6, 1, 2) 

    fig['data'][2].update(yaxis='y3')
    fig['data'][5].update(yaxis='y4')

    fig['layout']['xaxis'].update(autorange=True,
                                   tickfont=dict(size= 10), 
                                   title= f'{cat_col}', 
                                   type= 'category',
                                  )
    fig['layout']['yaxis'].update(title= 'Count')

    fig['layout']['xaxis2'].update(autorange=True,
                                   tickfont=dict(size= 10), 
                                   title= f'{cat_col}', 
                                   type= 'category',
                                  )
    fig['layout']['yaxis2'].update( title= 'Mean Monthly Charges' )

    fig['layout']['yaxis3']=dict(range= [0, 100], #right y-axis in subplot (1,1)
                              overlaying= 'y', 
                              anchor= 'x', 
                              side= 'right', 
                              showgrid= False, 
                              title= '%Churn Ratio'
                             )

# Insert a new key, yaxis4, and the associated value:
    fig['layout']['yaxis4']=dict(range= [-20, 100], #right y-axis in the subplot (1,2)
                              overlaying= 'y2', 
                              anchor= 'x2', 
                              side= 'right', 
                              showgrid= False, 
                              title= 'Monhtly % Difference'
                             )
    fig['layout']['title'] = f"{cat_col} Distributions"
    fig['layout']['height'] = 500
    fig['layout']['width'] = 1000

    iplot(fig)
    
def plot_dist_churn(df, col, binary=None):
    tmp_churn = df[df[binary] == 1]
    tmp_no_churn = df[df[binary] == 0]
    tmp_attr = round(tmp_churn[col].value_counts().sort_index() / df[col].value_counts().sort_index(),2)*100
    print(f'Distribution of {col}: ')
    trace1 = go.Bar(
        x=tmp_churn[col].value_counts().sort_index().index,
        y=tmp_churn[col].value_counts().sort_index().values,
        name='Yes_Churn',opacity = 0.8, marker=dict(
            color='seagreen',
            line=dict(color='#000000',width=1)))

    trace2 = go.Bar(
        x=tmp_no_churn[col].value_counts().sort_index().index,
        y=tmp_no_churn[col].value_counts().sort_index().values,
        name='No_Churn', opacity = 0.8, 
        marker=dict(
            color='indianred',
            line=dict(color='#000000',
                      width=1)
        )
    )

    trace3 =  go.Scatter(   
        x=tmp_attr.sort_index().index,
        y=tmp_attr.sort_index().values,
        yaxis = 'y2',
        name='% Churn', opacity = 0.6, 
        marker=dict(
            color='black',
            line=dict(color='#000000',
                      width=2 )
        )
    )
    
    layout = dict(title =  f'Distribution of {str(col)} feature by Target - With Churn Rates',
              xaxis=dict(), 
              yaxis=dict(title= 'Count'), 
              yaxis2=dict(range= [0, 100], 
                          overlaying= 'y', 
                          anchor= 'x', 
                          side= 'right',
                          zeroline=False,
                          showgrid= False, 
                          title= 'Percentual Churn Ratio'
                         ))

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    iplot(fig)
    
    
def plot_distribution(df, var_select=None, bins=1.0): 
# Calculate the correlation coefficient between the new variable and the target
    tmp_churn = df[df['Churn'] == 1]
    tmp_no_churn = df[df['Churn'] == 0]    
    corr = df_train['Churn'].corr(df_train[var_select])
    corr = np.round(corr,3)
    tmp1 = tmp_churn[var_select].dropna()
    tmp2 = tmp_no_churn[var_select].dropna()
    hist_data = [tmp1, tmp2]
    
    group_labels = ['Yes_churn', 'No_churn']
    colors = ['seagreen','indianred', ]

    fig = ff.create_distplot(hist_data,
                             group_labels,
                             colors = colors, 
                             show_hist = True,
                             curve_type='kde', 
                             bin_size = bins
                            )
    
    fig['layout'].update(title = var_select+' '+'(corr target ='+ str(corr)+')')

    iplot(fig, filename = 'Density plot')
    
def monthly_charges(df, col, binary=None):
# (df_train.groupby(['Churn', 'tenure'])['MonthlyCharges'].mean()).unstack('Churn').reset_index()
    df_tmp = (df_train.groupby([binary, col])['MonthlyCharges'].mean().reset_index())
    
    tmp_churn = df_tmp[df_tmp['Churn'] == 1]
    tmp_no_churn = df_tmp[df_tmp['Churn'] == 0]

    df_tmp = (df_train.groupby([binary, col])['MonthlyCharges'].mean()).unstack('Churn').reset_index()
    df_tmp['diff_rate'] = round((df_tmp[1] / df_tmp[0]) - 1,2) * 100
    
    trace1 = go.Bar(
        x=tmp_churn[col],
        y=tmp_churn['MonthlyCharges'],
        name='Mean Charge\nChurn',opacity = 0.8, marker=dict(
            color='seagreen',
            line=dict(color='#000000',width=1)))

    trace2 = go.Bar(
        x=tmp_no_churn[col],
        y=tmp_no_churn['MonthlyCharges'],
        name='Mean Charge No Churn', opacity = 0.8, 
        marker=dict(
            color='indianred',
            line=dict(color='#000000',
                      width=1)
        )
    )
    
    trace3 =  go.Scatter(   
        x=df_tmp[col],
        y=df_tmp['diff_rate'],
        yaxis = 'y2',
        name='% Diff Churn', opacity = 0.6, 
        marker=dict(
            color='black',
            line=dict(color='#000000',
                      width=5 )
        )
    )
        
    layout = dict(title =  f'Mean Monthly Charges of {str(col)} feature by Churn or Not Churn Customers - With Churn Ratio',
              xaxis=dict(), 
              yaxis=dict(title= 'Mean Monthly Charges'), 
              yaxis2=dict(range= [0, 100], 
                          overlaying= 'y', 
                          anchor= 'x', 
                          side= 'right',
                          zeroline=False,
                          showgrid= False, 
                          title= '% diff Monthly Charges Mean'
                         ))

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    iplot(fig)

# Importing the dataset
df_train =  pd.read_csv(r'D:\07-SANKET\DATA SCIENCE\0.A - DS Projects\A - CapStone Projects\Customer Churn\WA_Fn-UseC_-Telco-Customer-Churn.csv')

df_train.shape

df_train

cat_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PaperlessBilling', 'PhoneService', 'Contract', 'StreamingMovies',
                'StreamingTV', 'TechSupport', 'OnlineBackup', 'OnlineSecurity',
                'InternetService', 'MultipleLines', 'DeviceProtection', 'PaymentMethod']

# Understanding the Churn Distribution
print("CUSTOMERS %CHURN:")
print(round(df_train['Churn'].value_counts(normalize=True) * 100,2))

trace0 = go.Bar(
    x=df_train.groupby('Churn')['customerID'].count().index,
    y=df_train.groupby('Churn')['customerID'].count().values,
    marker=dict(
        color=['indianred', 'seagreen']),
               )
data = [trace0]
layout = go.Layout(
    title='Churn (Target) Distribution', 
    xaxis=dict(
        title='Customer Churn?'),
    yaxis=dict(
        title='Count')
              )
fig = go.Figure(data=data, layout=layout)
iplot(fig)

# Monthly Charges Distribution
df_train['TotalCharges'].fillna(df_train['MonthlyCharges'], inplace=True)
df_train['Churn'] = df_train.Churn.replace({'Yes': 1, 'No': 0})
print(f"The mininum value in Monthly Charges is {df_train['MonthlyCharges'].min()} and the maximum is {df_train['MonthlyCharges'].max()}")
print(f"The mean Monthly Charges of Churn Customers is {round(df_train[df_train['Churn'] != 0]['MonthlyCharges'].mean(),2)}\
      \nThe mean Monthly Charges of Non-churn Customers is {round(df_train[df_train['Churn'] == 0]['MonthlyCharges'].mean(),2)}")
plot_distribution(df_train, 'MonthlyCharges', bins=4.0)

# Ploting all categorical features
# The inspiration of this view is a Kernel that I saw in Vincent Lugat Kernel 
# I did some modifications but you can see the original on IBM 
for col in cat_features:
    binary_ploting_distributions(df_train, col) 
    
# Understanding the distribution of Total services provided for each Customer and the Churn % Rate
df_train['internet']= np.where(df_train.InternetService != 'No', 'Yes', 'No')
df_train['num_services'] = (df_train[['PhoneService', 'OnlineSecurity',
                                      'OnlineBackup', 'DeviceProtection', 
                                      'TechSupport', 'StreamingTV', 
                                      'StreamingMovies', 'internet']] == 'Yes').sum(axis=1)
    
# Based on Num Services
def countplot(x, hue, **kwargs):
    sns.countplot(x=x, hue=hue, **kwargs, order=['Month-to-month', 'One year', 'Two year'])

print("TOTAL NUMBER OF SERVICES BY CONTRACT AND CHURN")
grid = sns.FacetGrid(data=df_train,col='num_services', col_wrap=2,
                     aspect=1.9, height=3, sharey=False, sharex=False)
fig = grid.map(countplot,'Contract','Churn', palette=['indianred', 'seagreen'] )
fig.set_titles('Customer Total Services: {col_name}', fontsize=15)
fig.add_legend()
plt.show()

# Knowning the Numerical Features
df_train.loc[df_train['TotalCharges'] == ' ', 'TotalCharges'] = np.nan
df_train['TotalCharges'] = df_train['TotalCharges'].astype(float)

#Total of the Monthly Revenue Lose
print("Total Amount of Monthly Charges by each group: ")
print(round(df_train.groupby('Churn')['MonthlyCharges'].sum() ))

trace0 = go.Bar(
    x=round(df_train.groupby('Churn')['MonthlyCharges'].sum() \
      / df_train.groupby('Churn')['MonthlyCharges'].sum().sum() * 100).index, 
    y=round(df_train.groupby('Churn')['MonthlyCharges'].sum() \
      / df_train.groupby('Churn')['MonthlyCharges'].sum().sum() * 100).values,
    marker=dict(
        color=['indianred', 'seagreen']),
)

data = [trace0]
layout = go.Layout(
    title='Monthly Revenue % Lost by Churn Customer or not', 
    xaxis=dict(
        title='Customer Churn?', type='category'), 
    yaxis=dict(
        title='% of Total Monthly Revenue')
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

# Distribution of Total Charges
df_train['TotalCharges_log'] = np.log(df_train['TotalCharges']+1)
print(f"The mininum value in Total Charges is {df_train['TotalCharges'].min()} and the maximum is {df_train['TotalCharges'].max()}")
print(f"The mean Total Charges of Churn Customers is {round(df_train[df_train['Churn'] != 0]['TotalCharges'].mean(),2)}\
      \nThe mean Total Charges of Non-churn Customers is {round(df_train[df_train['Churn'] == 0]['TotalCharges'].mean(),2)}")

plot_distribution(df_train, 'TotalCharges_log', bins=.25)

# Tenure feature
print(f"The mininum value in Tenure is {df_train['tenure'].min()} and the maximum is {df_train['tenure'].max()}")
print(f"The mean Tenure of Churn Customers is {round(df_train[df_train['Churn'] != 0]['tenure'].mean())}\
      \nThe mean Tenure of Non-churn Customers is {round(df_train[df_train['Churn'] == 0]['tenure'].mean())}")

plot_dist_churn(df_train, 'tenure', 'Churn')

# Mean Monthly Charges by tenure with Churn Rate of tenure values
print("MEAN MONTHLY CHARGES OF TENURE FOR CHURN OR NO CHURN CUSTOMERS")
    
monthly_charges(df_train, 'tenure', 'Churn')

# The Average Monthly Charges by Total Number of Services Contracted
monthly_charges(df_train, 'num_services', 'Churn')

# Knowing Tenure by Total Charges for each Target value
tmp_churn = df_train[df_train['Churn'] == 1]
tmp_no_churn = df_train[df_train['Churn'] == 0]

tmp_churn_fiber = tmp_churn[tmp_churn['InternetService'] == 'Fiber optic']
tmp_churn_dsl = tmp_churn[tmp_churn['InternetService'] == 'DSL']
tmp_churn_no = tmp_churn[tmp_churn['InternetService'] == 'No']

tmp_no_churn_fiber = tmp_no_churn[tmp_no_churn['InternetService'] == 'Fiber optic']
tmp_no_churn_dsl = tmp_no_churn[tmp_no_churn['InternetService'] == 'DSL']
tmp_no_churn_no = tmp_no_churn[tmp_no_churn['InternetService'] == 'No']

# Create traces
trace0 = go.Scatter(
    x = tmp_churn_fiber['tenure'],
    y = tmp_churn_fiber['MonthlyCharges'],
    mode = 'markers', opacity=.6,
    name = 'Churn - Fiber', marker=dict(
        color='indianred', symbol='star'
))
trace1 = go.Scatter(
    x = tmp_churn_dsl['tenure'],
    y = tmp_churn_dsl['MonthlyCharges'],
    mode = 'markers', opacity=.6,
    name = 'Churn - DSL', marker=dict(
        color='indianred', symbol='square'
))
trace2 = go.Scatter(
    x = tmp_churn_no['tenure'],
    y = tmp_churn_no['MonthlyCharges'],
    mode = 'markers', opacity=.6,
    name = 'Churn - No', marker=dict(
        color='indianred', symbol='circle'
))

# Create traces
trace3 = go.Scatter(
    x = tmp_no_churn_fiber['tenure'],
    y = tmp_no_churn_fiber['MonthlyCharges'],
    mode = 'markers', opacity=.6,
    name = 'No-Churn-Fiber', marker=dict(
        color='seagreen', symbol='star'
))
trace4 = go.Scatter(
    x = tmp_no_churn_dsl['tenure'],
    y = tmp_no_churn_dsl['MonthlyCharges'],
    mode = 'markers', opacity=.6,
    name = 'No-Churn-DSL', marker=dict(
        color='seagreen', symbol='square'
))
trace5 = go.Scatter(
    x = tmp_no_churn_no['tenure'],
    y = tmp_no_churn_no['MonthlyCharges'],
    mode = 'markers', opacity=.6,
    name = 'No-Churn-No', marker=dict(
        color='seagreen', symbol='circle'
))

layout = dict(title ='Dispersion of Total Charges explained by Monthly Charges by Target',
              xaxis=dict(title='Internet Service Types'), 
              yaxis=dict(title= 'Monthly Charges'))

fig = go.Figure(data = [trace0, trace3, trace1, trace4, trace2, trace5], layout=layout)
iplot(fig)

df_train['assign_months'] = round(df_train['TotalCharges'] / df_train['MonthlyCharges'],0)

print("Comparing Tenure and Assign Months")
pd.concat([df_train['assign_months'].describe().reset_index(),
           df_train['tenure'].describe().reset_index()['tenure']], axis=1)

df_train.drop('assign_months', axis=1, inplace=True)

# Feature engineering and preprocessing
Id_col     = ['customerID']

target_col = ["Churn"]

cat_cols   = df_train.nunique()[df_train.nunique() < 10].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
binary_cols   = df_train.nunique()[df_train.nunique() == 2].keys().tolist()

multi_cols = [i for i in cat_cols if i not in binary_cols]
# Feature Engineering
df_train.loc[:,'Engaged'] = np.where(df_train['Contract'] != 'Month-to-month', 1,0)
df_train.loc[:,'YandNotE'] = np.where((df_train['SeniorCitizen']==0) & (df_train['Engaged']==0), 1,0)
df_train.loc[:,'ElectCheck'] = np.where((df_train['PaymentMethod'] == 'Electronic check') & (df_train['Engaged']==0), 1,0)
df_train.loc[:,'fiberopt'] = np.where((df_train['InternetService'] != 'Fiber optic'), 1,0)
df_train.loc[:,'StreamNoInt'] = np.where((df_train['StreamingTV'] != 'No internet service'), 1,0)
df_train.loc[:,'NoProt'] = np.where((df_train['OnlineBackup'] != 'No') |\
                                    (df_train['DeviceProtection'] != 'No') |\
                                    (df_train['TechSupport'] != 'No'), 1,0)

df_train['TotalServices'] = (df_train[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Creating new numerical columns
multi_cols.remove('Contract')
df_train['monthly_diff_mean'] = df_train['MonthlyCharges'] / df_train['MonthlyCharges'].mean() 
for cat in cat_cols:
    df_train[str(cat)+'_diff_mean'] = df_train['MonthlyCharges'] / df_train.groupby(['Contract',cat])['MonthlyCharges'].transform('mean')
    df_train[str(cat)+'_diff_std'] = df_train['MonthlyCharges'] / df_train.groupby(['Contract',cat])['MonthlyCharges'].transform('std')

# Encoding and binarizing features
le = LabelEncoder()
for cols in binary_cols :
    df_train[cols] = le.fit_transform(df_train[cols])
    
#Duplicating columns for multi value columns
df_train = pd.get_dummies(data = df_train,columns = multi_cols )

df_train.drop("Contract", axis=1, inplace=True)
num_cols   = [x for x in df_train.columns if x not in cat_cols + target_col + Id_col]

from sklearn.preprocessing import StandardScaler
df_train.fillna(-99, inplace=True)

#Scaling Numerical columns
ss = StandardScaler()
scl = ss.fit_transform(df_train[num_cols])
scl = pd.DataFrame(scl, columns=num_cols)

#dropping original values merging scaled values for numerical columns
# df_data_og = df_train.copy()

df_train = df_train.drop(columns = num_cols,axis = 1)
df_train = df_train.merge(scl, left_index=True, right_index=True, how = "left")
# Feature Selection
# Threshold for removing correlated variables
threshold = 0.90

# Absolute value correlation matrix
corr_matrix = df_train.corr().abs()

# Getting the upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))
print(list(to_drop))

df_train = df_train.drop(columns = to_drop)
print('Training shape: ', df_train.shape)

# Preprocessing - Seting X and y
from sklearn.model_selection import train_test_split
X_train = df_train.drop(['Churn', 'customerID'], axis=1)
y_train = df_train['Churn']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.20)

# Classifier models pipeline
clfs = []
seed = 3

clfs.append(("LogReg", 
             Pipeline([("Scaler", StandardScaler()),
                       ("LogReg", LogisticRegression())])))

clfs.append(("XGBClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("XGB", XGBClassifier())]))) 
clfs.append(("KNN", 
             Pipeline([("Scaler", StandardScaler()),
                       ("KNN", KNeighborsClassifier())]))) 

clfs.append(("DecisionTreeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("DecisionTrees", DecisionTreeClassifier())]))) 

clfs.append(("RandomForestClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RandomForest", RandomForestClassifier())]))) 

clfs.append(("GradientBoostingClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("GradientBoosting", GradientBoostingClassifier(max_features=15, 
                                                                       n_estimators=1000))]))) 

clfs.append(("RidgeClassifier", 
             Pipeline([("Scaler", StandardScaler()),
                       ("RidgeClassifier", RidgeClassifier())])))

clfs.append(("BaggingRidgeClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("BaggingClassifier", BaggingClassifier())])))

clfs.append(("ExtraTreesClassifier",
             Pipeline([("Scaler", StandardScaler()),
                       ("ExtraTrees", ExtraTreesClassifier())])))

#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
scoring = 'accuracy'
n_folds = 10

results, names  = [], [] 

for name, model  in clfs:
    kfold = KFold(n_splits=n_folds, random_state=None)
    cv_results = cross_val_score(model, X_train, y_train, 
                                 cv=kfold, scoring=scoring, n_jobs=-1)    
    names.append(name)
    results.append(cv_results)    
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  
                               cv_results.std())
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure(figsize=(15,6))
fig.suptitle('Classifier Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn", fontsize=20)
ax.set_ylabel("Accuracy of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.show()

# Logistic Regression Prediction and Feature Importance
rf =LogisticRegression(solver = "lbfgs", multi_class = "auto")
model(rf,X_train, y_train,
      X_val, y_val, "coef")

# XGBClassifier
xgb = XGBClassifier(n_estimators=800, n_jobs=-1)

xgb.fit(X_train.values,y_train.values)
predictions = xgb.predict(X_val.values)

print ("\naccuracy_score :",accuracy_score(y_val, predictions))

print ("\nclassification report :\n",(classification_report(y_val, predictions)))

plt.figure(figsize=(14,12))
plt.subplot(221)
sns.heatmap(confusion_matrix(y_val, predictions),
            annot=True,fmt = "d",linecolor="k",linewidths=3)

plt.title("CONFUSION MATRIX",fontsize=20)

# Random Forest Model and Feature Importances
rf =RandomForestClassifier(n_estimators=500)
model(rf,X_train, y_train,
      X_val, y_val, "feat")
