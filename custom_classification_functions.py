# Business parameters and Functions

# Initial imports
import pandas as pd
import numpy as np
from pathlib import Path
import timeit
import joblib
import mlflow
import warnings
warnings.filterwarnings("ignore")    


# ------------------------------------------------FUNCTIONS-------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def label_feature_split_scale_df(df_main, label_col_name, scale_obj):
    """
    Function to clean, split the features from the target column and scales X.
    """
    
    df = df_main.copy()
    
    # Drop NAs and Replace Infs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    y = df[f'{label_col_name}']
    X = df.drop(columns=f'{label_col_name}')

    # Create a Scaler object
    if scale_obj == 0:
        scaler = StandardScaler()
    elif scale_obj == 1:
        scaler = MinMaxScaler()
    else:
        print('Scaler Error')

    # Fit the MinMaxScaler object with the features data X
    scaler.fit(X)
    
    import os
    # If best_model folder does not exist it will create one
    if not os.path.exists("best_model"):
        os.makedirs("best_model")
    
    # Save scaler to be used with new data
    joblib.dump(scaler, 'best_model/scaler_fit_original.gz')

    # Scale the features
    X_scaled = scaler.transform(X)
    
    return X_scaled, y, scaler, X








from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def prec_recall(y_test, y_hat):
    """
    Function to create a confusion matrix and calculating Recall, Precision and F scores
    """

    # Create Confusion Matrix
    conf_matx = confusion_matrix(y_test, y_hat)

    # Create a data frame of the Confusion Matrix
    conf_matx_df = pd.DataFrame(conf_matx, index = ['Actual 0', 'Actual 1'], columns = ['Predicted 0', 'Predicted 1'])

    # Calculate True Negative (TN)
    tn = conf_matx_df.loc['Actual 0']['Predicted 0']
    # Calculate False Positive (FP) Type I Error
    fp = conf_matx_df.loc['Actual 0']['Predicted 1']
    # Calculate False Negative (FN) Type II Error
    fn = conf_matx_df.loc['Actual 1']['Predicted 0']
    # Calculate True Positive (FP)
    tp = conf_matx_df.loc['Actual 1']['Predicted 1']
    
    # Calculate Preccision: All the cases that the model predicted to be positive, how many actually are positive
    precision = tp / (tp + fp)
    
    # Calculate Recall: All the cases that are positive, how many did the model identify
    recall = tp / (tp + fn)
    

    return precision, recall








def multicollinearity_df(df, multicol_thresh):
    """
    Function to create a dataframe of columns with multicollinearity
    """

    # Create multicollinearity matrix
    multicollinearity_matrix = (df.corr())[abs(df.corr()) > multicol_thresh]

    # Removes correlation of itself and duplacates on lower half of matrix
    multicollinearity_matrix = multicollinearity_matrix.mask(np.tril(np.ones(multicollinearity_matrix.shape)).astype(np.bool))

    # Resahpe dataframe to show without NaN
    multicollinearity_matrix = multicollinearity_matrix.stack().reset_index()
    
    return multicollinearity_matrix








import plotly.graph_objects as go

def roc_fig(fpr, tpr, model_type, auc):
    """
    Function to configure plotly Receiver Operating Characteristic (ROC) Curve plot
    """

    # Define figure
    fig = go.Figure()

    # Create ROC plot
    fig.add_trace(go.Scatter(
        x = fpr,
        y = tpr,
        showlegend=False,
        fill='tonexty',
        mode='lines', line_color='aqua', line=dict(width=0)))

    # Add black outline to area chart
    fig.add_trace(go.Scatter(
        x = fpr,
        y = tpr,
        showlegend=False,
        line_color = 'black',
        line=dict(color='black', width=1)))

    # Adjust chart settings
    fig.update_layout(
        width=400,
        height=400,
        margin=dict(
            l=20,
            r=20,
            b=20,
            t=20,
            pad=0),
        paper_bgcolor="white",)

    # Update axis
    fig.update_xaxes(
            title_text = "False Positive Rate",
            title_standoff = 1,
            showgrid=False)
    fig.update_yaxes(
            title_text = "True Positive Rate",
            title_standoff = 1,
            showgrid=False)

    # Add triangle to make a filled in 50% line
    fig.update_layout(
        shapes=[
            # filled Triangle
            dict(type="path",
                path=" M 0 0 L 1 1 L 1 0 Z",
                fillcolor="white",
                line_color="white")])

    # Add dashed 50% line
    fig.add_shape(type="line",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="black",
            width=1,
            dash="dashdot"))

    # Add a box for a 1 by 1 square outline
    fig.add_shape(type="rect",
        x0=0, y0=0, x1=1, y1=1)

    # Create scatter trace of text labels
    fig.add_trace(go.Scatter(
        x=[.4],
        y=[.5],
        text=["50% Line"],
        mode="text",
        showlegend=False))

    # Add title
    fig.update_layout(
        template='plotly_white',
        title={
            'text': '{} ROC Curve - AUC: {}%'.format(model_type, round(auc*100,2)),
            'y':.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    return fig








def conf_matx_fig(conf_matx_df, model_type, precision, recall):
    """
    Function to configure plotly for a confusion matrix plot
    """

    # Define figure
    fig = go.Figure()

    # Create Confusion Matrix Plot
    fig.add_trace(go.Heatmap(
        z = [[conf_matx_df.loc['Actual 1']['Predicted 0'], conf_matx_df.loc['Actual 1']['Predicted 1']],
             [conf_matx_df.loc['Actual 0']['Predicted 0'], conf_matx_df.loc['Actual 0']['Predicted 1']]],
        text = [['FN', 'TP'], ['TN', 'FP']],
        x = ['Predicted 0', 'Predicted 1'],
        y = ['Actual 1', 'Actual 0'],
        colorscale = [[0, 'rgb(173,253,245)'], [1, 'rgb(9,4,183)']],
        showscale=False))

    # Adjust chart settings
    fig.update_layout(
        width=600,
        height=400,
        margin=dict(
            l=20,
            r=20,
            b=20,
            t=20,
            pad=0))

    # Add title
    fig.update_layout(
        title={
            'text': '{} Confusion Matrix - Precision: {} Recall {}'.format(model_type, round(precision,2), round(recall,2)),
            'y':.99,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    # Add text and legend to plot
    fig.add_trace(go.Scatter(
        x=['Predicted 0', 'Predicted 1', 'Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 0', 'Actual 1', 'Actual 1'],
        mode="text",
        name="Markers and Text",
        text= [f"{conf_matx_df.loc['Actual 0']['Predicted 0']} TN", f"{conf_matx_df.loc['Actual 0']['Predicted 1']} FP I P",
               f"{conf_matx_df.loc['Actual 1']['Predicted 0']} FN II R", f"{conf_matx_df.loc['Actual 1']['Predicted 1']} TP"],
        textposition="bottom center",
        textfont = dict(size = 50)))

    return fig








def feat_import_rf_fig(rf_importances, num_features_show, X):
    """
    Function to configure plotly for Random Forest feature importance plot
    """
    
    # Get the list the important features and sort them
    rf_importances_sorted = sorted(zip(rf_importances, X.columns), reverse=True)
    
    # create data frame of important features
    rf_importances_df = pd.DataFrame(rf_importances_sorted[:num_features_show])
    rf_importances_df.set_index(rf_importances_df[1], inplace=True)
    rf_importances_df.drop(columns=1, inplace=True)
    rf_importances_df.rename(columns={0: 'Feature Importances'}, inplace=True)
    rf_importances_sorted_df = rf_importances_df.sort_values(by='Feature Importances')

    # Define figure
    fig = go.Figure()

    # Add data to plot
    fig.add_trace(go.Bar(
        x = rf_importances_sorted_df['Feature Importances'],
        y = list(rf_importances_sorted_df.index),
        orientation='h',
        marker=dict(
            color='rgba(50, 171, 96, 0.6)',
            line=dict(
                color='rgba(50, 171, 96, 1.0)',
                width=1)),))

    # Update format of plot
    fig.update_layout(
        template='plotly_white',
        title={
            'text': 'Feature Importances from Random Forest',
            'y':.99,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True),
        xaxis=dict(
            zeroline=False,
            showline=False,
            showticklabels=True,
            showgrid=True),
        legend=dict(x=0.029, y=1.038, font_size=10),
        margin=dict(l=100, r=20, t=70, b=70),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',)

    # Adjust chart settings
    fig.update_layout(
        width=800,
        height=400,
        margin=dict(
            l=20,
            r=20,
            b=20,
            t=20,
            pad=0),
        paper_bgcolor="white",)

    # Adding % labels to bars
    annotations = []
    for yd, xd in zip(rf_importances_sorted_df['Feature Importances'], list(rf_importances_sorted_df.index)):
        annotations.append(dict(xref='x1', yref='y1',
                                y=xd, x=yd + .015,
                                text='{}%'.format(np.round(yd * 100, decimals=1)),
                                font=dict(family='Arial', size=12,
                                          color='rgb(50, 171, 96)'),
                                showarrow=False))
    
    # Add annotations to plot
    fig.update_layout(annotations=annotations)
    
    return fig