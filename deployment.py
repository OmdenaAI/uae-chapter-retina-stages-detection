import streamlit as st
import os
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import StringIO 


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

#Learning curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
import seaborn as sns
sns.set(style = 'whitegrid', palette = 'muted', font_scale = 2)
classes = ['DME', 'Normal']
cv = ShuffleSplit(n_splits = 100, test_size = 0.25, random_state = 0)
train_size = np.linspace(.1, 1.0, 15)
cmap = 'viridis'

@st.cache
def get_data(filename):
    dataset = pd.read_csv(filename)
    return dataset

df = get_data('./a.csv')
df2 = get_data('./b.csv')
df3 = get_data('./c.csv')
x_train = df.iloc[:,0:11]  #independent columns
y_train = df.iloc[:,-1]
x_test = df2.iloc[:,0:11]  #independent columns
y_test = df2.iloc[:,-1]
x_val = df3.iloc[:,0:11]  #independent columns
y_val = df3.iloc[:,-1]

# Learning curve
def LearningCurve(X, y, model, cv, train_sizes):

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv = cv, n_jobs = 4, 
                                                            train_sizes = train_sizes)

    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std  = np.std(train_scores, axis = 1)
    test_scores_mean  = np.mean(test_scores, axis = 1)
    test_scores_std   = np.std(test_scores, axis = 1)
    
    train_Error_mean = np.mean(1- train_scores, axis = 1)
    train_Error_std  = np.std(1 - train_scores, axis = 1)
    test_Error_mean  = np.mean(1 - test_scores, axis = 1)
    test_Error_std   = np.std(1 - test_scores, axis = 1)

    Scores_mean = np.mean(train_scores_mean)
    Scores_std = np.mean(train_scores_std)
    
    _, y_pred, Accuracy, Error, precision, recall, f1score,Acc_score = ApplyModel(X, y, model)
     
    return (model, Scores_mean, Scores_std)

def ApplyModel(X, y, model):
    
    model.fit(X, y)
    y_pred  = model.predict(X)

    Accuracy = round(np.median(cross_val_score(model, X, y, cv = cv)),2)*100
    Error   = 1 - Accuracy
    
    precision = precision_score(y_train, y_pred) * 100
    recall = recall_score(y_train, y_pred) * 100
    f1score = f1_score(y_train, y_pred) * 100
    Acc_score = accuracy_score(y_train, y_pred,normalize=True) * 100
    
    return (model, y_pred, Accuracy, Error, precision, recall, f1score,Acc_score )  
    
def PrintResults(model, X, y, title):
    
    model, y_pred, Accuracy, Error, precision, recall, f1score, Acc_score = ApplyModel(X, y, model)
    
    _, Score_mean, Score_std = LearningCurve(X, y, model, cv, train_size)
    Score_mean, Score_std = Score_mean*100, Score_std*100
    
    st.subheader('Here are the performance metrics for the trained model:')
    st.write('Scoring Accuracy: %.2f %%'%(Accuracy))
    st.write('Scoring Mean: %.2f %%'%(Score_mean))
    st.write('Scoring Standard Deviation: %.4f %%'%(Score_std))
    st.write("Precision: %.2f %%"%(precision))
    st.write("Recall: %.2f %%"%(recall))
    st.write('f1-score: %.2f %%'%(f1score))
    st.write('accuracy: %.2f %%'%(Acc_score))

def get_vertical_projection(img_dir):
    _, bw_img = cv2.threshold(img_dir, 127, 255, cv2.THRESH_BINARY)
    X_data = []
    Y_data = []
    x,y,c = bw_img.shape
    for Y in range(y):
        for X in range(x):
            if bw_img[X][Y][0] == 255:
                X_data.append(Y)
                Y_data.append(x - X)
                break
    poly = np.polyfit(X_data, Y_data , deg= 10)
    return np.array(poly)

def main():

    header = st.container()
    model_training = st.container()
    files = st.container()



    with header:
        st.title("Machine Learning Classifier for Retina Multi Stages Formation/Deformation Detection")
        st.subheader("This project investigate the opportunity to explore the best machine learning classifier to detect the retina deformation and supporting the DME identification by analysing the best features of the disease.")

    with model_training:
        st.subheader("Time to train the model")


        if st.button('Train the model'):

            model_rf_final = XGBClassifier(
            learning_rate =0.1,
            n_estimators=150, max_depth=6,
            min_child_weight=1, gamma=0, 
            subsample=0.8,colsample_bytree=0.8,
            nthread=4,
            scale_pos_weight=1,seed=27)

            PrintResults(model_rf_final, x_train, y_train, 'XGB')

    with files:
        st.subheader("You can now use the model for inferencing.")

        oct_image_file = st.file_uploader("Upload DME OCT", type=["jpeg"])


        if oct_image_file is not None:
            u_img = Image.open(oct_image_file)
            st.image(u_img)
            open_cv_image = np.array(u_img)

            model_rf_final = XGBClassifier(
            learning_rate =0.1,
            n_estimators=150, max_depth=6,
            min_child_weight=1, gamma=0, 
            subsample=0.8,colsample_bytree=0.8,
            nthread=4,
            scale_pos_weight=1,seed=27)

            model_rf_final.fit(x_train, y_train)

            img_poly = get_vertical_projection(open_cv_image)
            columns = np.arange(0,11,1)
            data = pd.DataFrame([img_poly], columns=columns)

            if model_rf_final.predict(data) > 0.5:
                st.subheader('The result is:')
                st.write('Diabetic Macular Edema is observable in the image.')
                st.write(model_rf_final.predict(data))
                st.write(model_rf_final.predict_proba(data))
                print(data)
            else:
                st.subheader('The result is:')
                st.write('Patient seems healthy')
                st.write(model_rf_final.predict(data))
                st.write(model_rf_final.predict_proba(data))
                print(data)
                


if __name__ == "__main__":
    main()


    
