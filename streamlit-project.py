
import pandas as pd
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 


def get_model(path):
    df = pd.read_csv(path)
    df.drop(["Cabin" axis=1, inplace=True)

    df["Fare"].fillna(value = dfSurvivals["fare"].mean(), inplace=True)
    df["Embarked"].fillna(dfSurvivals["embarked"].value_counts().idxmax(), inplace=True)
    df["Age"].fillna(value = dfSurvivals["age"].mean(), inplace=True)

    

    labelencoder_X = LabelEncoder()

    df["Name"] = labelencoder_X.fit_transform(dfSurvivals["name"])
    df["Embarked"] = labelencoder_X.fit_transform(dfSurvivals["embarked"])
    df["Ticket"] = dfSurvivals["ticket"].astype(str)
    df["Ticket"] = labelencoder_X.fit_transform(dfSurvivals["ticket"])



    result = OneHotEncoder().fit_transform(dfSurvivals["Sex"].values.reshape(-1, 1)).toarray()
    dfSurvivals[["female", "male"]] = pd.DataFrame(result, index = dfSurvivals.index)
    dfSurvivals.drop(["Sex"], axis=1, inplace=True)


    X = dfSurvivals.drop("survived", axis=1)
    y = dfSurvivals["survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train,y_train)

    return model  


def main():
    """ main() contains all UI structure elements; getting and storing user data can be done within it"""
    st.title("Titanic Classification")                                                                              ## Title/main heading
    st.image(r"the-titanic.webp", caption="Sinking of 'RMS Titanic' : 15 April 1912 in North Atlantic Ocean",use_column_width=True) ## image import
    st.write("""## Would you have survived From Titanic Disaster?""")                                                    ## Sub heading


    model = get_model('./titanic_train.csv')
 
    age = st.slider("Enter Age :", 1, 75, 30)                                                                  # slider for user input(ranges from 1 to 75 & default 30)

    fare = st.slider("Fare (in 1912 $) :", 15, 500, 40)                                                        # slider for user input(ranges from 15 to 500 & default 40)

    SibSp = st.selectbox("How many Siblings or spouses are travelling with you?", [0, 1, 2, 3, 4, 5, 6, 7, 8]) # Select box

    Parch = st.selectbox("How many Parents or children are travelling with you?", [0, 1, 2, 3, 4, 5, 6, 7, 8]) # Select box

    sex = st.selectbox("Select Gender:", ["Male","Female"])                         # select box for gender[Male|Female]
    if (sex == "Male"):                                                             # selected gender changes to [Male:0 Female:1]
        Sex=0
    else:
        Sex=1

    Pclass= st.selectbox("Select Passenger-Class:",[1,2,3])                        # Select box for passenger-class

    boarded_location = st.selectbox("Boarded Location:", ["Southampton","Cherbourg","Queenstown"]) ## Select Box for Boarding Location
    Embarked_C,Embarked_Q,Embarked_S=0,0,0                     # initial values are 0
    # As we encoded these using one-hot-encode im ml model; so if 'Q' selected value is C=0,Q=1;S=0 , if 'S' selected value is C=0,Q=0;S=1 likewise
    if boarded_location == "Queenstown":
        Embarked_Q=1
    elif boarded_location == "Southampton":
        Embarked_S=1
    else:
        Embarked_C=1

   
    data={"Age":age,"Fare":fare,"SibSp":SibSp,"Parch":Parch,"Sex":Sex,"Pclass":Pclass,"Embarked_Q":Embarked_Q,"Embarked_S":Embarked_S,"Embarked_C":Embarked_C}

    df=pd.DataFrame(data,index=[0])     
    return df

data=main()                         

## Prediction:
if st.button("Predict"):                                                              
    result = model.predict(data)                                                       
    proba=model.predict_proba(data)                                                     
    #st.success('The output is {}'.format(result))

    if result[0] == 1:
        st.write("***Congratulation!!!...*** **You probably would have made it!**")
        st.image(r"alive.jfif")
        st.write("**Survival Probability Chances :** 'NO': {}%  'YES': {}% ".format(round((proba[0,0])*100,2),round((proba[0,1])*100,2)))
    else:
        st.write("***Better Luck Next time!!!...*** **you're probably Ended up like 'Jack'**")
        st.image(r"restinpeace.jfif")
        st.write("**Survival Probability Chances :** 'NO': {}%  'YES': {}% ".format(round((proba[0,0])*100,2),round((proba[0,1])*100,2)))
