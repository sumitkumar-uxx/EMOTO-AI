import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib 
file  = pd.read_csv(r"C:\Users\sumit\OneDrive\Pictures\Documents\tweet_emotions.csv") #loading the dataset
frame =  pd.DataFrame(file)#converting dataset into the dataframe 
label = LabelEncoder()#encoding the catograial values
frame["sentiment"]=label.fit_transform(frame["sentiment"]) #encoding the catograial values
inputs = frame["content"]  # deciding the input
output = frame["sentiment"] # deciding the output
print(frame)#print the dataframe
fVectorizer = TfidfVectorizer() # for using to take input as the text from the dataset and converting into encoded for MODEL
fianl = fVectorizer.fit_transform(inputs) # for using to take input as the text from the dataset and converting into encoded for MODEL
x_train , x_test , y_train , y_test = train_test_split(fianl,output,random_state=42,train_size=0.8) #spliting the data for training and testing
model = LGBMClassifier() #model for training
model.fit(x_train,y_train)#fitting data into the model
joblib.dump(model,"coded_first.pkl")
joblib.dump(label,"label.pkl")
joblib.dump(fVectorizer,"vector.pkl")
print("done")