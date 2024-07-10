
import pandas as  pd 
msg=pd.read_csv('naivetext.csv',names=['message','label']) 
print("The dimension of the dataset",msg.shape) 
msg['labelnum']=msg.label.map({'pos':1,'neg':0}) 
X=msg['message'] 
y=msg['labelnum'] 
print(X) 
print(y) 
 
# Split data into training and testing sets
from sklearn.model_selection import train_test_split 
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y) 
print("\n The total no of training data",ytrain.shape) 
print("\n The total no of testing data",ytest.shape) 

# Vectorize text
from sklearn.feature_extraction.text import CountVectorizer 
count_vect=CountVectorizer() 
Xtrain_dtm=count_vect.fit_transform(Xtrain) 
Xtest_dtm=count_vect.transform(Xtest) 
print("\n The words or token in text document\n") 
print(count_vect.get_feature_names_out()) 

 

# Train classifier
from sklearn.naive_bayes import MultinomialNB 
clf=MultinomialNB().fit(Xtrain_dtm,ytrain) 
predicted=clf.predict(Xtest_dtm) 
 
# Predict and evaluate
from sklearn import metrics 
print("Accuracy:", metrics.accuracy_score(ytest, predicted))
print("Confusion Matrix:\n", metrics.confusion_matrix(ytest, predicted))
print("Precision:", metrics.precision_score(ytest, predicted))
print("Recall:", metrics.recall_score(ytest, predicted))