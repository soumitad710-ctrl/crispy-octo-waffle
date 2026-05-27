from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the Iris dataset
iris = load_iris()

#Features
X =iris.data

#Target labels
y = iris.target

#split the data
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size =0.2, random_state=42)

# create the model
model = RandomForestClassifier()

# train the moddel
model.fit(X_train, y_train)

#save model
pickle.dump(model, open('model.pkl','wb'))

print("Model trained and saved as model.pkl")
