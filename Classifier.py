
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import pandas as pd

legitimate_urls = pd.read_csv("legitimate-urls.csv")
phishing_urls = pd.read_csv("phishing-urls.csv")

print(len(legitimate_urls))
print(len(phishing_urls))

# Merging both the dataframe as they have same columns
urls = legitimate_urls.append(phishing_urls)

print(len(urls))
print(urls.columns)

# Removing Unnecessary columns
urls = urls.drop(urls.columns[[0, 3, 5]], axis=1)
print(urls.columns)

# Shuffling the rows in the dataset for equal distribution in train and test data
urls = urls.sample(frac=1).reset_index(drop=True)

# #### Removing class variable from the dataset
urls_without_labels = urls.drop('label', axis=1)
urls_without_labels.columns
labels = urls['label']
# labels

# Splitting the dataset into train data and test data
random.seed(100)
data_train, data_test, labels_train, labels_test = train_test_split(
    urls_without_labels, labels, test_size=0.20, random_state=100)
print(len(data_train), len(data_test), len(labels_train), len(labels_test))
print(labels_train.value_counts())
print(labels_test.value_counts())

# ## Random Forest

RFmodel = RandomForestClassifier()
RFmodel.fit(data_train, labels_train)
rf_pred_label = RFmodel.predict(data_test)

cm2 = confusion_matrix(labels_test, rf_pred_label)
print(cm2)
print(accuracy_score(labels_test, rf_pred_label))

# Saving the model to a file
file_name = "RandomForestModel.sav"
pickle.dump(RFmodel, open(file_name, 'wb'))
