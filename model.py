import pandas as pd
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from seqeval.metrics import  f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline

def write_results_to_file(ids, tags, filename):
    list_dict = {'Id':ids, 'Labels':tags}
    df = pd.DataFrame(list_dict)
    df.to_csv(filename, sep = ',', index=False)

def load_dataset(task):
    if task is 1:
        train_data = pd.read_csv('train_task1.csv', encoding='utf-8')
        train_text = list(train_data['text'])
        train_label= list(train_data['label'])
    else:
        train_data = pd.read_csv('train_task2.csv', encoding='utf-8')
        train_text = list(train_data['News'])
        train_label= list(train_data['Label'])
    return(train_text, train_label)

def load_dev():
    dev_data = pd.read_csv('dev_task1.csv', encoding='utf-8')
    dev_text = list(dev_data['text'])
    dev_label= list(dev_data['label'])
    return(dev_text, dev_label)


def load_test():
    test_data = pd.read_csv('test_task1.csv', encoding='utf-8')
    test_ids  = list(test_data['Id'])
    test_text = list(test_data['text'])
    return(test_ids, test_text)

def text_clean(text):
    modified_text = re.sub("<br>", " ", text)
    pattern = re.compile(r'(.)\1\1+')
    subst = r"\1\1"
    modified_text = re.sub(pattern, subst, modified_text)
    return(modified_text)



# Initialize Classifiers
clf1 = SGDClassifier(random_state=42)
clf2 = MultinomialNB()
clf3 = SVC( kernel='linear')
clf4 = VotingClassifier(estimators=[ ('SVM-SGD', clf1),('NB', clf2),('SVM', clf3)],
                                    voting='hard')#,weights = [0.875, 0.860,0.810,0.700])
clfs = [clf1, clf2,clf3, clf4]
# Feature extraction
#tf_vect = TfidfVectorizer(max_df=MAX_DF,sublinear_tf=True,use_idf=True,max_features=MAX_FEATURE, stop_words= 'english')
tf_vect = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6))
#tf_vect = TfidfVectorizer()

# Fitting data for task 1
x_train, y_train = load_dataset(1)
x_train = [text_clean(x) for x in x_train]
#x_train = [remove_emoji_non_letter(x) for x in x_train]

x_dev, y_dev = load_dev()
x_dev = [text_clean(x) for x in x_dev]

#tfidf = TfidfVectorizer()
X = tf_vect.fit_transform(x_train)
X_dev = tf_vect.transform(x_dev)

# Feature Engineering
#features = tf_vect.get_feature_names()
#print(features)
#print(len(features))
# Train models on the entire dataset (development phase)
for clf in clfs:
    clf.fit(X, y_train)
    # Predict on the dev data
    predictions = clf.predict(X_dev)
    print(confusion_matrix(y_dev, predictions))
    print(classification_report(y_dev, predictions))

exit()

#Applying Models to testset

IDs, text = load_test()
x_test = [text_clean(x) for x in text]
X_test = tf_vect.transform(x_test)
clf4.fit(X, y_train)
# Predict on the test data
predictions = clf4.predict(X_test)
write_results_to_file(IDs, predictions, "NAYEL_task1_run1.csv")
clf3.fit(X, y_train)
# Predict on the test data
predictions = clf3.predict(X_test)
write_results_to_file(IDs, predictions, "NAYEL_task1_run2.csv")
clf1.fit(X, y_train)
# Predict on the test data
predictions = clf1.predict(X_test)
write_results_to_file(IDs, predictions, "NAYEL_task1_run3.csv")

# Run 1 for ensemble (clf4)
# Run 2 for SVM (clf3)
# Run 3 for SGD (clf1)

exit()
#Hamada Nayel (hamada.ali@fci.bu.edu.eg), Mohammed Aldawsari (mohammed.aldawsari@psau.edu.sa), Hosahalli Shashirekha (hlsrekha@gmail.com)
############
#Briefly describe what methodology you have used for this task
#(minimum one(5 to 10 lines) paragraph)
#(This might appear in task description(Overview) paper)
############
#A machine learning-based model has been developed integrated with tf-idf of the character n-grams for a wide range of characters as a feature extraction approach. Few preprocessing steps have been implemented to clean the text. The implemented ML-based classifiers are SVM, SGD, Naive Bayes. Furthermore, voting-based ensemble model phase been applied using the aforementioned classifiers.

############
# What is the most interesting aspect of your system
# (i.e., if we are to briefly describe your system in the task (Overview) paper,
# what would you want us to mention) ?
############
#The most interesting aspect of the proposed system is its simplicity. The system does not use external data nor uses extraneous computation resources such as servers or cloud computing. The system uses a primary feature extraction technique (tf-idf over character n-grams) and classical ML classification algorithms (SVM, SGD and Naive Bayes) as well as voting-based ensemble algorithm.








######
