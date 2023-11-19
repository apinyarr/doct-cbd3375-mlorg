import boto3
from io import BytesIO
import pandas as pd
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import CountVectorizere
from sklearn.feature_extraction.text import CountVectorizer
# Import TfidfTransform
# Tfidf stands for term-frequency times inverse document-frequency.
from sklearn.feature_extraction.text import TfidfTransformer
# Inport Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# Import classification_report
from sklearn.metrics import classification_report
# import module
import pickle
import threading

tweeter_df = None
X, y, X_train, X_test, y_train, y_test = [], [], [], [], [], []
X_train_cv, X_test_cv = None, None
X_train_tf, X_test_tf = None, None
countvector, tfidf_transformer, nb_multic_lm = None, None, None
bucket = 'dataset-3375-2'

def download_dataset(bucket = 'dataset-3375-2', file = 'processed_dataset.csv'):
  try:
    # os.remove('dataset.csv')
    # Get the object from S3
    s3 = boto3.resource('s3')
    with BytesIO() as data:
      s3.Bucket(bucket).download_file(file, 'processed_dataset.csv')
  except Exception as e:
    print(e)
    print('error occurred')
    raise e
  
def load_dataframe(file = 'processed_dataset.csv'):
  global tweeter_df 
  tweeter_df = pd.read_csv(file)
  print(tweeter_df.head(5))

def train_test_data_split(ctest_size=0.20, crandom_state=50):
  global X, y, X_train, X_test, y_train, y_test, tweeter_df
  # Identify predictor and target features
  X = tweeter_df['clean_text']
  y = tweeter_df['encoded_type']
  # Identify train and test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ctest_size, random_state=crandom_state, stratify=y)

def set_count_vector():
  global X_train, X_test, X_train_cv, X_test_cv, countvector
  countvector = CountVectorizer()
  X_train_cv = countvector.fit_transform(X_train.values.astype('U'))
  X_test_cv = countvector.transform(X_test)

def set_tfidf_transformer():
  global X_train_tf, X_test_tf, tfidf_transformer
  tfidf_transformer = TfidfTransformer(use_idf = True).fit(X_train_cv)
  X_train_tf = tfidf_transformer.transform(X_train_cv)
  X_test_tf = tfidf_transformer.transform(X_test_cv)

def model_fitting():
  global nb_multic_lm
  nb_multic_lm = MultinomialNB()
  nb_multic_lm.fit(X_train_tf, y_train)

def show_evaluation_report():
  global nb_multic_lm
  # Predict the training data
  nb_train_pred = nb_multic_lm.predict(X_train_tf)
  type(nb_train_pred)
  # Predict the test data
  nb_test_pred = nb_multic_lm.predict(X_test_tf)
  map_types = ['age', 'ethnicity', 'gender', 'not_cyberbullying', 'religion']
  print("Classification Report for train data:\n===================================\n") 
  print(f"{classification_report(y_train, nb_train_pred, target_names = map_types)}\n\n")

  print("Classification Report for test data:\n===================================\n") 
  print(f"{classification_report(y_test, nb_test_pred, target_names = map_types)}\n")

def save_objects_to_files():
  filename = 'cyberbullying_model.sav'
  pickle.dump(nb_multic_lm, open(filename, 'wb'))

  countvector_file = 'countvector.sav'
  pickle.dump(countvector, open(countvector_file, 'wb'))

  tfidf_transformer_file = 'tfidf_transformer.sav'
  pickle.dump(tfidf_transformer, open(tfidf_transformer_file, 'wb'))

def upload_model_to_s3():
  try:
    # Get the object from S3
    s3 = boto3.client('s3')
    response = s3.upload_file('cyberbullying_model.sav', bucket, 'cyberbullying_model.sav')
    response = s3.upload_file('countvector.sav', bucket, 'countvector.sav')
    response = s3.upload_file('tfidf_transformer.sav', bucket, 'tfidf_transformer.sav')
  except Exception as e:
    print(e)
    print('error occurred')
    raise e

thread1 = threading.Thread(target=download_dataset)
thread1.start()
thread1.join()
thread2 = threading.Thread(target=load_dataframe)
thread2.start()
thread2.join()
thread3 = threading.Thread(target=train_test_data_split)
thread3.start()
thread3.join()
thread4 = threading.Thread(target=set_count_vector)
thread4.start()
thread4.join()
thread5 = threading.Thread(target=set_tfidf_transformer)
thread5.start()
thread5.join()
thread6 = threading.Thread(target=model_fitting)
thread6.start()
thread6.join()
thread7 = threading.Thread(target=show_evaluation_report)
thread7.start()
thread7.join()
thread8 = threading.Thread(target=save_objects_to_files)
thread8.start()
thread8.join()
thread9 = threading.Thread(target=upload_model_to_s3)
thread9.start()
thread9.join()