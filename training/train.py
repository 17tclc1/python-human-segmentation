from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
def train():
  path_embeddings = './training/output/embeddings.pickle'
  path_recognizer = './training/output/recognizer.pickle'
  path_le_pickle = './training/output/le.pickle'
  # load the face embeddings
  print("[INFO] loading face embeddings...")
  data = pickle.loads(open(path_embeddings, "rb").read())

  # encode the labels
  print("[INFO] encoding labels...")
  le = LabelEncoder()
  labels = le.fit_transform(data["names"])
  # train the model used to accept the 128-d embeddings of the face and
  # then produce the actual face recognition
  print("[INFO] training model...")
  recognizer = SVC(C=1.0, kernel="linear", probability=True)
  recognizer.fit(data["embeddings"], labels)

  # write the actual face recognition model to disk
  f = open(path_recognizer, "wb")
  f.write(pickle.dumps(recognizer))
  f.close()

  # write the label encoder to disk
  f = open(path_le_pickle, "wb")
  f.write(pickle.dumps(le))
  f.close()
if __name__ == "__main__":
  train()