from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
def extract_embeddings():
  a_dataset = 'dataset'
  a_embeddings = 'output/embeddings.pickle'
  a_embedding_model = 'openface_nn4.small2.v1.t7'
  a_knownNames = 'output/knownNames.pickle'
  a_knownEmbeddings = 'output/knownEmbeddings.pickle'
  # load our serialized face detector from disk
  print("[INFO] loading face detector...")
  cascade = 'face_detection_classifier/intel_frontal_face_classifier.xml'
  # load our serialized face embedding model from disk
  print("[INFO] loading face recognizer...")
  embedder = cv2.dnn.readNetFromTorch(a_embedding_model)
  path_embeddings = 'output/embeddings.pickle'
  # grab the paths to the input images in our dataset
  print("[INFO] quantifying faces...")
  imagePaths = list(paths.list_images(a_dataset))
  # initialize our lists of extracted facial embeddings and
  # corresponding people names
  knownEmbeddings = []
  knownNames = []
  # initialize the total number of faces processed
  total = 0
  # loop over the image paths
  for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
      len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    # load the image, resize it to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    face_cascade = cv2.CascadeClassifier(cascade)
    # Convert into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for i in range(0, len(faces)):
      box = faces[i]
      (fx, fy, fw, fh) = box.astype("int")
      # extract the face ROI
      face = image[fy:fy + fh, fx:fx + fw]
      (fH, fW) = face.shape[:2]
      # ensure the face width and height are sufficiently large
      if fW < 20 or fH < 20:
        continue
      faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                      (96, 96), (0, 0, 0), swapRB=True, crop=False)
      embedder.setInput(faceBlob)
      vec = embedder.forward()
      knownNames.append(name)
      knownEmbeddings.append(vec.flatten())
      total += 1
  ##############################
  # dump the facial embeddings + names to disk
  data = {"embeddings": knownEmbeddings, "names": knownNames}
  f = open(path_embeddings, "wb")
  f.write(pickle.dumps(data))
  f = open(a_knownNames, "wb")
  f.write(pickle.dumps(knownNames))
  f = open(a_knownEmbeddings, "wb")
  f.write(pickle.dumps(knownEmbeddings))
  f.close()
if __name__ == "__main__":
  extract_embeddings()

