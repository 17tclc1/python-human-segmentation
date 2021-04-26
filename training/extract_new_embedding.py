from imutils import paths
import imutils
from training import train
import pickle
import cv2
import os
def embeddings_new_ds():
    a_dataset = './training/temporary_dataset'
    a_embeddings = './training/output/embeddings.pickle'
    a_knownNames = './training/output/knownNames.pickle'
    a_knownEmbeddings = './training/output/knownEmbeddings.pickle'
    a_embedding_model = './training/openface_nn4.small2.v1.t7'
    cascade = './training/face_detection_classifier/intel_frontal_face_classifier.xml'

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(a_embedding_model)

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")

    imagePaths = list(paths.list_images(a_dataset))
    # initialize our lists of extracted facial embeddings and # corresponding people names

    knownEmbeddings = []
    knownNames = []
    # initialize the total number of faces processed

    total = 0

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):

        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        name = str(name)

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
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

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0), swapRB=True, crop=False)

            embedder.setInput(faceBlob)
            vec = embedder.forward()
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1
    ##############################

        t_knownNames = pickle.loads(open(a_knownNames, "rb").read())
        t_knownEmbeddings = pickle.loads(open(a_knownEmbeddings, "rb").read())

    list_index = []
    for i, names in enumerate(t_knownNames):
        if names == name:
            list_index.append(i)
    try:
        del t_knownNames[list_index[0]:list_index[len(list_index) - 1]]
        del t_knownEmbeddings[list_index[0]:list_index[len(list_index) - 1]]
        print('updated')
    except:
        print('added')

    try:
        for i, name in enumerate(t_knownNames):
            knownNames.append(name)
        for i, embedding in enumerate(t_knownEmbeddings):
            knownEmbeddings.append(embedding)
    except:
        print("error")

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(a_embeddings, "wb")
    f.write(pickle.dumps(data))
    f = open(a_knownNames, "wb")
    f.write(pickle.dumps(knownNames))
    f = open(a_knownEmbeddings, "wb")
    f.write(pickle.dumps(knownEmbeddings))
    f.close()
    
    if len(next(os.walk('./training/dataset/'))[1]) > 1 : # check number of subfolders(dataset)
        train.train()  # training from embbeddings
    else:
        print('There is only one face')




