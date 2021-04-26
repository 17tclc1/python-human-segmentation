from helper import *
checkup = Blueprint('checkup', __name__)
@checkup.route('/')
def index():
  data = []
  allUsers = []
  return render_template('./partials/index.html', allUsers=data, server=server)
@checkup.route('/start-check-up', methods=['POST'])
def startCheckUp():
  try:
    embedder = cv2.dnn.readNetFromTorch(path_embedding_model)
    # Face recognizer model
    recognizer = pickle.loads(open(path_recognizer, "rb").read())
    le = pickle.loads(open(path_le_pickle, "rb").read())

    cap = cv2.VideoCapture(0)
    cap.set(3,640) 
    cap.set(4,480)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = path_faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20,20)
    )
    if len(faces) > 0:
      for i in range(0, len(faces)):
        # extract the confidence (i.e., probability) associated with the
        box = faces[i]
        (fx, fy, fw, fh) = box.astype("int")
        # extract the face ROI
        face = frame[fy:fy + fh, fx:fx + fw]
        (fH, fW) = face.shape[:2]
        # ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
          continue
        # construct a blob for the face ROI, then pass the blob
        # through our face embedding model to obtain the 128-d
        # quantification of the face
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()
        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        prob = preds[j]
        name = le.classes_[j]
        if prob >= 0.4:
          id = name.split('-')[1]
          name = name.split('-')[0]
        else:
          id = 0
          name = 'unknown'
        attendanceID = id
        attendanceName = name
        print(id, ' ',name, ' ', prob * 100)
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
    # No face
    if(len(faces) == 0):
      attendanceID = "unknown"
      attendanceName="unknown"
    cv2.imwrite('./static/img/last_taken/lastImg.jpg', frame)
    # More than 1 face
    if len(faces) > 1:
      return jsonify({'status': 'duplicate'})
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'id': attendanceID, 'name': attendanceName, 'status': 'ok'})
  except Exception as error:
    print(error)
    return jsonify({'status': 'failed'})