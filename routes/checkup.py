import os
from helper import *
from datetime import datetime
from pathlib import Path

checkup = Blueprint('checkup', __name__)
BASE_DIR = Path(__file__).resolve().parent.parent
pathSave = os.path.join(BASE_DIR, 'static', 'img', 'taken')

@checkup.route('/')
def index():
  img = newest_image()
  return render_template('./partials/index.html', server=server, img=img)

@checkup.route('/upload', methods=['POST'])
def startCheckUp():
  try:
    # This is the blob sent from the client-side
    fileByte = request.files['data'].read()

    # Save this file to the local
    
    img_name = str(datetime.timestamp(datetime.now())) + '.jpg'
    img_to_save_path = os.path.join(
      pathSave,
      img_name
    )
    print(img_to_save_path)
    with open(img_to_save_path, 'wb') as file:
        file.write(fileByte)

    # Load modal
    model = tf.keras.models.load_model("unet(10).h5")

    # Read image contain person
    original = cv2.imread(img_to_save_path)
    heightPersonImage, widthPersonImage, _ = original.shape   # Image: 3D array

    # Background
    background = cv2.imread('./static/img/bg.jpg')
    background = cv2.resize(background, (widthPersonImage, heightPersonImage))

    # Tien hanh predict mask
    img = cv2.resize(original, (224, 224))
    img = img/255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    pred_mask = model.predict(img)[0] 

    # Flatten 3D array to 2D array
    pred_mask = (pred_mask > 0.5) * 255
    pred_mask = pred_mask.astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, (widthPersonImage, heightPersonImage))

    # get not of mask
    pred_mask_inv = cv2.bitwise_not(pred_mask)

    # Now black-out the area of logo in ROI
    print(pred_mask_inv.shape)
    img1_bg = cv2.bitwise_and(background,background,mask = pred_mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(original,original,mask = pred_mask)

    # Put logo in ROI and modify the main image
    background = cv2.add(img1_bg,img2_fg)
    cv2.imwrite(img_to_save_path, background)
    img_path = os.path.join(
      'static',
      'img',
      'taken',
      img_name
    )

    return jsonify({ 'status': 'ok', 'img': img_path })
  except Exception as error:
    print(error)
    return jsonify({'status': 'failed'})

def newest_image():
  imgs = []
  
  for f in os.listdir(pathSave):
      file_name = os.path.splitext(f)[0]
      imgs.append(file_name)
  imgs.sort(reverse=True)
  
  if len(imgs) == 0:
    return 'static/img/last_taken/null.jpg'
  
  img_with_ext = imgs[0] + '.jpg'
  img_path = 'static/img/taken/' + img_with_ext
  
  return img_path
