from helper import *
checkup = Blueprint('checkup', __name__)
@checkup.route('/')
def index():
  return render_template('./partials/index.html', server=server)
@checkup.route('/upload', methods=['POST'])
def startCheckUp():
  try:
    # This is the blob sent from the client-side
    fileByte = request.files['data'].read()
    # Please handle your segmentation logic here


    # Save this file to the local
    with open('./static/img/last_taken/lastImg.jpg', 'wb') as file:
        file.write(fileByte)
    return jsonify({ 'status': 'ok' })
  except Exception as error:
    print(error)
    return jsonify({'status': 'failed'})