from helper import *
from routes.checkup import checkup, newest_image
#! After request and before response
@app.after_request
def add_header(response):
  response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
  response.headers['Cache-Control'] = 'public, max-age=0, must-revalidate, no-store'
  return response
app.register_blueprint(checkup)
if __name__ == '__main__':
  app.run(host=server,use_reloader=False, debug=True, threaded=True)
  # newest_image()