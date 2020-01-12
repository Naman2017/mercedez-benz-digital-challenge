from flask import Flask, render_template
from flask import jsonify
from flask import request
import numpy as np
import base64
import io
from imageio import imread
import run_both
import cv2



def decd(b64_string):
    # reconstruct image as an numpy array
    img = imread(io.BytesIO(base64.b64decode(b64_string)))
    return img



app=Flask(__name__,template_folder='static')

@app.route('/',methods=["GET"])
def home():
	return render_template('index.html')

@app.route('/app',methods=["GET"])
def app_home():
	return render_template('app.html')

@app.route('/capture',methods=['POST'])
def capture():
    image=request.form['img']
    img=decd(image[23:])
    flag=run_both.pred(img)
    print(flag)

    return jsonify({'val':flag})

if __name__ == '__main__':
    app.run(debug=True)