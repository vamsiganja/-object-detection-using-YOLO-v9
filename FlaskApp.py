#Import All the Required Libraries
from flask import Flask, render_template, Response, jsonify, request, session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from detection import objectDetection
import os
import cv2

#Initialize the Flask Application
app = Flask(__name__)
#Configure a secret key
app.config['SECRET_KEY'] = 'lokesh'
#Store the input files uploaded in the application
app.config['UPLOAD_FOLDER'] = 'static/files'

#Use FlaskForm to get the input file from the user
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")
fpsCount = 0
frameSize = 0
detectedObjects = 0
def generate_frames(path):
    yolov9_output = objectDetection(path)
    for im0, frameRate, frameShape, totalDetection in yolov9_output:
        ret, buffer = cv2.imencode('.jpg', im0)
        global fpsCount
        fpsCount = str(frameRate)
        global frameSize
        frameSize = str(frameShape[0])
        global detectedObjects
        detectedObjects = str(totalDetection)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


@app.route('/', methods = ['GET', 'POST'])
def front():
    form  = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        session['filePath'] = file_path
        return render_template('index.html', form = form, uploaded = True, filename = filename)
    return render_template('index.html', form = form, uploaded = False)

@app.route('/detections', methods = ['GET', 'POST'])
def detections():
    file_path = session.get('filePath', None)
    if file_path:
        return Response(generate_frames(path= file_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Not even uploaded"

@app.route('/fps', methods = ['GET'])
def fps():
    global fpsCount
    return jsonify(fpsresult = fpsCount)

@app.route('/dcount', methods = ['GET'])
def dcount():
    global detectedObjects
    return jsonify(dcountresult = detectedObjects)

@app.route('/fsize', methods = ['GET'])
def fsize():
    global frameSize
    return jsonify(fsizeresult = frameSize)

if __name__ == "__main__":
    app.run(debug=True)


