from flask import Flask, render_template, request
import cv2
import numpy as np
from ..detection.face_tracker import StudyTracker

 # ✅ Correct import

app = Flask(__name__)
tracker = StudyTracker()

@app.route('/')
def index():
    return render_template('session.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    file = request.files['frame']
    img = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Optional: process frame
    print("Frame received for analysis ✅")

    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
