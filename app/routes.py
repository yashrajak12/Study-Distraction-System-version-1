from flask import Blueprint, Response, render_template
from .detection.face_tracker import StudyTracker

routes = Blueprint("routes", __name__)
tracker = StudyTracker()

@routes.route("/")
def home():
    return render_template("index.html")

@routes.route("/session")
def session():
    return render_template("session.html")

@routes.route("/video_feed")
def video_feed():
    return Response(
        tracker.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )