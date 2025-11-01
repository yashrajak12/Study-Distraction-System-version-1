from flask import Flask
import os

def create_app():
    # Get the directory where __init__.py is located
    here = os.path.dirname(os.path.abspath(__file__))

    app = Flask(
        __name__,
        template_folder=os.path.join(here, "templates"),
        static_folder=os.path.join(here, "static"),
    )

    # Register blueprint
    from .routes import routes
    app.register_blueprint(routes)

    return app
