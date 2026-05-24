from flask import Flask
from pathlib import Path
from dotenv import load_dotenv
from flask_wtf.csrf import CSRFProtect
import os


csrf = CSRFProtect()


def create_app():
    load_dotenv()

    app = Flask(__name__)

    flask_secret = os.getenv("FLASK_APP_SECRET")

    if not flask_secret:
        raise RuntimeError("Flask app secret environment variable is not set.")

    app.secret_key = flask_secret

    csrf.init_app(app)

    base = Path(__file__).resolve().parent
    app.config["UPLOAD_FOLDER"] = str(base / "static" / "uploads")
    app.config["RESULT_FOLDER"] = str(base / "static" / "results")
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["RESULT_FOLDER"]).mkdir(parents=True, exist_ok=True)

    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app