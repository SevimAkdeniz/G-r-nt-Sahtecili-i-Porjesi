from flask import Flask
from pathlib import Path
from dotenv import load_dotenv
import os


def create_app():
    load_dotenv()

    app = Flask(__name__)

    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

    if not app.config["SECRET_KEY"]:
        raise RuntimeError("SECRET_KEY environment variable is not set.")

    base = Path(__file__).resolve().parent
    app.config["UPLOAD_FOLDER"] = str(base / "static" / "uploads")
    app.config["RESULT_FOLDER"] = str(base / "static" / "results")
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["RESULT_FOLDER"]).mkdir(parents=True, exist_ok=True)

    from .routes import main_bp
    app.register_blueprint(main_bp)

    return app