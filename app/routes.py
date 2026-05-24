from pathlib import Path
from flask import Blueprint, render_template, request, current_app, url_for
from werkzeug.utils import secure_filename

from .utils import allowed_file, unique_name, read_image, save_result_image
from .detectors import (
    detect_copy_move,
    compare_images,
    ai_cnn_patch_detector,
    ai_lstm_sequence_detector,
    final_decision,
)


main_bp = Blueprint("main", __name__)

INDEX_TEMPLATE = "index.html"
ALGORITHMS = ["SIFT", "SURF", "AKAZE", "ORB"]
UNSUPPORTED_FILE_MESSAGE = (
    "Desteklenmeyen dosya formatı. jpg, png, webp, gif, bmp, tif kullanılabilir."
)

@main_bp.get("/")
def index():
    return render_template(INDEX_TEMPLATE, error=None, payload=None)


@main_bp.post("/")
def analyze_image():
    return handle_image_analysis_request()
def index():
    if request.method == "POST":
        return handle_image_analysis_request()

    return render_template(INDEX_TEMPLATE, error=None, payload=None)


def handle_image_analysis_request():
    original_file = request.files.get("original_image")
    suspect_file = request.files.get("suspect_image")

    error = validate_uploaded_files(original_file, suspect_file)

    if error:
        return render_template(INDEX_TEMPLATE, error=error, payload=None)

    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    result_dir = Path(current_app.config["RESULT_FOLDER"])

    suspect_name, suspect_img = save_uploaded_image(suspect_file, upload_dir)
    original_name, original_img = save_optional_original_image(original_file, upload_dir)

    results, visual_urls = run_detection_algorithms(
        suspect_name=suspect_name,
        suspect_img=suspect_img,
        original_img=original_img,
        result_dir=result_dir,
    )

    add_ai_results(results, suspect_img)

    payload = build_response_payload(
        suspect_name=suspect_name,
        original_name=original_name,
        original_img=original_img,
        results=results,
        visual_urls=visual_urls,
    )

    return render_template(INDEX_TEMPLATE, error=None, payload=payload)


def validate_uploaded_files(original_file, suspect_file):
    if not suspect_file or suspect_file.filename == "":
        return "Lütfen en az bir şüpheli görüntü yükleyin."

    files_to_validate = get_files_to_validate(original_file, suspect_file)

    for file in files_to_validate:
        if not allowed_file(file.filename):
            return UNSUPPORTED_FILE_MESSAGE

    return None


def get_files_to_validate(original_file, suspect_file):
    files = [suspect_file]

    if original_file and original_file.filename:
        files.append(original_file)

    return files


def save_uploaded_image(file, upload_dir):
    file_name = unique_name(secure_filename(file.filename))
    file_path = upload_dir / file_name

    file.save(file_path)

    image = read_image(str(file_path))

    return file_name, image


def save_optional_original_image(original_file, upload_dir):
    if not original_file or not original_file.filename:
        return None, None

    return save_uploaded_image(original_file, upload_dir)


def run_detection_algorithms(suspect_name, suspect_img, original_img, result_dir):
    results = []
    visual_urls = []

    for algorithm in ALGORITHMS:
        result, visual = run_single_detection(
            algorithm=algorithm,
            suspect_img=suspect_img,
            original_img=original_img,
        )

        visual_url = save_detection_visual(
            suspect_name=suspect_name,
            algorithm=algorithm,
            result=result,
            visual=visual,
            result_dir=result_dir,
        )

        results.append(result)
        visual_urls.append(visual_url)

    return results, visual_urls


def run_single_detection(algorithm, suspect_img, original_img):
    if original_img is not None:
        return compare_images(original_img, suspect_img, algorithm)

    return detect_copy_move(suspect_img, algorithm)


def save_detection_visual(suspect_name, algorithm, result, visual, result_dir):
    out_name = create_result_file_name(suspect_name, algorithm)

    save_result_image(str(result_dir / out_name), visual)

    return {
        "algorithm": result["algorithm"],
        "url": url_for("static", filename=f"results/{out_name}"),
    }


def create_result_file_name(suspect_name, algorithm):
    suspect_stem = Path(suspect_name).stem
    algorithm_name = algorithm.lower()

    return f"{suspect_stem}_{algorithm_name}_result.jpg"


def add_ai_results(results, suspect_img):
    ai_cnn_result = ai_cnn_patch_detector(suspect_img)
    ai_lstm_result = ai_lstm_sequence_detector(suspect_img)

    results.extend([ai_cnn_result, ai_lstm_result])


def build_response_payload(
    suspect_name,
    original_name,
    original_img,
    results,
    visual_urls,
):
    return {
        "suspect_url": url_for("static", filename=f"uploads/{suspect_name}"),
        "original_url": build_original_url(original_name),
        "mode": get_analysis_mode(original_img),
        "results": results,
        "visuals": visual_urls,
        "decision": final_decision(results),
    }


def build_original_url(original_name):
    if not original_name:
        return None

    return url_for("static", filename=f"uploads/{original_name}")


def get_analysis_mode(original_img):
    if original_img is not None:
        return "Orijinal + Şüpheli Karşılaştırma"

    return "Tek Görsel Copy-Move Analizi"