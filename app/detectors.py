import cv2
import numpy as np
from collections import Counter
from .utils import resize_for_speed, to_percent


def _gray(img):
    return cv2.cvtColor(resize_for_speed(img), cv2.COLOR_BGR2GRAY)


def _feature_detector(name: str):
    name = name.upper()

    if name == "SIFT" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=2500), cv2.NORM_L2, "SIFT"

    if name == "SURF":
        return _surf_detector()

    if name == "AKAZE":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING, "AKAZE"

    return cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING, "ORB"


def _surf_detector():
    try:
        detector = cv2.xfeatures2d.SURF_create(400)
        return detector, cv2.NORM_L2, "SURF"
    except Exception:
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=2500), cv2.NORM_L2, "SURF (SIFT fallback)"

        return cv2.ORB_create(nfeatures=2500), cv2.NORM_HAMMING, "SURF (ORB fallback)"


def _score_from_metrics(match_count, cluster_count, changed_ratio=0.0, inlier_ratio=0.0):
    score = 0.0
    score += min(match_count / 80.0, 1.0) * 0.25
    score += min(cluster_count / 18.0, 1.0) * 0.35
    score += min(changed_ratio / 0.18, 1.0) * 0.25

    if inlier_ratio:
        score += max(0.0, 1.0 - inlier_ratio) * 0.15

    return float(np.clip(score, 0, 1))


def _empty_copy_move_result(label, keypoints, gray):
    return {
        "algorithm": label,
        "tampered": False,
        "confidence": 0.0,
        "keypoints": len(keypoints) if keypoints else 0,
        "matches": 0,
        "cluster_matches": 0,
        "message": "Yeterli ayırt edici nokta bulunamadı."
    }, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _find_copy_move_matches(raw_matches, keypoints, gray_shape):
    good_matches = []

    for group in raw_matches:
        matched = _get_valid_copy_move_match(group, keypoints, gray_shape)

        if matched is not None:
            good_matches.append(matched)

    return good_matches


def _get_valid_copy_move_match(group, keypoints, gray_shape):
    for match in group[1:]:
        p1 = np.array(keypoints[match.queryIdx].pt)
        p2 = np.array(keypoints[match.trainIdx].pt)
        dist = np.linalg.norm(p1 - p2)

        if 25 < dist < max(gray_shape) * 0.75:
            return match, p1, p2

    return None


def _count_vector_clusters(matches):
    vectors = [_rounded_vector(p1, p2) for _, p1, p2 in matches]
    return Counter(vectors).most_common(1)[0][1] if vectors else 0


def _rounded_vector(p1, p2):
    dx, dy = p2 - p1
    return round(dx / 12) * 12, round(dy / 12) * 12


def _draw_copy_move_matches(gray, matches, tampered):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color = (0, 0, 255) if tampered else (0, 180, 0)

    draw_matches = sorted(matches, key=lambda x: x[0].distance)[:60]

    for _, p1, p2 in draw_matches:
        _draw_match_line(vis, p1, p2, color)

    return vis


def _draw_match_line(vis, p1, p2, color):
    p1 = tuple(np.int32(p1))
    p2 = tuple(np.int32(p2))

    cv2.circle(vis, p1, 4, color, 1)
    cv2.circle(vis, p2, 4, color, 1)
    cv2.line(vis, p1, p2, color, 1)


def detect_copy_move(img, algorithm="SIFT"):
    gray = _gray(img)
    detector, norm, label = _feature_detector(algorithm)
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 8:
        return _empty_copy_move_result(label, keypoints, gray)

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    k = min(3, max(2, len(descriptors)))
    raw_matches = matcher.knnMatch(descriptors, descriptors, k=k)

    good_matches = _find_copy_move_matches(raw_matches, keypoints, gray.shape)
    cluster_count = _count_vector_clusters(good_matches)

    confidence = _score_from_metrics(len(good_matches), cluster_count)
    tampered = confidence >= 0.45 and cluster_count >= 8

    vis = _draw_copy_move_matches(gray, good_matches, tampered)

    return {
        "algorithm": label,
        "tampered": bool(tampered),
        "confidence": to_percent(confidence),
        "keypoints": len(keypoints),
        "matches": len(good_matches),
        "cluster_matches": int(cluster_count),
        "message": "Kopyala-yapıştır/sahtecilik izi bulundu." if tampered else "Güçlü sahtecilik izi bulunmadı."
    }, vis


def _prepare_compare_images(original, suspect):
    img1 = resize_for_speed(original)
    img2 = resize_for_speed(suspect)

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return img1, img2, g1, g2


def _find_good_matches(des1, des2, norm):
    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(des1, des2, k=2)

    return [pair[0] for pair in pairs if _is_good_match_pair(pair)]


def _is_good_match_pair(pair):
    return len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance


def _align_suspect_image(img1, img2, kp1, kp2, good_matches):
    aligned = img2.copy()
    inlier_ratio = 0.0

    if len(good_matches) < 8:
        return aligned, inlier_ratio

    src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)

    if homography is None:
        return aligned, inlier_ratio

    aligned = cv2.warpPerspective(img2, homography, (img1.shape[1], img1.shape[0]))
    inlier_ratio = _calculate_inlier_ratio(mask)

    return aligned, inlier_ratio


def _calculate_inlier_ratio(mask):
    if mask is None:
        return 0.0

    return float(mask.sum() / len(mask))


def _match_and_align_images(img1, img2, g1, g2, algorithm):
    detector, norm, label = _feature_detector(algorithm)
    kp1, des1 = detector.detectAndCompute(g1, None)
    kp2, des2 = detector.detectAndCompute(g2, None)

    if _has_not_enough_descriptors(des1, des2, kp1, kp2):
        return img2.copy(), 0.0, 0, label

    good_matches = _find_good_matches(des1, des2, norm)
    aligned, inlier_ratio = _align_suspect_image(img1, img2, kp1, kp2, good_matches)

    return aligned, inlier_ratio, len(good_matches), label


def _has_not_enough_descriptors(des1, des2, kp1, kp2):
    return des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8


def _resize_aligned_if_needed(aligned, img1):
    if aligned.shape[:2] == img1.shape[:2]:
        return aligned

    return cv2.resize(aligned, (img1.shape[1], img1.shape[0]))


def _create_difference_mask(img1, aligned):
    diff = cv2.absdiff(img1, aligned)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, mask = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def _create_difference_overlay(img1, mask):
    heat = img1.copy()
    heat[mask > 0] = (0, 0, 255)

    return cv2.addWeighted(img1, 0.65, heat, 0.35, 0)


def compare_images(original, suspect, algorithm="SIFT"):
    img1, img2, g1, g2 = _prepare_compare_images(original, suspect)

    aligned, inlier_ratio, match_count, label = _match_and_align_images(
        img1,
        img2,
        g1,
        g2,
        algorithm
    )

    aligned = _resize_aligned_if_needed(aligned, img1)

    mask = _create_difference_mask(img1, aligned)
    changed_ratio = float(np.count_nonzero(mask) / mask.size)

    confidence = _score_from_metrics(match_count, 0, changed_ratio, inlier_ratio)
    tampered = changed_ratio >= 0.025 or confidence >= 0.42

    overlay = _create_difference_overlay(img1, mask)

    return {
        "algorithm": label,
        "tampered": bool(tampered),
        "confidence": to_percent(confidence),
        "matches": int(match_count),
        "inlier_ratio": to_percent(inlier_ratio),
        "changed_area": to_percent(changed_ratio),
        "message": "İki görüntü arasında değişiklik tespit edildi." if tampered else "Belirgin değişiklik tespit edilmedi."
    }, overlay


def _extract_patch_scores(gray, patch=32):
    h, w = gray.shape
    scores = []

    for y in range(0, h - patch + 1, patch):
        for x in range(0, w - patch + 1, patch):
            roi = gray[y:y + patch, x:x + patch]
            scores.append(_patch_feature_score(roi))

    return scores


def _patch_feature_score(roi):
    lap = cv2.Laplacian(roi, cv2.CV_64F).var()
    edges = cv2.Canny(roi, 80, 160).mean()

    return [lap, edges, roi.std()]


def ai_cnn_patch_detector(img):
    """CNN mantığına benzer patch tabanlı anomali skoru: kenar/gürültü/doku bloklarını inceler."""
    small = resize_for_speed(img, 768)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    scores = _extract_patch_scores(gray)

    if not scores:
        return {
            "algorithm": "AI-CNN",
            "tampered": False,
            "confidence": 0.0,
            "message": "Görüntü çok küçük."
        }

    arr = np.array(scores, dtype=np.float32)
    z = np.abs((arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-6))

    anomaly_ratio = float((z.max(axis=1) > 2.6).mean())
    confidence = float(np.clip(anomaly_ratio / 0.18, 0, 1))

    return {
        "algorithm": "AI-CNN",
        "tampered": bool(confidence >= 0.45),
        "confidence": to_percent(confidence),
        "anomaly_ratio": to_percent(anomaly_ratio),
        "message": "Patch tabanlı yapay zeka analizi şüpheli alanlar buldu." if confidence >= 0.45 else "AI-CNN analizinde güçlü şüphe bulunmadı."
    }


def _extract_row_features(gray, stripe=16):
    row_features = []

    for y in range(0, gray.shape[0] - stripe + 1, stripe):
        roi = gray[y:y + stripe, :]
        row_features.append(_sequence_feature_score(roi))

    return row_features


def _extract_col_features(gray, stripe=16):
    col_features = []

    for x in range(0, gray.shape[1] - stripe + 1, stripe):
        roi = gray[:, x:x + stripe]
        col_features.append(_sequence_feature_score(roi))

    return col_features


def _sequence_feature_score(roi):
    return [
        roi.mean(),
        roi.std(),
        cv2.Laplacian(roi, cv2.CV_64F).var()
    ]


def ai_lstm_sequence_detector(img):
    """LSTM mantığına benzer sıralı satır/sütun tutarlılığı analizi yapar."""
    small = resize_for_speed(img, 768)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    row_features = _extract_row_features(gray)
    col_features = _extract_col_features(gray)

    seq = np.vstack([row_features, col_features]).astype(np.float32)

    if len(seq) < 4:
        return {
            "algorithm": "AI-LSTM",
            "tampered": False,
            "confidence": 0.0,
            "message": "Görüntü çok küçük."
        }

    diffs = np.linalg.norm(np.diff(seq, axis=0), axis=1)
    z = np.abs((diffs - diffs.mean()) / (diffs.std() + 1e-6))

    anomaly_ratio = float((z > 2.4).mean())
    confidence = float(np.clip(anomaly_ratio / 0.16, 0, 1))

    return {
        "algorithm": "AI-LSTM",
        "tampered": bool(confidence >= 0.45),
        "confidence": to_percent(confidence),
        "sequence_anomaly": to_percent(anomaly_ratio),
        "message": "Sıralı tutarlılık analizinde değişiklik şüphesi var." if confidence >= 0.45 else "AI-LSTM analizinde güçlü şüphe bulunmadı."
    }


def final_decision(results):
    numeric = [r.get("confidence", 0) for r in results]
    positives = sum(1 for r in results if r.get("tampered"))

    avg = round(float(np.mean(numeric)) if numeric else 0.0, 2)
    final = positives >= max(2, len(results) // 2)

    return {
        "tampered": bool(final),
        "confidence": avg,
        "positive_algorithms": positives,
        "total_algorithms": len(results),
        "label": "GÖRÜNTÜ DEĞİŞTİRİLMİŞ / ŞÜPHELİ" if final else "GÖRÜNTÜ TEMİZ GÖRÜNÜYOR"
    }