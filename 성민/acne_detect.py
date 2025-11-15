import cv2
import numpy as np

# ---------------------------------------------------------
# 1. 피부 영역 추출 (BGR → YCrCb)
# ---------------------------------------------------------
def get_skin_mask_ycrcb(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # 일반적인 피부색 범위 (약간 넉넉하게)
    cr_min, cr_max = 133, 173
    cb_min, cb_max = 77, 127

    skin_mask = cv2.inRange(ycrcb, (0, cr_min, cb_min), (255, cr_max, cb_max))

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)

    return skin_mask


# ---------------------------------------------------------
# 2. HSV 기반 붉은색 후보 탐지 (Hue histogram peak)
# ---------------------------------------------------------
def get_red_candidate_mask(img, skin_mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 피부 영역 내 H 값만 분석
    h_skin = h[skin_mask > 0]

    # Hue 히스토그램
    hist = cv2.calcHist([h_skin], [0], None, [180], [0, 180])
    red_peak = int(np.argmax(hist[:20]))  # 0~20 사이에서 피크

    low = max(red_peak - 10, 0)
    high = min(red_peak + 10, 20)

    print(f"[DEBUG] red_peak = {red_peak}, range=({low}, {high})")

    # 후보 마스크
    red_mask = cv2.inRange(h, low, high)

    # 피부 영역과 AND
    red_mask = cv2.bitwise_and(red_mask, skin_mask)

    # denoise
    red_mask = cv2.medianBlur(red_mask, 5)

    return red_mask


# ---------------------------------------------------------
# 3. 컨투어 기반 여드름 탐지
# ---------------------------------------------------------
def detect_acne_contours(img, candidate_mask):
    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    acne_boxes = []
    acne_centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 3000:
            continue

        # 원형도
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.25:
            continue

        # 경계 대비 검사 (Lab a* difference)
        x, y, w, h = cv2.boundingRect(cnt)
        roi = img[y:y+h, x:x+w]

        expand = 5
        x1 = max(x - expand, 0)
        y1 = max(y - expand, 0)
        x2 = min(x + w + expand, img.shape[1])
        y2 = min(y + h + expand, img.shape[0])
        outer_roi = img[y1:y2, x1:x2]

        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
        outer_lab = cv2.cvtColor(outer_roi, cv2.COLOR_BGR2Lab)

        inside_a = np.mean(roi_lab[:, :, 1])
        outside_a = np.mean(outer_lab[:, :, 1])

        contrast = inside_a - outside_a
        if contrast < 2:
            continue

        acne_boxes.append((x, y, w, h))
        acne_centers.append((x + w//2, y + h//2))

    return acne_boxes, acne_centers


# ---------------------------------------------------------
# 4. 이미지에 박스 그리기
# ---------------------------------------------------------
def draw_results(img, acne_boxes):
    out = img.copy()
    for (x, y, w, h) in acne_boxes:
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.putText(out, f"Acne Count: {len(acne_boxes)}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)
    return out
