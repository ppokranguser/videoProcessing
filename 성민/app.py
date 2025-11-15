import streamlit as st
import cv2
import numpy as np

from acne_detect import (
    get_skin_mask_ycrcb,
    get_red_candidate_mask,
    detect_acne_contours,
    draw_results
)

# 페이지 기본 설정
st.set_page_config(page_title="여드름 탐지기", layout="centered")
st.title("여드름 탐지기")
st.write("이미지를 업로드하면 자동으로 여드름을 탐지합니다.")

# 이미지 업로드
uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded:
    # 파일 읽기
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.subheader("원본 이미지")
    st.image(img[:, :, ::-1], channels="RGB")

    # 여드름 탐지 파이프라인
    skin_mask = get_skin_mask_ycrcb(img)
    red_mask = get_red_candidate_mask(img, skin_mask)
    acne_boxes, centers = detect_acne_contours(img, red_mask)
    result_img = draw_results(img, acne_boxes)

    st.subheader("탐지 결과")
    st.image(result_img[:, :, ::-1], channels="RGB")

    st.success(f"🔥 총 탐지된 여드름 개수: {len(acne_boxes)}")

    # 중간 단계 표시
    with st.expander("🔍 중간 결과 보기"):
        st.write("🟫 피부 마스크 (YCrCb)")
        st.image(skin_mask, clamp=True)

        st.write("🔴 붉은 후보 마스크 (HSV)")
        st.image(red_mask, clamp=True)

else:
    st.info("좌측 또는 위쪽의 '파일 업로드' 버튼을 눌러 이미지를 업로드하세요.")
