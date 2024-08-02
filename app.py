import streamlit as st
import numpy as np
import cv2

def split_and_merge(image):
    img_array = np.array(image)

    if img_array is None:
        raise ValueError("Failed to load the image. Please make sure the image file is valid.")

    def should_split(region):
        if region is None or region.size == 0:
            return False
        std_dev = np.std(region)
        return std_dev > 30

    def split_merge(img):
        if should_split(img):
            h, w = img.shape[:2]
            if h <= 1 or w <= 1:
                return
            mid_h, mid_w = h // 2, w // 2
            quadrants = [
                img[:mid_h, :mid_w],
                img[:mid_h, mid_w:],
                img[mid_h:, :mid_w],
                img[mid_h:, mid_w:]
            ]
            for quadrant in quadrants:
                split_merge(quadrant)
        else:
            avg_color = np.mean(img)
            img[:, :] = avg_color

    split_merge(img_array)
    return img_array

def main():
    st.title("Image Segmentation Tool")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", channels="BGR", use_column_width=True)

        if st.button("Segment Image"):
            try:
                segmented_image = split_and_merge(image)
                st.image(segmented_image, caption="Segmented Image", channels="BGR", use_column_width=True)
            except ValueError as e:
                st.error(str(e))

if __name__ == "__main__":
    main()
