import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pandas as pd


def read_file():
    current_directory = os.path.dirname(__file__)
    relative_path = os.path.join("..", "..", "static", "src_image",
                                 "sample.jpg")
    image_path = os.path.abspath(os.path.join(current_directory, relative_path))

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def resizing(img):
    height, width, channel = img.shape

    if width > 888:
        ar = width / height
        new_width = 888
        new_height = int(new_width / ar)

        img = cv2.resize(img, (new_width, new_height),
                     interpolation=cv2.INTER_LINEAR)
    return img


def img_enhance(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY_INV)
    return thresh


def img_dilation(img):
    kernel = np.ones((3, 100), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    return dilated


def find_contours(img):
    (contours, heirarchy) = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours,
                                   key=lambda ctr: cv2.boundingRect(ctr)[
                                       1])  
    return sorted_contours_lines


def segment(img, contours):
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 100, 250), 2)

    plt.imshow(img)
    plt.show()


def img_segment(img, sorted_contours_lines):
    current_directory = os.path.dirname(__file__)
    relative_path = os.path.join("..", "handwritten_to_digit",
                                 "input_sentences")
    path = os.path.abspath(os.path.join(current_directory, relative_path))

    df = pd.DataFrame()

    for idx, ctr in enumerate(sorted_contours_lines):
        x, y, w, h = cv2.boundingRect(ctr)
        if h > 20:
            roi = img[y:y + h, x:x + w]  
            cv2.imwrite(os.path.join(path, "input_lines_in_jpg", f'roi'f'_{idx}.jpg'), roi)
            df_row = {os.path.join(path, "input_lines_in_jpg", f'roi_{idx}.jpg')}
            df = pd.concat([df, pd.DataFrame([df_row])], ignore_index=True)
            cv2.imshow(f'ROI {idx}', roi)
    cv2.destroyAllWindows()
    print(df)
    new_csv = os.path.join(path, 'input_img_paths.csv')
    df.to_csv(new_csv, index=False)
def start_line_seg():
    img = read_file()
    img_resized = resizing(img)
    img_enhanced = img_enhance(img_resized)
    img_dilated = img_dilation(img_enhanced)
    contours = find_contours(img_dilated)
    img_segment(img_resized, contours)
if __name__ == "__main__":
    start_line_seg()

