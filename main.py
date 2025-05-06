import pytesseract
import split_image as si
import os
import cv2
import subprocess
import shutil

# langs = "kor+chi_sim+chi_sim_vert+chi_tra+eng+ind+jpn+jpn_vert"
langs = "kor+eng+ind"
path = "test_pages"
pages = os.listdir(path)


def check_dir(*path: str):
    for _, e in enumerate(path):
        if not os.path.exists(e):
            os.makedirs(e, exist_ok=True)


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Clear all files and recreate
shutil.rmtree("working_folders")
check_dir("working_folders/output_slices", "working_folders/bubbles", "working_folders/enchanced_images")

# Split scene
for page_file in pages:
    img = os.path.join(path, page_file)
    si.process_manhwa(img)

# Enchanced image
path = "working_folders/output_slices"
des_path = "working_folders/enchanced_images"
for sub_dir in os.listdir(path):
    os.makedirs(os.path.join(des_path, sub_dir), exist_ok=True)
    subprocess.run([
        "Real-ESRGAN/realesrgan-ncnn-vulkan.exe",
        "-i", os.path.join(path, sub_dir),
        "-o", os.path.join(des_path, sub_dir),
        "-n", "realesrgan-x4plus"
    ])

# Filter get only image have text
for sub_dir in os.listdir(path):
    sub_dir = os.path.join(path, sub_dir)
    for f in os.listdir(sub_dir):
        path_f = os.path.join(sub_dir, f)
        text = pytesseract.image_to_string(path_f, lang=langs)
        print(f"Text amount: {len(text.strip().split())} from {path_f}")

        if not len(text.strip().split()) > 0:
            os.remove(path_f)


# for sub_dir in os.listdir(path):
#     for f in os.listdir(os.path.join(path, sub_dir)):
#         si.detect_bubble_shapes(os.path.join(path, sub_dir, f), output_dir=os.path.join('working_folders/bubbles', sub_dir))