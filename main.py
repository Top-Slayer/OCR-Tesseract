import pytesseract
import split_image as si
import os
import cv2
import subprocess
import shutil
import detect_text as dt

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
check_dir(
    "working_folders/output_slices", 
    "working_folders/bubbles", 
    "working_folders/enchanced_images",
    "working_folders/cropped_imgs"
)

# Split scene
print("[ Slice Parts ]")
for page_file in pages:
    img = os.path.join(path, page_file)
    si.process_manhwa(img)


path = "working_folders/output_slices"

# Filter get only image have text in scene
print("\n[ Filter Crop Text Parts ]")
dt.filter_image(path)


# Crop bubble text
# print("\n[ Crop Bubble Text Parts ]")
# des_path = "working_folders/bubbles"

# for sub_dir in os.listdir(path):
#     for file in os.listdir(os.path.join(path, sub_dir)):
#         si.detect_bubble_shapes(os.path.join(path, sub_dir, file), os.path.join(des_path, sub_dir))


# Filter get only image have bubble text
# print("\n[ Filter bubble text Parts ]")
# dt.filter_image("working_folders/bubbles")


# Convert image RGB to Gray scale
# for sub_dir in os.listdir(path):
#     sub_dir = os.path.join(path, sub_dir)
#     for path_img in os.listdir(sub_dir):
#         img = cv2.imread(os.path.join(sub_dir, path_img))
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite(os.path.join(sub_dir, path_img), gray)


# Enchanced image
path = "working_folders/cropped_imgs"
des_path = "working_folders/enchanced_images"

for sub_dir in os.listdir(path):
    for in_sub in os.listdir(os.path.join(path, sub_dir)):
        print(in_sub)

# for sub_dir in os.listdir(path):
#     os.makedirs(os.path.join(des_path, sub_dir), exist_ok=True)
#     subprocess.run([
#         "Real-ESRGAN/realesrgan-ncnn-vulkan.exe",
#         "-i", os.path.join(path, sub_dir),
#         "-o", os.path.join(des_path, sub_dir),
#         "-n", "realesrgan-x4plus"
#     ])

