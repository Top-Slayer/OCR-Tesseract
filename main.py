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
if os.path.exists("working_folders"):
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
        os.makedirs(os.path.join(des_path, sub_dir, in_sub), exist_ok=True)
        subprocess.run([
            "Real-ESRGAN/realesrgan-ncnn-vulkan.exe",
            "-i", os.path.join(path, sub_dir, in_sub),
            "-o", os.path.join(des_path, sub_dir, in_sub),
            "-n", "realesrgan-x4plus",
        ])


# Get text from enchanced image
count = 0
succ_count = 0

langs = "kor+eng+ind"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

for sub_dir in os.listdir(path):
    print(f"\nsub_dir: {sub_dir}")

    for in_sub in os.listdir(os.path.join(path, sub_dir)):
        print(f"\nin_sub: {in_sub}")
        in_sub = os.path.join(path, sub_dir, in_sub)

        for img in os.listdir(in_sub):
            image = cv2.imread(os.path.join(in_sub, img))
            # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            text = pytesseract.image_to_string(image, lang=langs, config="r'--oem 3 --psm 7'")
            text = text.replace("\n", '')

            print(f"{img}: ", end='')
            print(text)

            if len(text) != 0:
                succ_count += 1
            count += 1

    print("-" * 50)

print(f"Result: {succ_count}/{count} can read {succ_count * 100 / count :.2f}% in this comic")