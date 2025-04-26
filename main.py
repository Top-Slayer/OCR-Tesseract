import pytesseract
from PIL import Image
import os

langs = "chi_sim+chi_sim_vert+chi_tra+eng+ind+jpn+jpn_vert+kor"
path = "pages"

files = os.listdir(path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

for i, f in enumerate(files):
    img = Image.open(os.path.join(path, f))
    text = pytesseract.image_to_string(img, lang=langs)

    if text != "":
        print(f"Page: {i}")
        print(f"{'=' * 50} Top Page: {i} {'=' * 50}")
        print(text)
        print(f"{'=' * 50} Down Page: {i} {'=' * 50}")
        print(f"-" * 114)
    else:
        print(f"Page: {i} Nothing")
        print(f"-" * 114)
