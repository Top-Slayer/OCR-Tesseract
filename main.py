import patch_file
import split_image as si
import os, sys
import subprocess
import threading
import shutil
import detect_text as dt
import customtkinter as ctk

path = "pages"
os.makedirs(path, exist_ok=True)

pages = os.listdir(path)
dt.langs = 'eng'

def _check_dir(*path: str):
    for _, e in enumerate(path):
        if not os.path.exists(e):
            os.makedirs(e, exist_ok=True)


def process(path: str, langs=""):
    if langs == "" and len(langs) == 0:
        print("Using default language: ", dt.langs)
    else:
        dt.langs = langs
        print("Using custom language: ", dt.langs)

    # Clear all files and recreate
    if os.path.exists("working_folders"):
        shutil.rmtree("working_folders")
    _check_dir(
        "working_folders/output_slices", 
        "working_folders/2_Scale", 
        "working_folders/enchanced_images",
        "working_folders/cropped_imgs"
    )

    # Split scene
    print("[ Slice Part ]")
    for page_file in pages:
        img = os.path.join(path, page_file)
        si.process_manhwa(img)

    path = "working_folders/output_slices"

    # Filter get only image have text in scene
    print("\n[ Filter Crop Text Part ]")
    dt.filter_image(path)

    # Enchanced image
    print("\n[ Enchanced image Part ]")
    path = "working_folders/cropped_imgs"
    des_path = "working_folders/enchanced_images"

    for sub_dir in os.listdir(path):
        for in_sub in os.listdir(os.path.join(path, sub_dir)):
            os.makedirs(os.path.join(des_path, sub_dir, in_sub), exist_ok=True)
            proc = subprocess.run([
                    "Real-ESRGAN/realesrgan-ncnn-vulkan.exe",
                    "-i", os.path.join(path, sub_dir, in_sub),
                    "-o", os.path.join(des_path, sub_dir, in_sub),
                    "-n", "realesrgan-x4plus",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()

    # Convert image RGB to 2-Scale image
    print("\n[ Convert RGP to 2-Scale Part ]")
    dt.cvt2Scale("working_folders/enchanced_images")

    # Get text from enchanced image
    print("\n[ Get Text From Image Part ]")
    dt.extract_text("working_folders/2_Scale")


class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert("end", string)
        self.text_widget.see("end")  # auto-scroll

    def flush(self):
        pass  # needed for compatibility


button_status = {
    "Korean": {"active": False, "symbol": "kor"},
    "Japan": {"active": False, "symbol": "jpn+jpn_vert"},
    "English": {"active": False, "symbol": "eng"},
    "Indonesia": {"active": False, "symbol": "ind"},
    "China": {"active": False, "symbol": "chi_sim+chi_sim_vert+chi_tra"}
}

button_lists = list(button_status.keys())

def toggle_status(idx, button_id):
    button_status[button_id]["active"] = not button_status[button_id]["active"]
    new_text = f"{button_lists[idx]}: ON" if button_status[button_id]["active"] else f"{button_lists[idx]}: OFF"
    buttons[button_id].configure(text=new_text)


ctk.set_appearance_mode("Dark")       # Options: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")   # Options: "blue", "green", "dark-blue"

app = ctk.CTk()  
app.geometry("700x500")
app.title("Optical Character Recognition")

label = ctk.CTkLabel(app, text="Burh OCR Program", font=("Arial", 25))
label.pack(pady=20)

label = ctk.CTkLabel(app, text="[ Warning ]: if use all languages it will take time")
label.pack()

label = ctk.CTkLabel(app, text="Select language: ")
label.pack()

button_frame = ctk.CTkFrame(app)
button_frame.pack(pady=20, padx=20)

buttons = {}
columns = 3  # number of columns in the grid

for idx, button_id in enumerate(button_status):
    row = idx // columns
    col = idx % columns
    buttons[button_id] = ctk.CTkButton(
        button_frame,
        text=f"{button_lists[idx]}: OFF",
        command=lambda button_id=button_id, idx=idx: toggle_status(idx, button_id)
    )
    buttons[button_id].grid(row=row, column=col, padx=10, pady=10)


def list_used_lang():
    res = ""
    for idx, button_id in enumerate(button_status):
        if button_status[button_id]["active"]:
            if idx != 0:
                res += "+" + str(button_status[button_id]["symbol"])
            else:
                res += str(button_status[button_id]["symbol"])
    return res

def threaded_process():
    threading.Thread(target=process, args=(path, list_used_lang()), daemon=True).start()


label = ctk.CTkLabel(app, text="Don't forget put images into [ pages ] folder")
label.pack()

button = ctk.CTkButton(app, text="Start", command=threaded_process)
button.pack()

output_box = ctk.CTkTextbox(app, width=680, height=200, font=("Arial", 10))
output_box.pack(pady=20, padx=20, fill="both", expand=True)
sys.stdout = RedirectText(output_box)

# Run app
app.mainloop()
