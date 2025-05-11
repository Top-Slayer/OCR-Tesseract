import cv2
import numpy as np
import os

count = 0

def detect_bubble_shapes(image_path, output_dir):
    global count
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to detect white bubbles
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    # Morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 1000 and w/h < 3:  # Filter size & aspect ratio
            bubble = img[y:y+h, x:x+w]
            cv2.imwrite(f"{output_dir}/bubble_{count:03}.png", bubble)
            count += 1

    print(f"Extracted {count} bubble candidates to {output_dir}")


def _split_on_black_panel(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
    horizontal_projection = np.sum(thresh == 255, axis=1)

    black_rows = np.where(horizontal_projection > img.shape[1] * 0.95)[0]
    if len(black_rows) == 0:
        return [img]  # No black panel found, return original

    # Find largest contiguous black region
    splits = np.split(black_rows, np.where(np.diff(black_rows) > 5)[0] + 1)
    max_split = max(splits, key=len)
    y1 = max_split[0]
    y2 = max_split[-1]

    # Crop top and bottom
    parts = []
    if y1 > 10:
        parts.append(img[:y1, :])
    if img.shape[0] - y2 > 10:
        parts.append(img[y2+1:, :])

    return parts

def _detect_and_split_manhwa(image, output_dir='working_folders/output_slices', min_gap=250, white_threshold=0.99, prefix='slice', start_index=1):
    if isinstance(image, str):  # Allow image path
        color_img = cv2.imread(image)
    else:
        color_img = image
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    os.makedirs(output_dir, exist_ok=True)
    _, thresh = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY)
    horizontal_projection = np.sum(thresh == 255, axis=1)
    gap_indices = [i for i, val in enumerate(horizontal_projection) if val > gray_img.shape[1] * 0.98]

    refined_gaps = []
    prev = -min_gap * 2
    for idx in gap_indices:
        if idx - prev >= min_gap:
            refined_gaps.append(idx)
            prev = idx
    refined_gaps = [0] + refined_gaps + [gray_img.shape[0]]

    count = 0
    for i in range(len(refined_gaps) - 1):
        top = refined_gaps[i]
        bottom = refined_gaps[i + 1]
        slice_img = color_img[top:bottom, :]

        if slice_img.shape[0] == 0 or slice_img.shape[1] == 0:
            continue 

        slice_gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
        white_ratio = np.sum(slice_gray > 240) / slice_gray.size

        if white_ratio >= white_threshold:
            continue
        output_path = os.path.join(output_dir, f'{prefix}_{start_index + count:03}.png')
        cv2.imwrite(output_path, slice_img)
        count += 1

    return count

def process_manhwa(image_path, name, output_dir='working_folders/output_slices'):
    text, _ = os.path.splitext(name)
    sub_dir = f"{output_dir}/{text}"
    panels = _split_on_black_panel(image_path)
    total = 0

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    for i, panel in enumerate(panels):
        count = _detect_and_split_manhwa(panel, output_dir=sub_dir,  prefix='slice', start_index=total + 1)
        total += count

    print(f"Saved {total} slices to '{sub_dir}' from {image_path} (blank filtered).")