import sys
sys.path.append("CRAFT-pytorch/")

import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
import craft_utils
import imgproc

from craft import CRAFT
import file_utils

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


net = CRAFT()

print('Loading weights from checkpoint ( craft_mlt_25k.pth )\n')
net.load_state_dict(copyStateDict(torch.load("craft_mlt_25k.pth", map_location='cpu')))
net.eval()

refine_net = None

def _crop_polygons(image, polys):
    cropped_imgs = []

    for poly in polys:
        poly = np.array(poly).astype(np.int32)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        x, y, w, h = cv2.boundingRect(poly)

        masked_image = cv2.bitwise_and(image, image, mask=mask)
        cropped = masked_image[y:y+h, x:x+w]

        crop_mask = mask[y:y+h, x:x+w]
        result = cv2.bitwise_and(cropped, cropped, mask=crop_mask)
        cropped_imgs.append(result)

    return cropped_imgs


def filter_image(path: str):
    for sub_dir in os.listdir(path):
        print(f"Directory: [ {sub_dir} ]")
        full_sub_dir = os.path.join(path, sub_dir)
        max_file = len(os.listdir(full_sub_dir))

        for k, image_path in enumerate(os.listdir(full_sub_dir)):
            des_path = f"working_folders/cropped_imgs/{sub_dir}/{k}"

            image_path = os.path.join(full_sub_dir, image_path)
            print("Filtering text in image {:d}/{:d}: {:s}".format(k+1, max_file, image_path), end='')

            image = imgproc.loadImage(image_path)
            bboxes, polys, _ = test_net(net, image, 0.9, 0.005, 0.4, False, False, refine_net)

            if bboxes is None or len(bboxes) == 0:
                print(" >> [ Remove ]")
                os.remove(image_path)
            else:
                print(" >> [ Write ]")
                os.makedirs(des_path , exist_ok=True)
                for i, img in enumerate(_crop_polygons(image, polys)):
                    cv2.imwrite(f"{des_path}/crop_{i}.png", img)

        print()

    # for i, img in enumerate(cropped_imgs):
    #     cv2.imwrite(f"results/crop_{i}.png", img)