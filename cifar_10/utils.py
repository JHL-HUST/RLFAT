import numpy as np
import random
import cv2
import skimage as sk


def save_as_gray_image(img, filename, percentile=99):
    img_2d = np.sum(np.abs(img), axis=2)
    vmax = np.percentile(img_2d, percentile)
    vmin = np.min(img_2d)
    img_2d = np.clip((img_2d - vmin) / (vmax - vmin), 0, 1)
    cv2.imwrite(filename, img_2d * 255)


def brightness(x, c=0.15):
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1)


def batch_brightness(x, c=0.15):
    return np.array([brightness(xi, c) for xi in x])


def rbs_transformation(x, splited_block):

    clip_num = splited_block - 1
    _, w, h, c = x.shape
    result = np.zeros_like(x, dtype=np.float32)
    clip_index = list(range(clip_num+1))

    clip_width_points = np.random.randint(low=0, high=w, size=[clip_num], dtype=np.int32).tolist()
    clip_width_points.append(w)
    clip_width_points.sort()

    clip_high_points = np.random.randint(low=0, high=h, size=[clip_num], dtype=np.int32).tolist()
    clip_high_points.append(h)
    clip_high_points.sort()

    random.shuffle(clip_index)

    now_index = 0
    for ind, index in enumerate(clip_index):
        if index == 0:
            value = clip_width_points[index] + now_index
            result[:, now_index:value, :, :] = x[:, 0:clip_width_points[index], :, :]
            now_index = value
        else:
            value = clip_width_points[index] - clip_width_points[index-1] + now_index
            result[:, now_index:value, :, :] = x[:, clip_width_points[index-1]:clip_width_points[index], :, :]
            now_index = value

    random.shuffle(clip_index)
    x = np.copy(result)

    now_index = 0
    for ind, index in enumerate(clip_index):
        if index == 0:
            value = clip_high_points[index] + now_index
            result[:, :, now_index:value, :] = x[:, :, 0:clip_high_points[index],:]
            now_index = value
        else:
            value = clip_high_points[index] - clip_high_points[index - 1] + now_index
            result[:, :, now_index:value, :] = x[:, :, clip_high_points[index - 1]:clip_high_points[index], :]
            now_index = value

    return result
