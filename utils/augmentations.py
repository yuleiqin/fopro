import os
import sys
import math
import glob
import struct
from PIL import Image, ImageFile
import albumentations as alb
import cv2
import random
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations import functional as F


# def get_record(record_file, offset):
#     with open(record_file, 'rb') as ifs:
#         ifs.seek(offset)
#         byte_len_crc = ifs.read(12)
#         proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
#         pb_data = ifs.read(proto_len)
#         if len(pb_data) < proto_len:
#             print("read pb_data err,proto_len:%s pb_data len:%s" % (proto_len, len(pb_data)))
#             return None
#     example = example_pb2.Example()
#     example.ParseFromString(pb_data)
#     # keep key value in order
#     feature = sorted(example.features.feature.items())
#     for k, f in feature:
#         image_raw = f.bytes_list.value[0]
#         image = cv2.imdecode(np.frombuffer(image_raw, np.uint8), cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return image


# class SlideResize(ImageOnlyTransform):
#     """crop the image into square and resize
#     """
#     def __init__(self, input_size, always_apply=False, p=1.0):
#         super(SlideResize, self).__init__(always_apply, p)
#         self.input_size = input_size

#     def apply(self, img, **params):
#         height, width = img.shape[:2]
#         if height > width:
#             start = random.randint(0, height - width - 1)
#             end = start + width
#             return cv2.resize(img[start:end, :, :], self.input_size)
#         elif height < width:
#             start = random.randint(0, width - height - 1)
#             end = start + height
#             return cv2.resize(img[:, start:end, :], self.input_size)
#         else:
#             return cv2.resize(img, self.input_size)


# class MinResize(ImageOnlyTransform):
#     """resize along the smaller axis"""
#     def __init__(self, input_size, always_apply=False, p=1.0):
#         super(MinResize, self).__init__(always_apply, p)
#         self.input_size = input_size

#     def apply(self, img, **params):
#         height, width = img.shape[:2]
#         if height < width:
#             _width = int(self.input_size[1] * width / height)
#             return cv2.resize(img, (_width, self.input_size[1]))
#         elif height > width:
#             _height = int(self.input_size[0] * height / width)
#             return cv2.resize(img, (self.input_size[0], _height))
#         else:
#             return cv2.resize(img, self.input_size)


# class MaxResize(ImageOnlyTransform):
#     """resize along the bigger axis"""
#     def __init__(self, input_size, always_apply=False, p=1.0):
#         super(MaxResize, self).__init__(always_apply, p)
#         self.input_size = input_size

#     def apply(self, img, **params):
#         height, width = img.shape[:2]
#         if height > width:
#             _width = int(self.input_size[1] * width / height)
#             return cv2.resize(img, (_width, self.input_size[1]))
#         elif height < width:
#             _height = int(self.input_size[0] * height / width)
#             return cv2.resize(img, (self.input_size[0], _height))
#         else:
#             return cv2.resize(img, self.input_size)


# class PadResize(ImageOnlyTransform):
#     """pad the image and resize"""
#     def __init__(self, input_size, always_apply=False, p=1.0):
#         super(PadResize, self).__init__(always_apply, p)
#         self.input_size = input_size

#     def apply(self, img, **params):
#         height, width = img.shape[:2]
#         pad1 = abs(height - width) // 2
#         pad2 = abs(height - width) - pad1
#         if height > width:
#             return cv2.resize(cv2.copyMakeBorder(img, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=(255, 255, 255)),
#                               self.input_size)
#         elif height < width:
#             return cv2.resize(cv2.copyMakeBorder(img, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)),
#                               self.input_size)
#         else:
#             return cv2.resize(img, self.input_size)


# class CenterScaleCrop(ImageOnlyTransform):
#     def __init__(self, input_size, always_apply=False, p=1.0):
#         super(CenterScaleCrop, self).__init__(always_apply, p)
#         self.input_size = input_size

#     def apply(self, img, **params):
#         height, width = img.shape[:2]
#         center = [width // 2, height // 2]
#         length = int(min(center) / random.uniform(1.0, 2.0))
#         cropped_img = img[center[1] - length:center[1] + length, center[0] - length:center[0] + length, :]
#         return cv2.resize(cropped_img, self.input_size)


class RandomFlip(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.random_flip(img, random.choice([0, 1]))


class RandomBorder(ImageOnlyTransform):
    def apply(self, img, **params):
        height, width = img.shape[:2]
        flag = random.random()
        height_pad = int(random.uniform(0.0, 0.1) * height)
        width_pad = int(random.uniform(0.0, 0.1) * width)
        if random.random() < 0.5:
            border_value = (0, 0, 0)  # black
        else:
            border_value = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if flag < 0.15:
            img = cv2.resize(img, (width, height - 2 * height_pad))
            return cv2.copyMakeBorder(img, height_pad, height_pad, 0, 0, cv2.BORDER_CONSTANT, value=border_value)
        elif flag < 0.3:
            img = cv2.resize(img, (width - 2 * width_pad, height))
            return cv2.copyMakeBorder(img, 0, 0, width_pad, width_pad, cv2.BORDER_CONSTANT, value=border_value)
        elif flag < 0.45:
            img = cv2.resize(img, (width - 2 * width_pad, height - 2 * height_pad))
            return cv2.copyMakeBorder(img, height_pad, height_pad, width_pad, width_pad, cv2.BORDER_CONSTANT,
                                      value=border_value)
        else:
            return cv2.resize(img, (width - 2 * width_pad, height - 2 * height_pad))
    
    def get_transform_init_args_names(self):
        return ()


class RandomTranslate(ImageOnlyTransform):
    def apply(self, img, **params):
        height, width = img.shape[:2]
        rotation_mat = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1)
        rotation_mat[0, 2] = random.uniform(- width / 8, width / 8)
        rotation_mat[1, 2] = random.uniform(- height / 8, height / 8)
        # border_value = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        border_value = (0, 0, 0)
        rotated_img = cv2.warpAffine(img, rotation_mat, (width, height), borderValue=border_value)
        return rotated_img
           
    def get_transform_init_args_names(self):
        return ()


# class RandomPropertionScale(ImageOnlyTransform):
#     def apply(self, img, **params):
#         height, width = img.shape[:2]
#         if random.random() < 0.5:
#             scale = random.uniform(0.1, 0.3)
#             height_pad, width_pad = int(scale * height), int(scale * width)
#             height_new, width_new = height - 2 * height_pad, width - 2 * width_pad
#             top, left, bottom, right = height_pad, width_pad, -height_pad, -width_pad
#         else:
#             scale = random.uniform(0.6, 0.8)
#             height_new, width_new = int(scale * height), int(scale * width)
#             top = random.randint(0, height - height_new - 1)
#             left = random.randint(0, width - width_new - 1)
#             bottom = top + height_new
#             right = left + width_new
#         # color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)
#         # new_img = np.multiply(np.ones_like(img), color)
#         # new_img[top:bottom, left:right, :] = cv2.resize(img, (width_new, height_new))
#         new_img = cv2.resize(img, (width_new, height_new))
#         return new_img
    
#     def get_transform_init_args_names(self):
#         return ()


# class RandomRotate(ImageOnlyTransform):
#     def apply(self, img, **params):
#         height, width = img.shape[:2]
#         degree = random.randint(-180, 180)
#         cos_val = math.cos(math.radians(degree))
#         sin_val = math.sin(math.radians(degree))
#         new_height = int(width * abs(sin_val) + height * abs(cos_val))
#         new_width = int(height * abs(sin_val) + width * abs(cos_val))
#         rotation_mat = cv2.getRotationMatrix2D((new_width / 2, new_height / 2), degree, 1)
#         border_value = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#         rotated_img = cv2.warpAffine(img, rotation_mat, (new_width, new_height), borderValue=border_value)
#         rotated_img = cv2.resize(rotated_img, (width, height))
#         return rotated_img


# class RandomPerspective(ImageOnlyTransform):
#     def apply(self, img, **params):
#         height, width = img.shape[:2]
#         src = np.array([(0, 0), (0, height), (width, 0), (width, height)], dtype=np.float32)
#         flag = random.random()
#         dw = random.uniform(- width / 4, width / 4)
#         dh = random.uniform(- height / 4, height / 4)
#         if flag < 0.25:
#             dst = np.array([(dw, dh), (0, height), (width, 0), (width, height)], dtype=np.float32)
#         elif flag < 0.5:
#             dst = np.array([(0, 0), (dw, height - dh), (width, 0), (width, height)], dtype=np.float32)
#         elif flag < 0.75:
#             dst = np.array([(0, 0), (0, height), (width - dw, dh), (width, height)], dtype=np.float32)
#         else:
#             dst = np.array([(0, 0), (0, height), (width, 0), (width - dw, height - dh)], dtype=np.float32)
#         perspective_mat = cv2.getPerspectiveTransform(src, dst)
#         '''
#         if random.random() < 0.5:
#             border_value = (0, 0, 0)
#         else:
#             border_value = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#         '''
#         border_value = (0, 0, 0)
#         perspective_img = cv2.warpPerspective(img, perspective_mat, (width, height), borderValue=border_value)
#         return perspective_img
    
#     def get_transform_init_args_names(self):
#         return ()


class RandomFill(ImageOnlyTransform):
    def apply(self, img, **params):
        height, width = img.shape[:2]
        num = random.randint(2, 8)
        points = [[random.randint(0, height // 2 - 1), random.randint(0, width // 2 - 1)] for _ in range(num)]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        filled_img = cv2.fillPoly(img, [np.array(points)], color)
        return filled_img
    def get_transform_init_args_names(self):
        return ()


class RandomCompression(ImageOnlyTransform):
    def apply(self, img, **params):
        quality = random.randint(20, 50)
        return F.image_compression(img, quality, image_type=".jpg")


class RandomNoise(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.iso_noise(img)


class RandomGridDistortion(ImageOnlyTransform):
    def apply(self, img, **params):
        num_steps = random.randint(3, 10)
        stepsx = [1 + random.uniform(0.1, 0.5) for _ in range(num_steps + 1)]
        stepsy = [1 + random.uniform(0.1, 0.5) for _ in range(num_steps + 1)]
        interpolation = cv2.INTER_LINEAR
        border_mode = cv2.BORDER_REFLECT_101
        return F.grid_distortion(img, num_steps, stepsx, stepsy, interpolation, border_mode, None)


class RandomHueSaturationValue(ImageOnlyTransform):
    def apply(self, img, **params):
        hue_shift = random.uniform(5, 15)
        sat_shift = random.uniform(10, 20)
        val_shift = random.uniform(5, 15)
        return F.shift_hsv(img, hue_shift, sat_shift, val_shift)


class RandomGaussianBlur(ImageOnlyTransform):
    def apply(self, img, **params):
        ksize = random.choice([3, 5, 7, 9])
        sigma = random.uniform(0, 3.)
        return F.gaussian_blur(img, ksize, sigma)


class RandomEqualize(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.equalize(img, mode='cv', by_channels=True)


class RandomFog(ImageOnlyTransform):
    def apply(self, img, **params):
        fog_coef = random.uniform(0.1, 1.0)
        alpha_coef = random.uniform(0.0, 0.5)
        return F.add_fog(img, fog_coef, alpha_coef, [])


class RandomBrightness(ImageOnlyTransform):
    def apply(self, img, **params):
        brightness = random.uniform(0.0, 0.4)
        contrast = random.uniform(1.0, 1.4)
        return F.brightness_contrast_adjust(img, contrast, brightness, True)


# class RandomCrop(ImageOnlyTransform):
#     def apply(self, img, **params):
#         flag = random.random()
#         if flag < 0.2:
#             return img, flag
#         else:
#             height, width = img.shape[:2]
#             scale = random.uniform(1.25, 1.5)
#             new_height, new_width = int(height * scale), int(width * scale)
#             scaled_img = cv2.resize(img, (new_width, new_height))
#             y = random.randint(0, new_height - height - 1)
#             x = random.randint(0, new_width - width - 1)
#             cropped_img = scaled_img[y:y + height, x:x + width, :]
#             if flag < 0.6:
#                 return cropped_img, flag
#             else:
#                 if random.random() < 0.5:
#                     color = np.array([0, 0, 0], dtype=np.uint8)
#                 else:
#                     color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)],
#                                      dtype=np.uint8)
#                 _img = np.multiply(np.ones_like(scaled_img), color)
#                 _img[y:y + height, x:x + width, :] = cropped_img
#                 return cv2.resize(_img, (width, height)), flag
#     def get_transform_init_args_names(self):
#         return ()

# def _random_paste(temp_img, img, flag=-1, min_val=0.15, max_val=0.3):
#     # img(background), template
#     temp_height, temp_width = temp_img.shape[:2]
#     height, width = img.shape[:2]
#     roi_h = int(random.uniform(min_val, max_val) * temp_height)
#     roi_w = int(random.uniform(min_val, max_val) * temp_width)
#     roi_y = random.randint(0, temp_height - roi_h - 1)
#     roi_x = random.randint(0, temp_width - roi_w - 1)
#     _img = cv2.resize(img, (roi_w, roi_h))
#     if flag == -1:
#         flag = random.random()
#     if flag < 0.6:
#         if flag < 0.3:
#             if random.random() < 0.5:
#                 color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)
#             else:
#                 color = np.array([0, 0, 0], dtype=np.uint8)  # black
#             _img = cv2.resize(temp_img, (roi_w, roi_h))
#             temp_img = np.multiply(np.ones_like(temp_img), color)
#         temp_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, :] = _img
#     elif flag < 0.8:
#         # Poisson Fusion
#         mask = 255 * np.ones((roi_h, roi_w), dtype=np.uint8)
#         center = (roi_x + roi_w // 2, roi_y + roi_h // 2)
#         temp_img = cv2.seamlessClone(_img, temp_img, mask, center, cv2.MIXED_CLONE if flag < 0.95 else cv2.NORMAL_CLONE)
#     else:
#         src = np.array([(0, 0), (0, width), (height, 0), (height, width)], dtype=np.float32)
#         dst = np.array([(x + random.uniform(-20, 20), y + random.uniform(-20, 20)) for x, y in src], dtype=np.float32)
#         M = cv2.getPerspectiveTransform(src, dst)
#         h, w = int(np.max(dst[:, 0]) - np.min(dst[:, 0])), int(np.max(dst[:, 1]) - np.min(dst[:, 1]))
#         mask_ = cv2.resize(cv2.warpPerspective(255 * np.ones_like(img), M, (w, h)), (roi_w, roi_h))
#         img_ = cv2.resize(cv2.warpPerspective(img, M, (w, h)), (roi_w, roi_h))
#         roi_img = cv2.bitwise_and(temp_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, :], cv2.bitwise_not(mask_)) + \
#                   cv2.bitwise_and(img_, mask_)
#         temp_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, :] = roi_img
#     return cv2.resize(temp_img, (width, height))

# def _random_perspective_paste(img1, img2, flag=True):
#     height1, width1 = img1.shape[:2]
#     height2, width2 = img2.shape[:2]
#     scale = random.uniform(0.2, 0.5)
#     if flag:
#         src = np.array([(0, 0), (0, width1), (height1, 0), (height1, width1)], dtype=np.float32)
#         dst = np.array([(x + random.uniform(-20, 20), y + random.uniform(-20, 20)) for x, y in src], dtype=np.float32)
#         M = cv2.getPerspectiveTransform(src, dst)
#         h, w = int(np.max(dst[:, 0]) - np.min(dst[:, 0])), int(np.max(dst[:, 1]) - np.min(dst[:, 1]))
#         mask_ = cv2.warpPerspective(255 * np.ones_like(img1), M, (w, h))
#         img_ = cv2.warpPerspective(img1, M, (w, h))
#         mask, img = np.zeros_like(img2), np.zeros_like(img2)
#         new_height, new_width = int(height2 * scale), int(width2 * scale)
#         mask_ = cv2.resize(mask_, (new_width, new_height))
#         img_ = cv2.resize(img_, (new_width, new_height))
#         y, x = random.randint(0, width2 - new_width - 10), random.randint(0, height2 - new_height - 10)
#         mask[x:x + new_height, y:y + new_width, :] = mask_
#         img[x:x + new_height, y:y + new_width, :] = img_
#         img = cv2.bitwise_and(img2, cv2.bitwise_not(mask)) + cv2.bitwise_and(img, mask)
#     else:
#         src = np.array([(0, 0), (0, width2), (height2, 0), (height2, width2)], dtype=np.float32)
#         dst = np.array([(x + random.uniform(-20, 20), y + random.uniform(-20, 20)) for x, y in src], dtype=np.float32)
#         M = cv2.getPerspectiveTransform(src, dst)
#         h, w = int(np.max(dst[:, 0]) - np.min(dst[:, 0])), int(np.max(dst[:, 1]) - np.min(dst[:, 1]))
#         mask_ = cv2.warpPerspective(255 * np.ones_like(img2), M, (w, h))
#         img_ = cv2.warpPerspective(img2, M, (w, h))
#         mask, img = np.zeros_like(img1), np.zeros_like(img1)
#         new_height, new_width = int(height1 * scale), int(width1 * scale)
#         mask_ = cv2.resize(mask_, (new_width, new_height))
#         img_ = cv2.resize(img_, (new_width, new_height))
#         y, x = random.randint(0, width1 - new_width - 10), random.randint(0, height1 - new_height - 10)
#         mask[x:x + new_height, y:y + new_width, :] = mask_
#         img[x:x + new_height, y:y + new_width, :] = img_
#         img = cv2.bitwise_and(img1, cv2.bitwise_not(mask)) + cv2.bitwise_and(img, mask)
#     return img

# class RandomPasteOverlay(ImageOnlyTransform):
#     def __init__(self, tfrecord_dir, template_list, always_apply=False, p=1.0):
#         super(RandomPasteOverlay, self).__init__(always_apply, p)
#         self.template_paths = []
#         with open(template_list) as fr:
#             for line in fr:
#                 record_name, record_index, offset, _ = line.strip().split('\t')
#                 fn = os.path.join(tfrecord_dir, record_name, "%s-%05d.tfrecord" % (record_name, int(record_index)))
#                 self.template_paths.append([fn, int(offset)])

#     def apply(self, img, **params):
#         fn, offset = random.choice(self.template_paths)
#         template_img = get_record(fn, offset)
#         flag = random.uniform(0.3, 0.6)
#         return _random_paste(template_img, img, flag, min_val=0.9, max_val=0.98)


# class RandomPerspectiveOverlay(ImageOnlyTransform):
#     def __init__(self, tfrecord_dir, template_list, always_apply=False, p=1.0):
#         super(RandomPerspectiveOverlay, self).__init__(always_apply, p)
#         self.template_paths = []
#         with open(template_list) as fr:
#             for line in fr:
#                 record_name, record_index, offset, _ = line.strip().split('\t')
#                 fn = os.path.join(tfrecord_dir, record_name, "%s-%05d.tfrecord" % (record_name, int(record_index)))
#                 self.template_paths.append([fn, int(offset)])

#     def apply(self, img, **params):
#         fn, offset = random.choice(self.template_paths)
#         template_img = get_record(fn, offset)
#         return _random_perspective_paste(img, template_img)


# class RandomTemplateOverlay(ImageOnlyTransform):
#     def __init__(self, template_dir, always_apply=False, p=1.0):
#         super(RandomTemplateOverlay, self).__init__(always_apply, p)
#         self.template_paths = glob.glob(os.path.join(template_dir, '*.jpg'))

#     def apply(self, img, **params):
#         template_img = cv2.imread(random.choice(self.template_paths))
#         return _random_paste(template_img, img)

# class RandomEmojiOverlay(ImageOnlyTransform):
#     def __init__(self, emoji_dir, always_apply=False, p=1.0):
#         super(RandomEmojiOverlay, self).__init__(always_apply, p)
#         self.emoji_paths = glob.glob(os.path.join(emoji_dir, '*/*.png')) + glob.glob(os.path.join(emoji_dir, '*/*.jpg'))

#     def apply(self, img, **params):
#         emoji = cv2.imread(random.choice(self.emoji_paths))
#         flag = random.uniform(0.3, 1)
#         return _random_paste(img, emoji, flag)
    
#     def get_transform_init_args_names(self):
#         return ()


class RandomTextOverlay(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(RandomTextOverlay, self).__init__(always_apply, p)
        self.fonts = [cv2.FONT_ITALIC,
                      cv2.FONT_HERSHEY_COMPLEX,
                      cv2.FONT_HERSHEY_DUPLEX,
                      cv2.FONT_HERSHEY_PLAIN,
                      cv2.FONT_HERSHEY_SIMPLEX,
                      cv2.FONT_HERSHEY_COMPLEX_SMALL,
                      cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                      cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                      cv2.FONT_HERSHEY_TRIPLEX]
        self.lines = [cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA]
        self.text = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                     'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                     '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                     '`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')',
                     '-', '=', '+', '<', ',', '>', '.', '?', '/', '{', '[', ']',
                     '}', ':', ';', '\'', '\"', '\\', '|', ' ']

    def apply(self, img, **params):
        height, width = img.shape[:2]
        text = str(random.choice(self.text[:52]) + ''.join(random.sample(self.text, random.randint(5, 50))))
        center = (random.randint(width//4, 3*width//4), random.randint(height//4, 3*height//4))
        font = random.choice(self.fonts)
        fontsize = random.randint(1, 6)
        fontcolor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        thickness = random.randint(1, 5)
        line = random.choice(self.lines)
        cv2.putText(img, text, center, font, fontsize, fontcolor, thickness, line)
        return img
    
    def get_transform_init_args_names(self):
        return ()


class RandomStripesOverlay(ImageOnlyTransform):
    def apply(self, img, **params):
        height, width = img.shape[:2]
        num = random.randint(1, 10)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        aug_img = img
        if random.random() > 0.5:
            starts = random.sample(list(range(height)), num)
            ends = [min(x + random.randint(0, 10), height) for x in starts]
            for x, y in zip(starts, ends):
                aug_img[x:y, :, 0] = color[0]
                aug_img[x:y, :, 1] = color[1]
                aug_img[x:y, :, 2] = color[2]
        else:
            starts = random.sample(list(range(width)), num)
            ends = [min(x + random.randint(0, 10), width) for x in starts]
            for x, y in zip(starts, ends):
                aug_img[:, x:y, 0] = color[0]
                aug_img[:, x:y, 1] = color[1]
                aug_img[:, x:y, 2] = color[2]
        return aug_img


if __name__ == "__main__":
    save_dir_path = "augmentation_test"
    os.makedirs(save_dir_path, exist_ok=True)
    img_path = "./dataset/WebFG496/web-bird/train/001.Black_footed_Albatross/001.Black_footed_Albatross_00001.jpg"
    trans = [
        RandomBorder(),
        RandomTranslate(),
        RandomFill(),
        RandomCompression(),
        RandomNoise(),
        RandomGridDistortion(),
        RandomHueSaturationValue(),
        RandomGaussianBlur(),
        RandomEqualize(),
        RandomFog(),
        RandomBrightness(),
        RandomTextOverlay(),
        RandomStripesOverlay(),
        alb.Equalize(p=1),
        alb.ColorJitter(p=1),
        alb.ToGray(p=1),
        alb.Sharpen(p=1),
        alb.OpticalDistortion(p=1),
        alb.GridDistortion(p=1, border_mode=cv2.BORDER_REPLICATE),
        alb.HueSaturationValue(p=1),
        alb.RandomBrightness(p=1),
        alb.RandomBrightnessContrast(p=1),
        # noise
        alb.ISONoise(p=1),
        alb.RandomFog(p=1, fog_coef_upper=0.5),
        alb.RandomSnow(p=1, brightness_coeff=1.2),
        alb.RandomRain(p=1, drop_length=5),
        alb.RandomShadow(p=1, num_shadows_lower=0, num_shadows_upper=1),
        alb.RandomToneCurve(p=1),
        alb.GaussNoise(p=1),
        ## blur and scale
        alb.Downscale(p=1),
        alb.ImageCompression(quality_lower=80, p=1),
        alb.MotionBlur(p=1),
        alb.Blur(p=1),
        alb.GaussianBlur(p=1),
        alb.GlassBlur(sigma=0.2, p=1),
    ]
    trans_names = [
        "rand_border",
        "rand_translate",
        "rand_fill",
        "rand_compression",
        "rand_noise",
        "rand_grid_distort",
        "rand_hue_saturation_value",
        "rand_gaussian_blur",
        "rand_equalize",
        "rand_fog",
        "rand_bright",
        "rand_text",
        "rand_stripe"
        "alb_equal",
        "alb_colorjitter",
        "alb_ToGray",
        "alb_Sharpen",
        "alb_OpticalDistortion",
        "alb_GridDistortion",
        "alb_HueSaturationValue",
        "alb_RandomBrightness",
        "alb_RandomBrightnessContrast",
        "alb_ISONoise",
        "alb_RandomFog",
        "alb_RandomSnow",
        "alb_RandomRain",
        "alb_RandomShadow",
        "alb_RandomToneCurve",
        "alb_GaussNoise",
        "alb_Downscale",
        "alb_ImageCompression",
        "alb_MotionBlur",
        "alb_Blur",
        "alb_GaussianBlur",
        "alb_GlassBlur",
    ]
    for tran, tran_name in zip(trans, trans_names):
        image = Image.open(img_path).convert("RGB")
        image.save(os.path.join(save_dir_path, "original.jpg"))
        image = np.array(image)
        alb_trans = alb.Compose([tran])
        image_aug = alb_trans(image=image)['image']
        image_aug = Image.fromarray(image_aug)
        image_aug.save(os.path.join(save_dir_path, "aug_{}.jpg".format(tran_name)))

