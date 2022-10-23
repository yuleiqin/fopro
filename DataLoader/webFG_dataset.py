import os
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import albumentations as alb
import cv2
import torch
import random
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from copy import deepcopy as copy
import json
import sys
sys.path.append("../")
from utils.rotate import rotate_and_crop
from utils.augmentations import RandomBorder, RandomTranslate
from utils.augmentations import RandomTextOverlay, RandomStripesOverlay



def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
                    if target_class not in available_classes:
                        available_classes.add(target_class)
    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        mode: str,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        transform_strong: Optional[Callable] = None,
        no_color_transform = False,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        self.mode = mode
        assert (transform is not None)
        if self.mode == 'train':
            assert(transform_strong is not None)
            self.transform_strong = transform_strong
        self.albs_transform_color = [alb.Equalize(p=1), alb.ColorJitter(p=1),\
            alb.ToGray(p=1), alb.Sharpen(p=1), alb.HueSaturationValue(p=1),\
                alb.RandomBrightness(p=1), alb.RandomBrightnessContrast(p=1),\
                    alb.RandomToneCurve(p=1)]
        self.albs_transform_basic = [RandomBorder(), RandomTranslate(),\
            RandomTextOverlay(), RandomStripesOverlay(),
                alb.OpticalDistortion(p=1),\
                    alb.GridDistortion(p=1, border_mode=cv2.BORDER_REPLICATE)]
        self.albs_transform_noise = [alb.ISONoise(p=1), alb.RandomFog(p=1, fog_coef_upper=0.5),\
            alb.RandomSnow(p=1, brightness_coeff=1.2), alb.RandomRain(p=1, drop_length=5),\
                alb.RandomShadow(p=1, num_shadows_lower=0, num_shadows_upper=1),\
                    alb.GaussNoise(p=1), alb.ImageCompression(quality_lower=90, p=1),\
                        alb.MotionBlur(p=1), alb.Blur(p=1),\
                            alb.GaussianBlur(p=1), alb.GlassBlur(sigma=0.2, p=1)]
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        idx_to_class = {v:k for k, v in self.class_to_idx.items()}
        self.idx_to_class = idx_to_class
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.no_color_transform = no_color_transform

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def alb_transform(self, image):
        """use albumentation transform for data augmentation"""
        image = np.array(image)
        alb_trans = [random.choice(self.albs_transform_basic),\
            random.choice(self.albs_transform_noise)]
        if not (self.no_color_transform):
            alb_trans.append(random.choice(self.albs_transform_color))
        alb_trans = alb.Compose(random.sample(alb_trans, random.randint(1, len(alb_trans))))
        image_aug = alb_trans(image=image)['image']
        image_aug = Image.fromarray(image_aug)
        return image_aug

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.mode == "train":
            img = self.alb_transform(sample)
        else:
            img = sample
        img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.mode == 'train':
            if self.no_color_transform:
                rotate_deg = np.random.randint(low=-45, high=45)
                sample = rotate_and_crop(sample, rotate_deg)
            img_aug = self.alb_transform(sample)
            img_aug = self.transform_strong(img_aug)
            return img, target, img_aug, torch.Tensor([self.domains[index]]), path, torch.Tensor([self.samples_index[index]])
        elif self.mode == 'test' or self.mode == "fewshot":
            return img, target, torch.Tensor([self.domains[index]]), path, torch.Tensor([self.samples_index[index]])

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# # TODO: specify the return type
# def accimage_loader(path: str) -> Any:
#     import accimage
#     try:
#         return accimage.Image(path)
#     except OSError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def default_loader(path: str) -> Any:
    # from torchvision import get_image_backend
    # if get_image_backend() == "accimage":
    #     return accimage_loader(path)
    # else:
    return pil_loader(path)



class webFG496_dataset(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = default_loader,
        root_target: str = "",
        label_target: str = "",
        mode: str = "train",
        save_dir: str = "",
        dry_run: bool = False,
        use_fewshot: bool = True,
        annotation: str = "",
        no_color_transform: bool = False,
        transform: Optional[Callable] = None,
        transform_strong: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            mode,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            transform_strong=transform_strong,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            no_color_transform = no_color_transform,
        )
        if mode == "train" and annotation != "" and os.path.exists(annotation):
            print("[TRAIN] Load samples by pseudo label json for web domain")
            annotation_json = json.load(open(annotation, "r"))
            samples = annotation_json['samples']
            root_dirs = annotation_json["roots"]
            index2roots = annotation_json["index2root"]
            samples_full = []
            for sample, root_dir in zip(samples, root_dirs):
                ## 增加根目录
                img_name, label = sample
                img_name_full = os.path.join(index2roots[str(root_dir)], img_name)
                assert(os.path.exists(img_name_full))
                # print(img_name_full)
                samples_full.append([img_name_full, label])
            self.samples = samples_full
            self.targets = annotation_json['targets']
            self.domains = annotation_json['domains']
            print("number of samples", len(self.samples),\
                "number of labels", len(self.targets),\
                    "number of domain labels", len(self.domains))
        else:
            if mode == "train" or (mode == "fewshot" and not use_fewshot):
                print("[TRAIN] Load samples by image folder for web domain")
                self.domains = [0 for _ in range(len(self.samples))]
                print("number of samples", len(self.samples),\
                    "number of labels", len(self.targets),\
                        "number of domain labels", len(self.domains),\
                            "number of classes", len(self.class_to_idx))
                path_filter_name = os.path.join(os.path.dirname(root), "train-list-filter.txt")
                if os.path.exists(path_filter_name):
                    print("Load Filter Name Pathlsit")
                    filter_names = set()
                    with open(path_filter_name, "r") as f:
                        lines = f.readlines()
                    for line in lines:
                        img_path_i = os.path.join(os.path.dirname(root), line.strip().split(": ")[0])
                        filter_names.add(img_path_i)
                    print("Total number of filtered images={}".format(len(filter_names)))
                    samples, targets, domains = [], [], []
                    for sample, target, domain in zip(self.samples, self.targets, self.domains):
                        if not (sample[0] in filter_names):
                            samples.append(sample)
                            targets.append(target)
                            domains.append(domain)
                    self.samples = samples
                    self.targets = targets
                    self.domains = domains
                    print("Filtered number of samples {} labels {} domains {}".format(
                        len(self.samples), len(self.targets), len(self.domains)
                    ))
            elif mode == "test":
                print("[TEST] Load samples by image folder for target domain")
                self.domains = [1 for _ in range(len(self.samples))]
                print("number of samples", len(self.samples),\
                    "number of labels", len(self.targets),\
                        "number of domain labels", len(self.domains),\
                            "number of classes", len(self.class_to_idx))
            if root_target != "" and label_target != "" and os.path.exists(root_target) and os.path.exists(label_target):
                # print("dataset root dir ", root_target)
                # print("target dir ", label_target)
                if mode != "test":
                    samples_supp = []
                    targets_supp = []
                    class_names_supp = set()
                    with open(label_target, "r") as f_read:
                        for line in f_read:
                            img_path, class_name = line.strip().split("@")
                            path = os.path.join(root_target, img_path)
                            assert (os.path.isfile(path)), "please make sure the path to target domain dir is valid {}".format(path)
                            class_index = self.class_to_idx[class_name]
                            class_names_supp.add(class_name)
                            item = path, class_index
                            samples_supp.append(item)
                            targets_supp.append(class_index)
                    domains_supp = [1 for _ in range(len(samples_supp))]
                    if mode == "train" and use_fewshot:
                        print("[TRAIN] Load samples by pathlist for target domain")
                        self.samples += samples_supp
                        self.targets += targets_supp
                        self.domains += domains_supp
                    elif mode == "fewshot":
                        print("[FEWSHOT] Load samples by pathlist for target domain")
                        if use_fewshot:
                            ## 使用fewshot样本
                            self.samples = samples_supp
                            self.targets = targets_supp
                            self.domains = domains_supp
                        else:
                            ## 使用抽样每类样本
                            imgs_by_target = {}
                            for sample, target, domain in zip(self.samples, self.targets, self.domains):
                                if not target in imgs_by_target:
                                    imgs_by_target[target] = []
                                imgs_by_target[target].append([sample, target, domain])
                            samples_fewshot = []
                            targets_fewshot = []
                            domains_fewshot = []
                            for target, img_list in imgs_by_target.items():
                                img_list_sampled = random.sample(img_list, min(16, len(img_list)))
                                for img_item in img_list_sampled:
                                    samples_fewshot.append(img_item[0])
                                    targets_fewshot.append(img_item[1])
                                    domains_fewshot.append(img_item[2])
                            self.samples = samples_fewshot
                            self.targets = targets_fewshot
                            self.domains = domains_fewshot

                    print("number of samples", len(self.samples),\
                        "number of labels", len(self.targets),\
                            "number of domain labels", len(self.domains),\
                                "number of classes", len(class_names_supp))
            else:
                print("image path {} or label path {} does not exist".format(root_target, label_target))

        if dry_run:
            if mode == "train":
                target_ids = set()
                random_index = []
                for idx, target_id in enumerate(self.targets):
                    if not (target_id in target_ids):
                        target_ids.add(target_id)
                        random_index.append(idx)
                random_index += np.arange(len(self.samples)-32, len(self.samples)).tolist()
                random.shuffle(random_index)
                self.samples = [self.samples[idx] for idx in random_index]
                self.targets = [self.targets[idx] for idx in random_index]
                self.domains = [self.domains[idx] for idx in random_index]
            else:
                self.samples = self.samples[:64]
                self.targets = self.targets[:64]
                self.domains = self.domains[:64]
            print("domains ", self.domains)
            print("[Dry-run] number of samples", len(self.samples),\
                "number of labels", len(self.targets),\
                    "number of domains", len(self.domains))

        print("Preparing mapping from image path to image index number")
        self.samples_index = []
        save_mapping_index = os.path.join(save_dir, "index_mapping_{}.txt".format(mode))
        with open(save_mapping_index, "w") as f_write:
            for index, item in enumerate(self.samples):
                path, target = item
                self.samples_index.append(index)
                f_write.write(" ".join([str(index), str(path), str(target)]) + "\n")
        
        self.samples_copy = copy(self.samples)
        self.targets_copy = copy(self.targets)
        self.domains_copy = copy(self.domains)
        self.samples_index_copy = copy(self.samples_index)

    def repeat(self):
        assert(self.mode == "train"), "repeat for data-resampling is only valid for traininig mode"
        print('=> repeating dataset for balancing')
        ## 需要更新4个列表 samples; targets; domains; sample_index
        targets = np.array(self.targets_copy)
        uniq, freq = np.unique(targets, return_counts=True)
        inv = (1/freq)**0.5
        p = inv/inv.sum()
        weight = (10 * p)/(p.min())
        weight = weight.astype(int)
        weight = {u:w for u,w in zip(uniq, weight)}
        ## 从初始化的所有数据列表中进行re-balance抽样
        samples_balance = []
        targets_balance = []
        domains_balance = []
        samples_index_balance = []
        for im, lab, domain, im_index in zip(self.samples_copy, self.targets_copy,\
            self.domains_copy, self.samples_index_copy):
            samples_balance += [im]*weight[lab]
            targets_balance += [lab]*weight[lab]
            domains_balance += [domain]*weight[lab]
            samples_index_balance += [im_index]*weight[lab]
        ## 更新样本集
        self.samples = []
        self.targets = []
        self.domains = []
        self.samples_index = []
        ## 从平衡的样本集中随机抽取固定数目的样本
        index_shuf = list(range(len(samples_balance)))
        random.shuffle(index_shuf)
        for i in index_shuf[:len(self.targets_copy)]:
            self.samples.append(samples_balance[i])
            self.targets.append(targets_balance[i])
            self.domains.append(domains_balance[i])
            self.samples_index.append(samples_index_balance[i])
        print('=> done')
        return


class webFG496_dataloader():  
    def __init__(self, batch_size, num_class, num_workers, root_dir,\
        distributed, crop_size=0.2, root_dir_target="", pathlist_target="",\
            save_dir="", dry_run=False, use_fewshot=True, annotation="",\
                no_color_transform=False, eval_only=False):
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.use_fewshot = use_fewshot
        self.annotation = annotation
        self.root_dir = root_dir
        self.distributed = distributed
        ## path to the target domain directory
        self.root_dir_target = root_dir_target
        self.pathlist_target = pathlist_target
        ## path to save the image pathlist mapping
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        ## transform
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(448, scale=(crop_size, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.no_color_transform = no_color_transform
        if no_color_transform:
            self.transform_strong = transforms.Compose([
                transforms.RandomResizedCrop(448, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform_strong = transforms.Compose([
                transforms.RandomResizedCrop(448, scale=(crop_size, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),            
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])      
        self.transform_test = transforms.Compose([
                transforms.Resize(512), 
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])
        self.eval_only = eval_only
        if eval_only:
            self.transform_train = self.transform_test
            self.transform_strong = self.transform_test

    def run(self):
        print("------------------------------------------------------")
        save_dir_train = os.path.join(self.save_dir, "data_train")
        os.makedirs(save_dir_train, exist_ok=True)
        train_dataset = webFG496_dataset(
            root=os.path.join(self.root_dir, "train"),
            root_target=self.root_dir_target,
            label_target=self.pathlist_target,
            mode="train",
            save_dir=save_dir_train,
            transform=self.transform_train,
            transform_strong=self.transform_strong,
            dry_run=self.dry_run,
            use_fewshot=self.use_fewshot,
            annotation=self.annotation,
            no_color_transform=self.no_color_transform,
        )
        print("------------------------------------------------------")
        save_dir_fewshot = os.path.join(self.save_dir, "data_fewshot")
        os.makedirs(save_dir_fewshot, exist_ok=True)
        fewshot_dataset = webFG496_dataset(
            root=os.path.join(self.root_dir, "train"),
            root_target=self.root_dir_target,
            label_target=self.pathlist_target,
            mode="fewshot",
            save_dir=save_dir_fewshot,
            use_fewshot=self.use_fewshot,
            transform=self.transform_train,
            transform_strong=self.transform_strong,
            dry_run=self.dry_run,
        )
        print("------------------------------------------------------")
        save_dir_test = os.path.join(self.save_dir, "data_test")
        os.makedirs(save_dir_test, exist_ok=True)       
        test_dataset = webFG496_dataset(
            root=os.path.join(self.root_dir, "val"),
            mode="test",
            save_dir=save_dir_test,
            transform=self.transform_test,
            dry_run=self.dry_run,
        )
        print("------------------------------------------------------")
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            fewshot_sampler = torch.utils.data.distributed.DistributedSampler(fewshot_dataset, shuffle=False)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            fewshot_sampler = None
            test_sampler = None

        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True)                                       

        fewshot_loader = DataLoader(
            dataset=fewshot_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=fewshot_sampler) 
                 
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=test_sampler)                                     
    
        return train_loader, fewshot_loader, test_loader
