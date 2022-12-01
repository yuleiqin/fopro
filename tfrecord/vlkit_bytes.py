import numpy as np
from io import BytesIO
import PIL
from PIL import Image


def bytes2array(b: bytes) -> np.ndarray:
    """toolkit for tf-record packing: bytes 2 array"""
    return np.load(BytesIO(b), allow_pickle=True)


def bytes2image(b: bytes) -> PIL.Image.Image:
    """toolkit for tf-record packing: bytes 2 image"""
    return Image.open(BytesIO(b))


def bytes2image2array(b: bytes) -> np.ndarray:
    """toolkit for tf-record packing: bytes 2 image 2 array"""
    return np.array(bytes2image(b))


def array2bytes(b: bytes) -> bytes:
    """
    save ndarray shape information into bytes
    refer to https://stackoverflow.com/a/61838233/4095689
    """
    np_bytes = BytesIO()
    np.save(np_bytes, b, allow_pickle=True)
    return np_bytes.getvalue()


def image2bytes(image: PIL.Image.Image) -> bytes:
    """toolkit for tf-record packing: image 2 bytes"""
    assert isinstance(image, Image.Image)
    b = BytesIO()
    image.save(b)
    return b.getvalue()
