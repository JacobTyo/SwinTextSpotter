# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Implement many useful :class:`Augmentation`.
"""
import numpy as np
import sys
from fvcore.transforms.transform import (
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    VFlipTransform,
)
from PIL import Image
import random
from .augmentation import Augmentation, _transform_to_aug
from .transform import ExtentTransform, ResizeTransform, RotationTransform

__all__ = [
    "RandomApply",
    "RandomBrightness",
    "RandomContrast",
    "RandomCrop",
    "RandomExtent",
    "RandomFlip",
    "RandomSaturation",
    "RandomLighting",
    "RandomRotation",
    "Resize",
    "ResizeShortestEdge",
    "RandomCrop_CategoryAreaConstraint",
    "RandomCropWithInstance",
]


class SpeckleTransform(Transform):
    def __init__(
            self,
            fill_value: int = -1,
            number_speckles: int = 15,
            max_speckle_size: int = 10,
            max_speckle_aspect_ratio: float = 0.5
    ):
        """
        Args:
            fill_value: the value to fill the speckle with, if -1, fill with noise
            number_speckles: the number of speckles to add
            max_speckle_size: the maximum size of a speckle
            max_speckle_aspect_ratio: the maximum aspect ratio of a speckle
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Generate and apply all speckles as a single operation.

        Args:
            img: (H, W, C) array in range [0, 255] or [0, 1]
        Returns:
            img: (H, W, C) array in range [0, 255] or [0, 1]
        """
        if self.fill_value == -1:
            img = self._add_noise(img)
        else:
            img = self._add_color(img)
        return img

    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Generate and apply all speckles as a single operation.

        Args:
            img: (H, W, C) array in range [0, 255] or [0, 1]
        Returns:
            img: (H, W, C) array in range [0, 255] or [0, 1]
        """
        h, w, c = img.shape
        for _ in range(self.number_speckles):
            speckle_size = random.randint(1, self.max_speckle_size)
            speckle_aspect_ratio = random.uniform(0, self.max_speckle_aspect_ratio)
            speckle_h = int(np.sqrt(speckle_size / speckle_aspect_ratio))
            speckle_w = int(speckle_aspect_ratio * speckle_h)
            speckle_x = random.randint(0, w - speckle_w)
            speckle_y = random.randint(0, h - speckle_h)
            img[speckle_y:speckle_y + speckle_h, speckle_x:speckle_x + speckle_w, :] = np.random.randint(0, 255, (speckle_h, speckle_w, c))
        return img

    def _add_color(self, img: np.ndarray) -> np.ndarray:
        """
        Generate and apply all speckles as a single operation.

        Args:
            img: (H, W, C) array in range [0, 255] or [0, 1]
        Returns:
            img: (H, W, C) array in range [0, 255] or [0, 1]
        """
        h, w, c = img.shape
        for _ in range(self.number_speckles):
            speckle_size = random.randint(1, self.max_speckle_size)
            speckle_aspect_ratio = random.uniform(0, self.max_speckle_aspect_ratio)
            speckle_h = int(np.sqrt(speckle_size / speckle_aspect_ratio))
            speckle_w = int(speckle_aspect_ratio * speckle_h)
            speckle_x = random.randint(0, w - speckle_w)
            speckle_y = random.randint(0, h - speckle_h)
            img[speckle_y:speckle_y + speckle_h, speckle_x:speckle_x + speckle_w, :] = self.fill_value
        return img

    def inverse(self) -> Transform:
        raise NotImplementedError("Inverse is not implemented for SpeckleTransform")


class RandomApply(Augmentation):
    """
    Randomly apply an augmentation with a given probability.
    """

    def __init__(self, tfm_or_aug, prob=0.5):
        """
        Args:
            tfm_or_aug (Transform, Augmentation): the transform or augmentation
                to be applied. It can either be a `Transform` or `Augmentation`
                instance.
            prob (float): probability between 0.0 and 1.0 that
                the wrapper transformation is applied
        """
        super().__init__()
        self.aug = _transform_to_aug(tfm_or_aug)
        assert 0.0 <= prob <= 1.0, f"Probablity must be between 0.0 and 1.0 (given: {prob})"
        self.prob = prob

    def get_transform(self, *args):
        do = self._rand_range() < self.prob
        if do:
            return self.aug.get_transform(*args)
        else:
            return NoOpTransform()

    def __call__(self, aug_input):
        do = self._rand_range() < self.prob
        if do:
            return self.aug(aug_input)
        else:
            return NoOpTransform()


class RandomFlip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class Resize(Augmentation):
    """ Resize image to a fixed target size"""

    def __init__(self, shape, interp=Image.BILINEAR):
        """
        Args:
            shape: (h, w) tuple or a int
            interp: PIL interpolation method
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        self._init(locals())

    def get_transform(self, image):
        return ResizeTransform(
            image.shape[0], image.shape[1], self.shape[0], self.shape[1], self.interp
        )


class ResizeShortestEdge(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        if self.is_range:
            size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        else:
            size = np.random.choice(self.short_edge_length)
        if size == 0:
            return NoOpTransform()
        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)


class RandomRotation(Augmentation):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around the given center.
    """

    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None):
        """
        Args:
            angle (list[float]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the angle (in degrees).
                If ``sample_style=="choice"``, a list of angles to sample from
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (list[[float, float]]):  If ``sample_style=="range"``,
                a [[minx, miny], [maxx, maxy]] relative interval from which to sample the center,
                [0, 0] being the top left of the image and [1, 1] the bottom right.
                If ``sample_style=="choice"``, a list of centers to sample from
                Default: None, which means that the center of rotation is the center of the image
                center has no effect if expand=True because it only affects shifting
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        if angle % 360 == 0:
            return NoOpTransform()

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp)


class RandomSpeckles(Augmentation):
    def __init__(self,
            fill_value: int = -1,
            number_speckles: int = 15,
            max_speckle_size: int = 10,
            max_speckle_aspect_ratio: float = 0.5):
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        return SpeckleTransform(
            fill_value=self.fill_value,
            number_speckles=self.number_speckles,
            max_speckle_size=self.max_speckle_size,
            max_speckle_aspect_ratio=self.max_speckle_aspect_ratio
        )


class RandomCrop(Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.

        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        # TODO style of relative_range and absolute_range are not consistent:
        # one takes (h, w) but another takes (min, max)
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            NotImplementedError("Unknown crop type {}".format(self.crop_type))


# class RandomSpecks(Augmentation):
#     """
#     Randomly apply randon number of speckles, of random sizes, to an image. The specles can either be black,
#     random noise, or random colors. The speckles are added to the image with a random probability.
#     """
#
#     def __init__(self, mean_number_of_speckles: int, max_speckle_size: int, speckle_type: str = "any", speckle_prob: float = 0.5):
#         """
#         Args:
#             mean_number_of_speckles (int): the mean number of speckles to add to the image (Poisson distribution)
#             max_speckle_size (int): the maximum size of a speckle (in pixels)
#             speckle_type (str): the type of speckle to add to the image. One of "any", "black", "noise", "color"
#             speckle_prob (float): the probability of adding a speckle to the image
#         """
#         super().__init__()
#         self._init(locals())
#
#     def get_transform(self, image):




class RandomCrop_CategoryAreaConstraint(Augmentation):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that no single category
    occupies a ratio of more than `single_category_max_area` in semantic segmentation ground
    truth, which can cause unstability in training. The function attempts to find such a valid
    cropping window for at most 10 times.
    """

    def __init__(
        self,
        crop_type: str,
        crop_size,
        single_category_max_area: float = 1.0,
        ignored_category: int = None,
    ):
        """
        Args:
            crop_type, crop_size: same as in :class:`RandomCrop`
            single_category_max_area: the maximum allowed area ratio of a
                category. Set to 1.0 to disable
            ignored_category: allow this category in the semantic segmentation
                ground truth to exceed the area ratio. Usually set to the category
                that's ignored in training.
        """
        self.crop_aug = RandomCrop(crop_type, crop_size)
        self._init(locals())

    def get_transform(self, image, sem_seg):
        if self.single_category_max_area >= 1.0:
            return self.crop_aug.get_transform(image)
        else:
            h, w = sem_seg.shape
            for _ in range(10):
                crop_size = self.crop_aug.get_crop_size((h, w))
                y0 = np.random.randint(h - crop_size[0] + 1)
                x0 = np.random.randint(w - crop_size[1] + 1)
                sem_seg_temp = sem_seg[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
                labels, cnt = np.unique(sem_seg_temp, return_counts=True)
                if self.ignored_category is not None:
                    cnt = cnt[labels != self.ignored_category]
                if len(cnt) > 1 and np.max(cnt) < np.sum(cnt) * self.single_category_max_area:
                    break
            crop_tfm = CropTransform(x0, y0, crop_size[1], crop_size[0])
            return crop_tfm


class RandomExtent(Augmentation):
    """
    Outputs an image by cropping a random "subrect" of the source image.

    The subrect can be parameterized to include pixels outside the source image,
    in which case they will be set to zeros (i.e. black). The size of the output
    image will vary with the size of the random subrect.
    """

    def __init__(self, scale_range, shift_range):
        """
        Args:
            output_size (h, w): Dimensions of output image
            scale_range (l, h): Range of input-to-output size scaling factor
            shift_range (x, y): Range of shifts of the cropped subrect. The rect
                is shifted by [w / 2 * Uniform(-x, x), h / 2 * Uniform(-y, y)],
                where (w, h) is the (width, height) of the input image. Set each
                component to zero to crop at the image's center.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        img_h, img_w = image.shape[:2]

        # Initialize src_rect to fit the input image.
        src_rect = np.array([-0.5 * img_w, -0.5 * img_h, 0.5 * img_w, 0.5 * img_h])

        # Apply a random scaling to the src_rect.
        src_rect *= np.random.uniform(self.scale_range[0], self.scale_range[1])

        # Apply a random shift to the coordinates origin.
        src_rect[0::2] += self.shift_range[0] * img_w * (np.random.rand() - 0.5)
        src_rect[1::2] += self.shift_range[1] * img_h * (np.random.rand() - 0.5)

        # Map src_rect coordinates into image coordinates (center at corner).
        src_rect[0::2] += 0.5 * img_w
        src_rect[1::2] += 0.5 * img_h

        return ExtentTransform(
            src_rect=(src_rect[0], src_rect[1], src_rect[2], src_rect[3]),
            output_size=(int(src_rect[3] - src_rect[1]), int(src_rect[2] - src_rect[0])),
        )


class RandomContrast(Augmentation):
    """
    Randomly transforms image contrast.

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=image.mean(), src_weight=1 - w, dst_weight=w)


class RandomBrightness(Augmentation):
    """
    Randomly transforms image brightness.

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return BlendTransform(src_image=0, src_weight=1 - w, dst_weight=w)


class RandomSaturation(Augmentation):
    """
    Randomly transforms saturation of an RGB image.
    Input images are assumed to have 'RGB' channel order.

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation (make the image more grayscale)
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min, intensity_max):
        """
        Args:
            intensity_min (float): Minimum augmentation (1 preserves input).
            intensity_max (float): Maximum augmentation (1 preserves input).
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        assert image.shape[-1] == 3, "RandomSaturation only works on RGB images"
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        grayscale = image.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1 - w, dst_weight=w)


class RandomLighting(Augmentation):
    """
    The "lighting" augmentation described in AlexNet, using fixed PCA over ImageNet.
    Input images are assumed to have 'RGB' channel order.

    The degree of color jittering is randomly sampled via a normal distribution,
    with standard deviation given by the scale parameter.
    """

    def __init__(self, scale):
        """
        Args:
            scale (float): Standard deviation of principal component weighting.
        """
        super().__init__()
        self._init(locals())
        self.eigen_vecs = np.array(
            [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
        )
        self.eigen_vals = np.array([0.2175, 0.0188, 0.0045])

    def get_transform(self, image):
        assert image.shape[-1] == 3, "RandomLighting only works on RGB images"
        weights = np.random.normal(scale=self.scale, size=3)
        return BlendTransform(
            src_image=self.eigen_vecs.dot(weights * self.eigen_vals), src_weight=1.0, dst_weight=1.0
        )


def gen_crop_transform_with_instance(crop_size, image_size, instances, crop_box=True):
    """
        Generate a CropTransform so that the cropping region contains
        the center of the given instance.
        Args:
            crop_size (tuple): h, w in pixels
            image_size (tuple): h, w
            instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    bbox = random.choice(instances)
    bbox[::2] = np.clip(bbox[::2],0,image_size[1])
    bbox[1::2] = np.clip(bbox[1::2],0,image_size[0])
    crop_size = np.asarray(crop_size, dtype=np.int32)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"
    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))
    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
    if not crop_box:
        num_modifications = 0
        modified = True
        crop_size = crop_size.astype(np.float32)
        while modified:
            modified, x0, y0, crop_size = adjust_crop(x0, y0, crop_size, instances)
            num_modifications += 1
            if num_modifications > 100:
                raise ValueError(
                        "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
                                len(instances)
                                )
                        )
                return CropTransform(0, 0, image_size[1], image_size[0])
    if (x0 < 0) or (y0 < 0):
        x0 = np.maximum(x0,0)
        y0 = np.maximum(y0,0)
    return CropTransform(*map(int, (x0, y0, crop_size[1], crop_size[0])))

def adjust_crop(x0, y0, crop_size, instances, eps=1e-3):
    modified = False
    x1 = x0 + crop_size[1]
    y1 = y0 + crop_size[0]
    for bbox in instances:

        if bbox[0] < x0 - eps and bbox[2] > x0 + eps:
            crop_size[1] += x0 - bbox[0]
            x0 = bbox[0]
            modified = True

        if bbox[0] < x1 - eps and bbox[2] > x1 + eps:
            crop_size[1] += bbox[2] - x1
            x1 = bbox[2]
            modified = True

        if bbox[1] < y0 - eps and bbox[3] > y0 + eps:
            crop_size[0] += y0 - bbox[1]
            y0 = bbox[1]
            modified = True

        if bbox[1] < y1 - eps and bbox[3] > y1 + eps:
            crop_size[0] += bbox[3] - y1
            y1 = bbox[3]
            modified = True

    return modified, x0, y0, crop_size


class RandomCropWithInstance(RandomCrop):
    def __init__(self, crop_type, crop_size, crop_instance=False):
        """
        Args:
        crop_instance (bool): if False, extend cropping boxes to avoid cropping instances
        """
        super().__init__(crop_type, crop_size)
        self.crop_instance = crop_instance
        self.input_args = ("image", "boxes")
    def get_transform(self, img, boxes):
        image_size = img.shape[:2]
        crop_size = self.get_crop_size(image_size)
        return gen_crop_transform_with_instance(
            crop_size, image_size, boxes, crop_box=self.crop_instance
        )
