from skimage import img_as_uint
from skimage.io import imread, imsave
from skimage.color import rgb2gray, label2rgb
from skimage.measure import regionprops
from numpy import ndarray

from colonyscanalyser import imaging


def remove_background_mask(image: ndarray, smoothing: float = 1, sigmoid_cutoff: float = 0.4, **filter_args) -> ndarray:
    """
    Separate the image foreground from the background

    Returns a boolean mask of the image foreground

    :param image: an image as a numpy array
    :param smoothing: a sigma value for the gaussian filter
    :param sigmoid_cutoff: cutoff for the sigmoid exposure function
    :param filter_args: arguments to pass to the gaussian filter
    :returns: a boolean image mask of the foreground
    """
    from skimage import img_as_bool
    from skimage.exposure import adjust_sigmoid
    from skimage.filters import gaussian, threshold_triangle

    if image.size == 0:
        raise ValueError("The supplied image cannot be empty")

    image = image.astype("float64", copy = True)

    # Do not process the image if it is empty
    if not image.any():
        return img_as_bool(image)

    # Apply smoothing to reduce noise
    image = gaussian(image, smoothing, **filter_args)

    # Heighten contrast
    image = adjust_sigmoid(image, cutoff = sigmoid_cutoff, gain = 10)

    imsave(f"figures/plates/plate_contrast.png", image)

    # Find background threshold and return only foreground
    return image > threshold_triangle(image, nbins = 10)


def segment_image(
    plate_image: ndarray,
    plate_mask: ndarray = None,
    plate_noise_mask: ndarray = None,
    area_min: float = 1
) -> ndarray:
    """
    Attempts to separate and label all colonies on a plate

    :param plate_image: an image containing colonies
    :param plate_mask: a boolean image mask to remove from the original image
    :param plate_noise_mask: a black and white image as a numpy array
    :param area_min: the minimum area for a colony, in pixels
    :returns: a segmented and labelled image as a numpy array
    """
    from numpy import unique, isin
    from skimage.measure import label
    from skimage.morphology import remove_small_objects, binary_erosion
    from skimage.segmentation import clear_border

    plate_image = remove_background_mask(plate_image, smoothing = 0.5)

    imsave(f"figures/plates/plate_binarized.png", img_as_uint(plate_image))

    if plate_mask is not None:
        # Remove mask from image
        plate_image = plate_image & plate_mask
        # Remove objects touching the mask border
        plate_image = clear_border(plate_image, bgval = 0, mask = binary_erosion(plate_mask))
    else:
        # Remove objects touching the image border
        plate_image = clear_border(plate_image, buffer_size = 2, bgval = 0)

    imsave(f"figures/plates/plate_clear_border.png", img_as_uint(plate_image))

    plate_image = label(plate_image, connectivity = 2)

    # Remove background noise
    if len(unique(plate_image)) > 1:
        plate_image = remove_small_objects(plate_image, min_size = area_min)

    imsave(f"figures/plates/plate_clear_noise.png", img_as_uint(plate_image > 0))

    # Remove colonies that have grown on top of image artefacts or static objects
    if plate_noise_mask is not None:
        plate_noise_image = imaging.remove_background_mask(plate_noise_mask, smoothing = 0.5)
        if len(unique(plate_noise_mask)) > 1:
            noise_mask = remove_small_objects(plate_noise_image, min_size = area_min)
        imsave(f"figures/plates/plate_noise_binarized.png", img_as_uint(noise_mask))
        # Remove all objects where there is an existing static object
        exclusion = unique(plate_image[noise_mask])
        exclusion_mask = isin(plate_image, exclusion[exclusion > 0])
        imsave(f"figures/plates/plate_exclusion.png", img_as_uint(exclusion_mask))
        plate_image[exclusion_mask] = 0

    imsave(f"figures/plates/plate_clear_static.png", img_as_uint(plate_image > 0))

    return plate_image

# Save images of the different steps taken when binarizing a plate image

# Plate #6 (300DPI)
# Center (2892, 1844)
# Diameter 956 pixels
# Edge cut 53 pixels

img = imread(f"figures/plates/plate_lattice.tif", as_gray = False)
img = imaging.cut_image_circle(img, (2892, 1844), (956 // 2) - 53)

img_noise_mask = imread(f"figures/plates/plate_lattice_empty.tif", as_gray = False)
img_noise_mask = imaging.cut_image_circle(img_noise_mask, (2892, 1844), (956 // 2) - 53)

imsave(f"figures/plates/plate_cut.png", img)
imsave(f"figures/plates/plate_cut_noise.png", img_noise_mask)

img = rgb2gray(img)
img_segmented = segment_image(img, img > 0, plate_noise_mask = rgb2gray(img_noise_mask), area_min = 1.5)

imsave(f"figures/plates/plate_segmented.png", img_as_uint(img_segmented > 0))

imsave(f"figures/plates/plate_labels.png", label2rgb(img_segmented, bg_label = 0))
imsave(f"figures/plates/plate_labels_overlay.png", label2rgb(img_segmented, image = img, bg_label = 0))

for rp in regionprops(img_segmented):
    if rp.area < 50 or rp.area > 760:
        img_segmented[img_segmented == rp.label] = 0

imsave(f"figures/plates/plate_segmented_filtered.png", img_as_uint(img_segmented > 0))

imsave(f"figures/plates/plate_labels_filtered.png", label2rgb(img_segmented, bg_label = 0))
imsave(f"figures/plates/plate_labels_overlay_filtered.png", label2rgb(img_segmented, image = img, bg_label = 0))