from typing import List
from pathlib import Path
from skimage.color import rgb2gray
from skimage.io import imread
from numpy import ndarray

from colonyscanalyser.imaging import mm_to_pixels
from colonyscanalyser.plate import Plate, PlateCollection


def plot_plate_map(plate_image: ndarray, plates: List[Plate], save_path: Path) -> Path:
    """
    Saves original plate image with overlaid plate outline

    :param plate_image: the final timepoint image of all plates
    :param plates: a PlateCollection of Plate instances
    :param save_path: the directory to save the plot image
    :returns: a file path object if the plot was saved sucessfully
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # Calculate the image size in inches
    dpi = rcParams['figure.dpi']
    height, width, depth = plate_image.shape
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure that takes up the full size of the image
    fig = plt.figure(figsize = figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(plate_image)

    for plate in plates:
        center_y, center_x = plate.center

        # Label plates
        ax.annotate(
            f"Plate #{plate.id}".upper(),
            (center_x, center_y - plate.radius - (plate.edge_cut * 1.4)),
            xycoords = "data",
            horizontalalignment = "center",
            verticalalignment = "center",
            fontsize = "40",
            backgroundcolor = "black",
            color = "white"
        )

        # Mark the detected boundary of the plate
        plate_circle = plt.Circle(
            (center_x, center_y),
            radius = plate.radius,
            facecolor = "none",
            edgecolor = "red",
            linewidth = "5",
            linestyle = "-",
            label = "Detected plate boundary"
        )
        ax.add_artist(plate_circle)

    save_path = save_path.with_name(save_path.stem + "_outlined.png")
    try:
        plt.savefig(str(save_path), format = "png")
    except Exception:
        save_path = None
    finally:
        plt.close()
        return save_path


# Output a plate image with plate locations outlined
image_files = [
    ["./images/outlined/3_1_95.tif", (3, 1), 95],
    ["./images/outlined/3_2_35_empty.tif", (3, 2), 35],
    ["./images/outlined/3_2_35.tif", (3, 2), 35],
    ["./images/outlined/3_2_90_empty.tif", (3, 2), 90],
    ["./images/outlined/3_2_90.tif", (3, 2), 90],
    ["./images/outlined/6_4_35_empty.tif", (6, 4), 35],
    ["./images/outlined/6_4_35.tif", (6, 4), 35],
]

for image in image_files:
    image_file, lattice, size = image
    size = int(mm_to_pixels(size, dots_per_inch = 300))

    # Load the image
    img = imread(image_file, as_gray = False, plugin = "pil")

    # Locate plates in the image and store as Plate instances
    plates = PlateCollection.from_image(
        shape = lattice,
        image = rgb2gray(img),
        diameter = size,
        search_radius = size // 20,
        edge_cut = int(round(size * (5 / 100))),
        labels = []
    )

    # Plot the locations on the plate image
    plot_plate_map(img, plates.items, Path(image_file))
