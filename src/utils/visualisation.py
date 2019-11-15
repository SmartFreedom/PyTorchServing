import matplotlib.pyplot as plt
import seaborn as sns


def make_image_row(image, subax):
    subax[0].imshow(image[0])
    subax[0].axis('off')
    subax[1].imshow(image[1])
    subax[1].axis('off')
    subax[2].imshow(image[2])
    subax[2].axis('off')
    subax[3].imshow(image[3])
    subax[3].axis('off')
    return subax


def make_image_rows(image, subax, title=None):
    assert len(image) == len(subax)

    for i, el in enumerate(subax):
        el.imshow(image[i])
        if title is not None:
            el.title.set_text(title[i])
        el.axis('off')

    return subax
