import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.color import rgb2gray
import os

# You should use the function scipy.ndimage.filters.convolve to apply the filter on the image for best results.
from scipy.ndimage.filters import convolve
# or
from scipy.signal import convolve2d


def blur_image(image, filter_vector):
    """
    blur image by filter_vector
    :param image: the image we blur
    :param filter_vector: the filter vector
    :return: blur image by x and then y axis
    """
    blur_by_x_axis = convolve(image, filter_vector, mode="mirror")
    blur_by_y_axis_after_x_axis = convolve(blur_by_x_axis, filter_vector.T, mode="mirror")
    return blur_by_y_axis_after_x_axis



def reduce_image(image, filter_vector):
    """
    :param image: the image we reduce
    :param filter_vector: the filter vector
    :return: reduced image
    """
    image_to_reduce = blur_image(image, filter_vector)
    reduced_image = image_to_reduce[::2, ::2]  # each second pixel
    return reduced_image


def expand_image(image, filter_vector):
    """
    :param image: the image we expand
    :param filter_vector: the filter vector
    :return: expandd image
    """
    base_image = np.zeros((2 * image.shape[0], 2 * image.shape[1])).astype(np.float64)  # astype(np.float64)?
    base_image[::2, ::2] = image  # each 2'nd pixel
    blur_after_expanded = blur_image(base_image, 2 * filter_vector)
    return blur_after_expanded




def build_filter_for_pyr(filter_size):
    """
    This filter should be built using a consequent 1D convolutions of [1,1] with itself in order
    to derive a row of the binomial coefficients which is a good approximation to the Gaussian profile.
    The filter_vec should be normalized.
    :param filter_size: is row vector of shape (1, filter_size)
    :return: normalized filter vector
    """
    if filter_size == 0:
        return
    elif filter_size == 1:
        filter_vector = np.array([[1]]).astype(np.float64)  #astype(np.float64)
        return filter_vector
    start_filter, filter_vector = np.array([[1, 1]]).astype(np.float64), np.array([[1, 1]]).astype(np.float64)

    while filter_vector.size != filter_size:  # loop until equal size
        filter_vector = convolve2d(start_filter, filter_vector)

    # normalized
    sum_of_filter_vector = np.sum(filter_vector)
    filter_vector_normalized = filter_vector / sum_of_filter_vector

    return filter_vector_normalized  # the normalized filter vector after convolve2d process



def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im: – a grayscale image with double values in [0, 1]
        (e.g. the output of ex1’s read_image with the representation set to 1).
    :param max_levels: – the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
            in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
    :return:  pyr, filter_vec: pyramid pyr as a standard python array (i.e. not numpy’s array)
     with maximum length of max_levels, where each element of the array is a grayscale image.

    """
    # first -  we should use read_image from ex1
    pyr = [im]
    reduce_img = im
    filter_vec = build_filter_for_pyr(filter_size)  # get filter_vector

    # The number of levels in the resulting pyramids should be the largest possible s.t. max_levels isn’t exceeded and
    #  the minimum dimension (height or width) of the lowest resolution image in the pyramid is not smaller than 16
    while (len(pyr) < max_levels) and (min(reduce_img.shape) / 2 >= 16):  # change "min(reduced_img.shape)/2 > 16:?
        reduce_img = reduce_image(reduce_img, filter_vec)
        # you should convolve with this filter_vec twice - once as a row vector and then as
        # a column vector (for efficiency) (in reduce_image that calls blur_image)
        pyr.append(reduce_img)

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    :param im: – a grayscale image with double values in [0, 1]
        (e.g. the output of ex1’s read_image with the representation set to 1).
    :param max_levels: – the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
            in constructing the pyramid filter (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
    :return: pyr, filter_vec: pyramid pyr as a standard python array (i.e. not numpy’s array)
     with maximum length of max_levels, where each element of the array is a grayscale image.
    """
    pyr = []
    gaussian_pyramid, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)  # create gaussian pyramid

    while len(pyr) < len(gaussian_pyramid) - 1:
        laplacian_pyramid = gaussian_pyramid[len(pyr)] - expand_image(gaussian_pyramid[len(pyr) + 1], filter_vec)
        # to maintain constant brightness in the expand operation 2*filter should actually be used in each convolution.
        pyr.append(laplacian_pyramid)

    pyr.append(gaussian_pyramid[len(pyr)])

    return pyr, filter_vec


def read_image(filename, representation):
    '''
    :param filename: filename
    :param representation: 1 (to gray) or 2 (to RGB)
    :return: change color to gray (type of float64 and normlazied)
    '''
    image = np.float_(skimage.io.imread(filename))
    max_pix = 255
    image = (image / max_pix)

    if representation == 1:  # to gray scale
        gray_image = rgb2gray(image)
        return gray_image

    return image


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    reconstruction of an image from its Laplacian Pyramid.
    :param lpyr:  Laplacian pyramid and the filter that are generated by the second function in 3.1
    :param filter_vec: Laplacian pyramid and the filter that are generated by the second function in 3.1
    :param coeff: a python list. The list length is the same as the number of levels in the pyramid lpyr.
    Before reconstructing the image img you should multiply each level i of the laplacian pyramid by
    its corresponding coefficient coeff[i].
    :return: img
    """
    img = coeff[-1] * lpyr[-1]
    for level in range(2, len(lpyr) + 1):  # loop only by level (I understand we can do this)
        img = coeff[-level] * lpyr[-level] + expand_image(img, filter_vec)

    return img

def render_pyramid(pyr, levels):
    """
    a single black image in which the pyramid levels of the given pyramid pyr are stacked horizontally
    :param pyr: Gaussian or Laplacian pyramid as defined above
    :param levels: is the number of levels to present in the result ≤ max_levels.
    :return: The function render_pyramid should only return the big image res
    """

    # You should stretch the values of each pyramid level to [0, 1] before composing it into the black wide
    # image. Note that you should stretch the values of both pyramid types: Gaussian and Laplacian

    # trivial case:
    if len(pyr) == 0:
        return np.array([0])  # 0 array
    else:
        min_pyr = np.min(pyr[0])
        max_pyr = np.max(pyr[0])
        res = np.array((pyr[0] - min_pyr) / (max_pyr - min_pyr))
        for i in range(1, levels):
            i_level = np.zeros((pyr[0].shape[0], pyr[i].shape[1]))  # base level structure
            i_level[:pyr[i].shape[0]] = ((pyr[i] - min_pyr) / (max_pyr - min_pyr))
            res = np.concatenate((res, i_level), axis=1)

    return res  # The function render_pyramid should only return the big image res


def display_pyramid(pyr, levels):
    """
    should use render_pyramid to internally render and then display the stacked pyramid image using plt.imshow()
    :param pyr: Gaussian or Laplacian pyramid as defined above
    :param levels: is the number of levels to present in the result ≤ max_levels.
    :return: none (plot)
    """
    return_render_pyramid = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(return_render_pyramid, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.show()

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1: input grayscale images to be blended.
    :param im2: input grayscale images to be blended.
    :param mask: is a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts
            of im1 and im2 should appear in the resulting im_blend. Note that a value of True corresponds to 1,
            and False corresponds to 0.
    :param max_levels: is the max_levels parameter you should use when generating the Gaussian and Laplacian pyramids.
    :param filter_size_im: the size of the Gaussian filter (an odd scalar that represents a squared filter) which
            defining the filter used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: – is the size of the Gaussian filter(an odd scalar that represents a squared filter) which
            defining the filter used in the construction of the Gaussian pyramid of mask.
    :return:
    """
    # Construct Laplacian pyramids L1 and L2 for the input images im1 and im2, respectively
    laplacian_im1, filter_im1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    laplacian_im2, filter_im2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)

    # Construct a Gaussian pyramid Gm for the provided mask (convert it first to np.float64)
    gaus_mask, filter_mask = build_gaussian_pyramid(np.float64(mask), max_levels, filter_size_mask)

    pyramid_blend = []
    # Construct the Laplacian pyramid Lout of the blended image for each level k the formula
    for level_index in range(len(laplacian_im1)):
        level_of_pyr = gaus_mask[level_index] * laplacian_im1[level_index] + \
                       (1 - gaus_mask[level_index]) * laplacian_im2[level_index]
        pyramid_blend.append(level_of_pyr)

    # Reconstruct the resulting blended image from the Laplacian pyramid Lout (using ones for coefficients).
    # Make sure the output im_blend is a valid grayscale image in the range [0, 1], by clipping the result to range
    blended_image = laplacian_to_image(pyramid_blend, filter_im1, [1] * max_levels)
    return np.clip(blended_image, 0, 1)



# Q2 run "blend_imgs" with filter_size_im = 1,3,5,7...
# Q3 run "blend_imgs" with max_levels_im = 1,2,3,4,5...


def blending_example1():
    image_1 = read_image(relpath(r'pumpkin.jpg'), 2)
    image_2 = read_image(relpath(r'helmet.jpg'), 2)
    mask = read_image(relpath(r'mask_pumpkin.jpg'), 1)
    mask = mask > 0.5  # make mask boolean (black and white only).
    blend_product = np.clip(rgb_blend(image_1, image_2, mask, 9, 70, 3), 0, 1)
    plot_images(image_1, image_2, mask, blend_product)
    return image_1, image_2, mask.astype('bool'), blend_product  # mask – is a boolean

def blending_example2():
    image_1 = read_image(relpath(r'girl.jpg'), 2)
    image_2 = read_image(relpath(r'back_italy.jpg'), 2)
    mask = read_image(relpath(r'mask_girl.jpg'), 1)
    mask = mask > 0.5  # make mask boolean (black and white only).
    blend_product = np.clip(rgb_blend(image_1, image_2, mask, 11, 50, 3), 0, 1)
    plot_images(image_1, image_2, mask, blend_product)
    return image_1, image_2, mask.astype('bool'), blend_product  # mask – is a boolean

def rgb_blend(image_1, image_2, mask, max_levels, filter_size_im, filter_size_mask):
    # for color use rgb (same function by on another index)
    r = pyramid_blending(image_1[:, :, 0], image_2[:, :, 0], mask, max_levels, filter_size_im, filter_size_mask)
    g = pyramid_blending(image_1[:, :, 1], image_2[:, :, 1], mask, max_levels, filter_size_im, filter_size_mask)
    b = pyramid_blending(image_1[:, :, 2], image_2[:, :, 2], mask, max_levels, filter_size_im, filter_size_mask)
    blended_image = np.empty(image_1.shape)
    blended_image[:, :, 1] = g
    blended_image[:, :, 2] = b
    blended_image[:, :, 0] = r
    return blended_image


def plot_images(im1, im2, mask, im_blend):
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(im1)
    plt.subplot(2,2,2)
    plt.imshow(im2)
    plt.subplot(2,2,3)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(im_blend)
    plt.show()


#like we asked to
def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


