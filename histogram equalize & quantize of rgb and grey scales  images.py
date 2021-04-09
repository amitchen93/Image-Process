import numpy as np
from skimage.color import rgb2gray
from skimage import io  # import io.imread
import matplotlib.pyplot as plt
import skimage  # for skimage.color
from scipy import stats
from sklearn import cluster as helping_cluster


# Q3.2
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


# Q3.3
def imdisplay(filename, representation):
    '''
    :param filename: filename
    :param representation: 1 (to gray) or 2 (to RGB)
    :return: open a new figure and display the loaded image in the converted representation
    '''
    image = read_image(filename, representation)
    #image = np.clip(image, 0, 1)    check normalized - we asked to avoid this

    if (representation == 1):
        plt.imshow(image, cmap=plt.cm.gray) #todo
    else:
        plt.imshow(image)

    plt.show()


# Q3.4
def rgb2yiq(imRGB):
    """
    :param imRGB:
    :return: returns an image having the same dimensions as the input
    """
    to_YIQ_matrix = np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.275, -0.321],
                              [0.212, -0.523, 0.311]])
    YIQ_matrix = np.dot(imRGB, to_YIQ_matrix.T)
    return YIQ_matrix



def yiq2rgb(imYIQ):
    """
    :param imYIQ:
    :return: returns an image having the same dimensions as the input
    """
    to_YIQ_matrix = np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.275, -0.321],
                              [0.212, -0.523, 0.311]])

    to_RGB_matrix = np.linalg.inv(to_YIQ_matrix)  # inverse the famous matrix "to_YIQ_matrix"
    RGB_matrix = np.dot(imYIQ, to_RGB_matrix.T)  # dot product with transpose
    return RGB_matrix


# Q 3.5
def histogram_equalize(im_orig):
    """
    :param im_orig: the input grayscale or RGB float64 image with values in [0, 1].
    :return: a list [im_eq, hist_orig, hist_eq] such:
    im_eq - is the equalized image. grayscale or RGB float64 image with values in [0; 1].
    hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
    hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    yiq_img = None

    # check if the image is RGB
    if im_orig[0][0].size == 3:
        yiq_img = rgb2yiq(im_orig)
        im_tmp = yiq_img[:, :, 0]

    else:
        im_tmp = im_orig


    # 256 bin histogram of the original image (array with shape (256,) ).
    flat_image = im_tmp.flatten()
    hist_orig, bounds_orig = np.histogram(flat_image, bins=256, range=(0, 1))   # create hist with range=(0, 1)


    # cum histogram and normalize:
    cum_hist = np.cumsum(hist_orig)
    cum_hist = (cum_hist * 255) / cum_hist[-1]


    # normalize histogram by the formula we saw in lecture (while cum_hist[-1] is number of pixels in the image)
    # find the first gray level for which C(m) != 0:
    first_not_zero = np.where(cum_hist > 0, cum_hist, cum_hist.max()).min(0)


    # calc look up table
    mone = 255 * (cum_hist - first_not_zero)
    mechane = cum_hist.max() - first_not_zero

    hist_eq = np.uint8(mone / mechane)  # calc and make each val int
    # hist_eq[hist_eq < 0] = 0  # each val is not 0 - make it 0

    new_image = hist_eq[(255 * flat_image).astype('uint8')] / 255

    if im_orig[0][0].size == 3:
        im_eq = np.reshape(new_image, im_orig[:, :, 0].shape)
        yiq_img[:, :, 0] = im_eq
        im_eq = yiq2rgb(yiq_img)


    else:
        im_eq = np.reshape(new_image, im_orig.shape)

    #plt.plot(hist_eq)
    #plt.show()
    return [im_eq, hist_orig, hist_eq]



# Q 3.6
def quantize(im_orig, n_quant, n_iter):
    """
    :param im_orig: is the input grayscale or RGB image to be quantized (float64 image with values in [0; 1])
    :param n_quant: is the number of intensities your output im_quant image should have.
    :param n_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return: list [im_quant, error] such:
     im_quant is the quantized output image. (float64 image with values in [0; 1]).
     error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
     quantization procedure.
    """
    yiq_img = None

    # check if the image is RGB
    if im_orig[0][0].size == 3:
        yiq_img = rgb2yiq(im_orig)
        im_tmp = yiq_img[:, :, 0]

    else:
        im_tmp = im_orig

    im_flat = im_tmp.flatten()

    # check if the image is RGB
    if im_orig[0][0].size == 3:
        yiq_img = rgb2yiq(im_orig)
        im_tmp = yiq_img[:, :, 0]

    hist_for_quantile, bounds_orig = np.histogram(im_orig, bins=256, range=(0, 1))

    update_z_list = list_of_z(n_quant, hist_for_quantile)  # helper function for list of z's
    # array of arrays that split by update_z_list

    # create a list of error:
    list_of_error = []  # empty list

    for i in range(n_iter):

        hist_cum = np.cumsum(hist_for_quantile * range(0, 256))  # without 0 index
        hist_cum = hist_cum[update_z_list[1:]]
        hist_cum = np.append([0], hist_cum)  # now add 0 in index 0 of hist_cum
        mone = hist_cum[1:] - hist_cum[:-1]  # diff of 2 array from hist_cum: a. [:-1], b. [1:]

        hist_cum = np.cumsum(hist_for_quantile)  # without 0 index
        hist_cum = hist_cum[update_z_list[1:]]
        hist_cum = np.append([0], hist_cum)  # now add 0 in index 0 of hist_cum
        sum_h_of_g = hist_cum[1:] - hist_cum[:-1]  # diff of 2 array from hist_cum: a. [:-1], b. [1:]

        q_list = mone / sum_h_of_g

        helper_z_list = np.uint8((q_list[1:] + q_list[:-1]) / 2)

        # if condition - if error is converge we stop
        if (helper_z_list == update_z_list[1:-1]).all():
            break

        update_z_list[1:-1] = helper_z_list  # update z list without 0 and 256 (don't want to update this)


        # calc the error (we want to minimize it)
        q_list_minus_find_g = np.split(np.arange(0, 256), update_z_list[1:-1]) - q_list
        q_list_minus_find_g = np.concatenate(q_list_minus_find_g)


        calc_inner_error = np.power(q_list_minus_find_g, 2) * hist_for_quantile  # by foumula
        calc_inner_error = calc_inner_error.cumsum()[update_z_list[1:]]   # calc the inner sigma
        calc_inner_error = np.append([0], calc_inner_error)   # append in first index

        calc_inner_error = (calc_inner_error[1:] - calc_inner_error[:-1])  # (q_i - g)^2 * h(g)


        error = calc_inner_error.sum()  # sum it
        list_of_error.append(error)  # update list_of_error in each iteration



    helper_look_up_table = np.zeros(256)
    for j in range(len(update_z_list) - 1):
        helper_look_up_table[update_z_list[j]:update_z_list[j+1]] = q_list[j]
    helper_look_up_table[-1] = q_list[-1]  # for 256 place

    new_image = helper_look_up_table[(255 * im_flat).astype('uint8')] / 255

    if im_orig[0][0].size == 3:
        im_quant = np.reshape(new_image, im_orig[:, :, 0].shape)
        yiq_img[:, :, 0] = im_quant
        im_quant = yiq2rgb(yiq_img)

    else:
        im_quant = np.reshape(new_image, im_orig.shape)

    return [im_quant, list_of_error]


def list_of_z(n_quant, histogram):
    """
    helper function for quantize that create list of z values
    :param n_quant: num of parts
    :param histogram: our histogram
    :return: list of z values
    """
    num_of_elements = 255
    cum_hist = np.cumsum(histogram)  # create cum_hist

    # like we asked to in ex1
    z_list = []
    n = 1 / n_quant
    for i in range(1, n_quant):
        index = np.where(cum_hist <= cum_hist[-1] * n)[0][-1]
        z_list.append(index)
        n += (1 / n_quant)
        #another option: start_index = (cum_hist[num_of_elements]) * n
        # this is update start_index to find the right patition

    final_z_list = [0] + z_list + [num_of_elements]
    return np.array(final_z_list)


# bonus:
def quantize_rgb(im_orig, n_quant):
    """
    :param im_orig: our im orig
    :param n_quant: n_quant like other questions
    :return: the final image
    """
    shape_1, shape_2, shape_3 = im_orig.shape  # the size of each dimension in img
    f_labels, as_plat = helper_function(im_orig, n_quant, shape_1, shape_2, shape_3)  # use get_label_plat()
    final_img = np.reshape(as_plat[f_labels], (shape_1, shape_2, shape_3))
    return final_img


def helper_function(orig_img, n_quant, shape_1, shape_2, shape_3=3):
    """
    :param im_orig: our im orig
    :param n_quant: n_quant like other questions
    :param shape_1: from matrix of rgb
    :param shape_2: from matrix of rgb
    :param shape_3: the last part is 3 because this the format of rgb
    :return: labels_of_predicts, plat for final image
    """
    clusters = helping_cluster.KMeans(n_clusters=n_quant)  # get clusters
    img = np.reshape(orig_img, (shape_1 * shape_2, shape_3))  # the last part is 3 because this the format of rgb
    labels_of_predicts = clusters.fit_predict(img)
    plat = clusters.cluster_centers_
    # Coordinates of cluster centers. If the algorithm stops before fully
    # converging (see tol and max_iter), these will not be consistent with labels_.

    return labels_of_predicts, plat

