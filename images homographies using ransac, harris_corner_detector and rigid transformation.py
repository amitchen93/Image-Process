



# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged

# add those functions to sol4??? _utils should include:
#  1. read_image  2. build_gaussian_pyramid  3.  pyramid_blending for the bonus


# 3.1 -  you will implement a function harris_corner_detector which gets
# a grayscale image and returns (x, y) locations that represent corners in the image

import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy import convolve
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite
from scipy.signal import convolve2d

import sol4_utils


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    param_for_R = 0.04  # for R calc
    # using the filters [1, 0, −1], [1, 0, −1]^T  respectively :
    convolve_kernel = np.array([[1, 0, -1]], dtype=np.float64)
    # from scipy:
    I_x = convolve2d(im, convolve_kernel, mode='same', boundary='symm')
    I_y = convolve2d(im, convolve_kernel.T, mode='same', boundary='symm')

    # blur with blur_spatial function from sol4_utils.py with kernel_size=3:
    blurred_x_x = sol4_utils.blur_spatial(I_x * I_x, kernel_size=3)
    blurred_y_y = sol4_utils.blur_spatial(I_y * I_y, kernel_size=3)
    blurred_x_y = sol4_utils.blur_spatial(I_y * I_x, kernel_size=3)
    k = 0.04
    R = calculate_R(blurred_x_x, blurred_y_y, blurred_x_y, k)  # use helper function

    corners = non_maximum_suppression(R)
    return np.fliplr(np.argwhere(corners))  # Flip array in the left/right direction
    # # I put a note (#) in ._sol4_utils.py (look at this file) because it's make some error


def calculate_R(blurred_x_x, blurred_y_y, blurred_x_y, k):
    """
    :param k: 0.04 constant
    :param blurred_x_x: blurred_x_x
    :param blurred_y_y: blurred_y_y
    :param blurred_x_y: blurred_x_y
    :return: R
    """
    trace_of_matrix = blurred_x_x + blurred_y_y  # sum of *diagonal* of matrix

    # calc determinanta of matrix:
    main_diagonal_of_matrix = (blurred_x_x * blurred_y_y)
    second_diagonal_of_matrix = (blurred_x_y ** 2)  # blurred_x_y * blurred_x_y
    determinanta_of_matrix = main_diagonal_of_matrix - second_diagonal_of_matrix

    #  One way to measure how big are the two eigenvalues is (We will use k = 0.04):
    R = determinanta_of_matrix - (k * (trace_of_matrix ** 2))
    return R


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    # K = 1+2∗desc rad
    k = 1 + (2 * desc_rad)
    descriptors_for_points = np.zeros((len(pos), k, k))  # another "()" because the data type is not understood without
    arange = np.arange((-1 * desc_rad), desc_rad + 1)

    #try this if repeat don't work:_coordinates = np.arange((-1 * desc_rad), desc_rad + 1)
    y_coordinates = np.repeat(arange, k, axis=0).astype('float64')
    x_coordinates = np.tile(arange, k).astype('float64')

    ezer = np.array([y_coordinates, x_coordinates])  # float

    #patch_for_x, patch_for_y = np.meshgrid(arange, y_coordinates, indexing='xy')
    for index in range(pos.shape[0]):
        ezer_i = np.copy(ezer)
        ezer_i[0] = ezer_i[0] + pos[index][1]
        ezer_i[1] = ezer_i[1] + pos[index][0]

        #  use map_coordinates from scipy.ndimage (with order=1 and prefilter=False for linear interpolation)
        patch = map_coordinates(im, ezer_i, order=1, prefilter=False)
        mean_of_patch = np.mean(patch)


        # use the euclidean norm operation (use np.linalg.norm)
        patch = patch - mean_of_patch
        normalized_patch = np.linalg.norm(patch)  # decrease the mean
        patch = patch.reshape((7, 7))

        if normalized_patch == 0:
            descriptors_for_points[index] = patch
        else:
            descriptors_for_points[index] = patch / normalized_patch

    return descriptors_for_points




def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    desc_rad = 3
    pyr_level = 2  # 3 or 2?

    # Note also that to obtain (m × n) = (7 × 7) descriptors, desc_rad should be set to 3
    #  getting the key points
    points_of_interest = spread_out_corners(pyr[0], 7, 7, desc_rad)

    # r sampling a descriptor for each key point
    descriptors_for_points = sample_descriptor(pyr[pyr_level], (points_of_interest / (pyr_level ** 2)), desc_rad)

    return [points_of_interest, descriptors_for_points]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """


    score = np.tensordot(desc1, desc2, [(1, 2), (1, 2)])
    max2_desc1 = np.partition(score, -2)[:, -2]
    max2_desc2 = np.partition(score.T, -2)[:, -2]
    reshape_max_second_desc1 = max2_desc1.reshape((max2_desc1.shape[0], 1))
    reshape_max_second_desc2 = max2_desc2.reshape((max2_desc2.shape[0], 1))

    # 3 conditions should be True:
    cond_desc1 = (score >= reshape_max_second_desc1)
    cond_desc2 = (score.T >= reshape_max_second_desc2)
    cond_desc2_trans = cond_desc2.T
    cond_3 = (score > min_score)
    all_true = cond_desc1 & cond_desc2_trans & cond_3
    match_features1, match_features2 = np.where(all_true)
    return [match_features1, match_features2]



def do_tensordot(desc1, desc2):
    """
    :param desc1:
    :param desc2:
    :return: score of tensor-dot
    """
    score = np.tensordot(desc1, desc2, [(1, 2), (1, 2)])
    return score


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y]
     point coordinates obtained from transforming pos1 using H12.
    """
    matrix = np.insert(pos1, obj=pos1.shape[1], values=1, axis=1).T
    trans_matrix = matrix.T
    x_y_z = np.dot(H12, matrix).T  # dot product of H12 and matrix
    x_y_z = np.divide(x_y_z, x_y_z[:, -1].reshape(x_y_z.shape[0], 1))

    return x_y_z[:, :x_y_z.shape[1] - 1]


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    if num_iter == 0:
        unit_matrix = np.eye(3)
        empty_array = np.array([])
        return [unit_matrix, empty_array]
    if (len(points1) == 0) or (len(points2) == 0):
        unit_matrix = np.eye(3)
        empty_array = np.array([])
        return [unit_matrix, empty_array]

    is_there_match = 0
    num_of_points_matches = 2
    if translation_only:
        num_of_points_matches = 1
        is_there_match = 1
    inliers = np.array([])

    for i in range(num_iter):  # run as num_iter iterations
        # create random pick and then pick 2 points randomly from each array
        t = np.arange(len(points1))
        pick_randomly = np.random.choice(np.arange(len(points1)), num_of_points_matches, replace=False)
        p_1 = np.array(points1[pick_randomly])
        p_2 = np.array(points2[pick_randomly])

        # To simplify matters you have been provided with the estimate_rigid_transform function that performs this step
        #  This function returns a 3x3 homography matrix which performs the rigid transformation.
        H = estimate_rigid_transform(p_1, p_2, translation_only)  # use translation only = True

        # Use H_1_2 to transform the set of points P1 in image 1 to the transformed set P_2
        points1_apply_homography = apply_homography(points1, H)

        E_j = np.linalg.norm(points1_apply_homography - points2, axis=1)
        maybe_inliers_points = np.argwhere(E_j < inlier_tol).ravel()  # Mark all matches having this condition

        if len(maybe_inliers_points) > len(inliers):
            inliers = maybe_inliers_points

    H = estimate_rigid_transform(points1[inliers], points2[inliers], translation_only)

    return [H, inliers]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    concatenated_image = np.hstack((im1, im2))
    plt.imshow(concatenated_image, cmap="gray")

    outliers = np.delete(np.arange(len(points1)), inliers)
    more_layer = 0.9
    less_layer = 0.1
    plt.plot([points1[outliers, 0], points2[outliers, 0] + im1.shape[1]], [points1[outliers, 1], points2[outliers, 1]],
             mfc='r', c='b', lw=less_layer, ms=2, marker='o')

    plt.plot([points1[inliers, 0], points2[inliers, 0] + im1.shape[1]], [points1[inliers, 1], points2[inliers, 1]],
             mfc='r', c='y', lw=more_layer, ms=5, marker='o')

    plt.show()

def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    H_eye = [(np.eye(3))]
    H = [np.array(np.eye(3))]
    H_left = H_succesive[:m]
    H_right = H_succesive[m:]   # Compute the (multiplicative) inverse of a matrix.

    loop_of_H_left = H_left[::-1]

    for i, H_i in enumerate(loop_of_H_left):
        multiplication_matrix = H[-1] @ H_i  # Matrix Multiplication Operator
        # normalizing them as follows before using them to perform transformations H /= H[2,2]
        norm_par = multiplication_matrix / multiplication_matrix[2, 2]
        H.append(norm_par)

    H = H[::-1]  # reverse H

    for i, H_i in enumerate(H_right):
        multiplication_matrix = H[-1] @ np.linalg.inv(H_i)  # Matrix Multiplication Operator
        # normalizing them as follows before using them to perform transformations H /= H[2,2]
        norm_par = (multiplication_matrix / multiplication_matrix[2, 2])
        H.append(norm_par)

    return H



def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    transformation = apply_homography(np.array([[0, 0], [0, h-1], [w-1, 0], [w-1, h-1]]), homography)
    min_bound = [min(transformation[:, 0]), min(transformation[:, 1])]
    max_bound = [max(transformation[:, 0]), max(transformation[:, 1])]
    final_bounds = np.array([min_bound, max_bound]).reshape(2, 2).astype(np.int)  # maybe dtype=int?

    return final_bounds


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    h = image.shape[0]
    w = image.shape[1]
    #  these should be in the bounding box calculated by compute bounding box
    [[x_1, y_1], [x_2, y_2]] = compute_bounding_box(homography, w, h)

    # using the function np.meshgrid to hold the x and y coordinates of each of the warped image
    arange_of_x = np.arange(x_1, x_2)
    arange_of_y = np.arange(y_1, y_2)
    x_v, y_v = np.meshgrid(arange_of_x, arange_of_y)

    #  denoted Xi and Y i should be transformed by the inverse homography using
    #  apply_homography back to the coordinate system of frame i
    x_flatten = x_v.flatten()
    y_flatten = y_v.flatten()
    inv_homography = np.linalg.inv(homography)
    back = apply_homography(np.array([x_flatten, y_flatten]).T, inv_homography)

    x_prime = back[:, 0].reshape(x_v.shape)
    y_prime = back[:, 1].reshape(y_v.shape)

    # These back-warped coordinates can now be used to interpolate the image with map_coordinates
    wrap_the_image = map_coordinates(image, [y_prime, x_prime], order=1, prefilter=False)

    return wrap_the_image


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()

# check photo of street - this check the display_matches:

# def read_image(filename, representation):
#     """
#     :param filename: the filename of an image on disk (could be grayscale or RGB).
#     :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
#            image (1) or an RGB image (2).
#            NOTE: If the input image is grayscale, we won't call it with representation = 2.
#     :return: This function returns an image.
#     """
#     # load and normalize the image
#     image = io.imread(filename)
#     image = image / 255
#     # grayscale if needed
#     if representation == 1:
#         image = color.rgb2gray(image)
#     return image.astype(np.float64)
#
# from skimage import io
# from skimage import color
#
# img_p = r"oxford1.jpg"
# img_p2 = r"oxford2.jpg"
#
# im1 = sol4_utils.read_image(img_p, 1)
# im2 = sol4_utils.read_image(img_p2, 1)
#
#
# from skimage.transform import pyramid_gaussian
#
# im1_pyr_gen = tuple(pyramid_gaussian(im1))
# im2_pyr_gen = tuple(pyramid_gaussian(im2))
#
# im1_pyr = [im1_pyr_gen[0], im1_pyr_gen[1], im1_pyr_gen[2]]
# im2_pyr = [im2_pyr_gen[0], im2_pyr_gen[1], im2_pyr_gen[2]]
#
#
# im1_features = find_features(im1_pyr)
# im2_features = find_features(im2_pyr)
#
# points_to_draw_im1 = im1_features[0]
# points_to_draw_im2 = im2_features[0]
#
# matches = match_features(im1_features[1], im2_features[1], min_score=0.5)
#
# pos1 = points_to_draw_im1[matches[0]]
# pos2 = points_to_draw_im2[matches[1]]
#
# inliers = ransac_homography(points1=pos1,
#                             points2=pos2,
#                             num_iter=100,
#                             inlier_tol=.5)
# display_matches(im1, im2, pos1, pos2, inliers[1])
