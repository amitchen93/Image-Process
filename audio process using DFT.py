import numpy as np
import scipy.io.wavfile as wav
# for helper functions:
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from matplotlib import pyplot as plt

import skimage
from skimage.color import rgb2gray


""" DFT part """


def DFT(signal):
    """
    :param signal: an array of dtype float64 with shape (N,1) (technically it’s 2D)
    :return: the complex Fourier signal and complex signal, respectively, both of shape (N,1).
    """
    n = signal.shape[0]
    range_of_N = np.arange(n)  # range of N
    use_vandar = np.vander(np.exp((-2j * range_of_N * np.pi) / n), increasing=True)
    after_DFT = np.dot(use_vandar, signal)  # Dot product of two arrays
    return after_DFT.astype(np.complex128)   # array of dtype complex128


def IDFT(fourier_signal):
    """
    inverse DFT (IDFT)
    :param fourier_signal: an array of dtype float64 with shape (N,1) (technically it’s 2D)
    :return: the complex Fourier signal and complex signal, respectively, both of shape (N,1).
    """
    n = fourier_signal.shape[0]
    range_of_N = np.arange(n)  # range of N

    # "You should use the following formulas"
    # weight = np.exp((2j * np.pi) / n)  # doesn't work for decimal 5 precise (only 3)
    use_vandar = np.vander(np.exp((2j * range_of_N * np.pi) / n), increasing=True)
    after_IDFT = np.dot(use_vandar, fourier_signal)  # Dot product of two arrays
    after_IDFT_normalized = after_IDFT / n  # normalized by N
    return after_IDFT_normalized.astype(np.complex128)  # array of dtype complex128


def DFT2(image):
    """
    convert a 2D discrete signal to its Fourier representation
    :param image: a grayscale image of dtype float64, shape (M,N).
    :return: Fourier representation
    """
    image = DFT(image).T
    return DFT(image).T  # another DFT on second variable

def IDFT2(fourier_image):
    """
    convert a Fourier representation to its 2D discrete signal
    :param fourier_image: a 2D array of dtype complex128 shape (M,N).
    :return: 2D discrete signal
    """
    fourier_image = IDFT(fourier_image).T
    return IDFT(fourier_image).T  # another IDFT on second variable


""" Speech Fast Forward part """


def change_rate(filename, ratio):
    """
    changes the duration of an audio file by keeping the same samples, but changing the
    sample rate written in the file header. You may assume that 0.25 < ratio < 4
    :param filename:  a string representing the path to a WAV file,
    :param ratio: a positive float64 representing the duration change
    :return: None
    """
    sample_rate, data = wav.read(filename)  # Return the sample rate (in samples/sec) and data from a WAV file

    # change sample rate: if the original sample rate is 4,000Hz
    # and ratio is 1.25, then the new sample rate will be 5,000Hz
    changed_sample_rate = int(sample_rate * ratio)

    # Write a NumPy array as a WAV file with new name and new sample rate.
    wav.write("change_rate.wav", changed_sample_rate, data)

    # The function should not return anything


""" Fast forward using Fourier part """


def change_samples(filename, ratio):
    """
    a fast forward function that changes the duration of an audio file by reducing the number of samples using Fourier.
    This function does not change the sample rate of the given file.
    The result should be saved in a file called change_samples.wav.
    :param filename: a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change
    :return: 1D ndarray of dtype float64 representing the new sample points
    """
    sample_rate, data = wav.read(filename)  # Return the sample rate (in samples/sec) and data from a WAV file

    # call the function resize to change the number of samples by the given ratio
    resized_data = np.real(resize(data, ratio)).astype('int16')


    # Write a NumPy array as a WAV file with new name and new sample.
    wav.write("change_samples.wav", sample_rate, resized_data)

    return resized_data.astype('float64')


def resize(data, ratio):
    """
    change the number of samples by the given ratio (0.25 < ratio < 4)
    :param data: a 1D ndarray of dtype float64 or complex128(*) representing the original sample points
    :param ratio: as "change_samples"
    :return: a 1D ndarray of the dtype of data representing the new sample points
    """
    # same ratio
    if ratio == 1:
        return data

    # case of empty data
    if (len(data) == 0) or ((data.shape[0] // ratio) == 0):
        return np.array([])

    samples_count = data.shape[0]

    # DFT and shift
    dft_of_data = DFT(data)
    shift_dft_data = np.fft.fftshift(dft_of_data)
    round_samples = np.round(samples_count // ratio).astype(int)

    if ratio > 1:
        min_idx = np.ceil((samples_count - round_samples) / 2).astype(int)
        resized_dft = shift_dft_data[min_idx: min_idx + round_samples]

    elif ratio < 1:  # pad with zeroes
        add_to_pad = (round_samples - samples_count)

        if add_to_pad % 2 != 0:  # not equal / not even
            pad = (add_to_pad // 2, add_to_pad // 2 + 1)

        else:  # equal / even
            pad = (add_to_pad // 2, add_to_pad // 2)

        if len(data.shape) == 1:  # sound data file with len=1
            resized_dft = np.pad(shift_dft_data, pad, 'constant', constant_values=0)

        else:
            resized_dft = np.pad(shift_dft_data, (pad, (0, 0)), 'constant', constant_values=0)

    resized_dft = np.fft.ifftshift(resized_dft)  # shift back
    resized_data = IDFT(resized_dft)  # inverse IDFT

    return resized_data


def resize_spectrogram(data, ratio):
    """
    speeds up a WAV file, without changing the pitch, using spectrogram scaling. This is done by computing
    the spectrogram, changing the number of spectrogram columns, and creating back the audio
    :param data:  1D ndarray of dtype float64 representing the original sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return: new sample points according to ratio with the same datatype as data.
    """
    # This function should use the provided functions stft and istft in order to transfer the data to the
    # spectrogram and back:
    return_from_stft = stft(data)

    # Each row in the spectrogram can be resized using resize according to ratio
    resized_stft = resize((return_from_stft).T, ratio)
    resized_stft_transposed = resized_stft.T
    apply_istft = istft(resized_stft_transposed)
    clean_sample_points = np.round(apply_istft).astype(np.int16)
    return clean_sample_points


def resize_vocoder(data, ratio):
    """
    speedups a WAV file by phase vocoding its spectrogram
    :param data: a 1D ndarray of dtype float64 representing the original sample points
    :param ratio: positive float64 representing the rate change of the WAV file
    :return:  the given data rescaled according to ratio with the same datatype as data
    """
    return_from_stft = stft(data)

    # You can use the supplied function phase_vocoder(spec, ratio), which scales the spectrogram spec
    # by ratio and corrects the phases
    return_from_phase_vocoder = phase_vocoder(return_from_stft, ratio)
    apply_istft = istft(return_from_phase_vocoder)
    clean_sample_points = np.round(apply_istft).astype(np.int16)
    return clean_sample_points


"""Part 3 - Image derivatives"""

def conv_der(im):
    """
    computes the magnitude of image derivatives
    :param im: grayscale images of type float64
    :return:  magnitude of the derivative, with the same dtype and shape
    """
    #  using simple convolution with [0.5, 0, −0.5]  in proper size (3X3) as a row and column vectors

    matrix_for_x = np.array([[0,0,0], [0.5,0,-0.5], [0,0,0]])
    matrix_for_y = np.array([[0,0.5,0], [0,0,0], [0,-0.5,0]])

    # Convolve two 2-dimensional arrays. mode = The output is the same size as in1
    dx = signal.convolve2d(im, matrix_for_x, mode='same')

    # same on Y:
    dy = signal.convolve2d(im, matrix_for_y, mode='same')

    # use this formula :   magnitude = np.sqrt (np.abs(dx)**2 + np.abs(dy)**2)
    magnitude = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)

    # you should not normalize the magnitude values to be in the range of [0,1], just return the values you get.
    return magnitude


def fourier_der(im):
    """
    computes the magnitude of the image derivatives using Fourier transform
    :param im: float64 grayscale image
    :return: the magnitude of the image, float64 grayscale image
    """
    # multiply the frequencies in the range [−N/2, ..., N/2], You may not assume the image is square (dif x and y)
    x_size, y_size = (im.shape[0] // 2), (im.shape[1] // 2)
    u_range, v_range = np.arange(-x_size, x_size), np.arange(-y_size, y_size)

    # we get image - so it's 2D (use DFT and np.fft.fftshift):
    image_after_DFT = DFT2(im)
    image_shifted = np.fft.fftshift(image_after_DFT)  # so (U,V)=(0,0) frequency will be at the center of the image

    # derivatives by formula from class:
    shaped_u = u_range.reshape(im.shape[0], 1)
    dx = (np.pi * image_shifted * shaped_u * 2j / im.shape[1])
    dy = np.pi * image_shifted * v_range * 2j / im.shape[0]

    dx_shift_back = np.fft.ifftshift(dx)
    dy_shift_back = np.fft.ifftshift(dy)

    # Use IDFT
    dx_after_IDFT = IDFT2(dx_shift_back)
    dy_after_IDFT = IDFT2(dy_shift_back)

    # magnitude = np.sqrt (np.abs(dx)**2 + np.abs(dy)**2)
    magnitude = np.sqrt(np.abs(dx_after_IDFT) ** 2 + np.abs(dy_after_IDFT) ** 2)

    # you should not normalize the magnitude values to be in the range of [0,1], just return the values you get.
    return magnitude



""" my read_image() from ex1: """

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




"""provided helper function from ex2_helper:"""


def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec
