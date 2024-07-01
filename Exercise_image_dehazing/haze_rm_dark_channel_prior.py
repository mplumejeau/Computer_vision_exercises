import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Calculate the dark channel of the image
def dark_channel_prior(image, size=15):
    min_img = cv2.min(cv2.min(image[:, :, 0], image[:, :, 1]), image[:, :, 2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel

# Estimate the atmospheric light in the image
def atmospheric_light(image, dark_channel):
    flat_image = image.reshape(-1, image.shape[2])
    flat_dark = dark_channel.ravel()
    search_idx = (-flat_dark).argsort()[:int(0.001 * len(flat_dark))]
    atmospheric_light = np.mean(flat_image[search_idx], axis=0)
    return atmospheric_light

# Estimate the transmission map of the image
def transmission_estimate(image, atmospheric_light, omega=0.95, size=15):
    norm_image = image / atmospheric_light
    transmission = 1 - omega * dark_channel_prior(norm_image, size)
    return transmission

# Apply a guided filter to refine the transmission map
def guided_filter(image, p, radius=60, eps=1e-3):

    mean_I = cv2.boxFilter(image, cv2.CV_64F, (radius, radius))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (radius, radius))
    mean_Ip = cv2.boxFilter(image * p, cv2.CV_64F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(image * image, cv2.CV_64F, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

    q = mean_a * image + mean_b
    return q

# Recover the image by removing haze
def recover(image, t, A, t0=0.1):
    t = np.maximum(t, t0)
    t = t[:, :, np.newaxis]  # Add a new axis to match image dimensions
    J = (image - A) / t + A
    J = np.clip(J, 0, 1)
    return J

# Main function to remove haze from an image
def dehaze(image, size, omega, t0):
    
    image = image.astype('float64') / 255
    dark_channel = dark_channel_prior(image, size)
    A = atmospheric_light(image, dark_channel)
    t_estim = transmission_estimate(image, A, omega, size)

    # Convert image to uint8 with cv2.cvtColor
    gray_image = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_BGR2GRAY)
    t_guided = guided_filter(gray_image, t_estim)
    
    J = recover(image, t_guided, A, t0)
    return (J * 255).astype('uint8'), dark_channel, A, t_estim, t_guided


####### MAIN #######

image_name = 'road3'

folder_path = 'Exercise_image_dehazing/Single_image_dehazing/'

image_path = folder_path + 'images/' + image_name + '.jpg'
output_path = folder_path + 'results/' + image_name
dark_channel_path = folder_path + 'dark_channels/' + image_name + '.jpg'

image = cv2.imread(image_path)

# compute and store 27 dehazed images with variations in size, omega and t0 parameters

for size in range(5, 20, 5):

    for omega in range(85, 100, 5):

        for t0 in range(10, 25, 5):

            dehazed_image, dark_channel, atmo_light, t_estim, t_guided = dehaze(image, size, omega/100, t0/100)

            output_path_real = output_path + '/' + str(size) + '_' + str(omega) + '_' + str(t0) + '.jpg'

            cv2.imwrite(output_path_real, dehazed_image)

# show partial results

cv2.imshow('dark channel', dark_channel)
print('atmospheric light : ' + str(atmo_light))
cv2.imshow('estimated transmission map', t_estim)
cv2.imshow('guided transmission map', t_guided)

# show 27 dehazed images with matplotlib 3x3 grid

image111 = mpimg.imread(output_path + '/5_85_10.jpg')
image112 = mpimg.imread(output_path + '/5_85_15.jpg')
image113 = mpimg.imread(output_path + '/5_85_20.jpg')

image121 = mpimg.imread(output_path + '/5_90_10.jpg')
image122 = mpimg.imread(output_path + '/5_90_15.jpg')
image123 = mpimg.imread(output_path + '/5_90_20.jpg')

image131 = mpimg.imread(output_path + '/5_95_10.jpg')
image132 = mpimg.imread(output_path + '/5_95_15.jpg')
image133 = mpimg.imread(output_path + '/5_95_20.jpg')

image211 = mpimg.imread(output_path + '/10_85_10.jpg')
image212 = mpimg.imread(output_path + '/10_85_15.jpg')
image213 = mpimg.imread(output_path + '/10_85_20.jpg')

image221 = mpimg.imread(output_path + '/10_90_10.jpg')
image222 = mpimg.imread(output_path + '/10_90_15.jpg')
image223 = mpimg.imread(output_path + '/10_90_20.jpg')

image231 = mpimg.imread(output_path + '/10_95_10.jpg')
image232 = mpimg.imread(output_path + '/10_95_15.jpg')
image233 = mpimg.imread(output_path + '/10_95_20.jpg')

image311 = mpimg.imread(output_path + '/15_85_10.jpg')
image312 = mpimg.imread(output_path + '/15_85_15.jpg')
image313 = mpimg.imread(output_path + '/15_85_20.jpg')

image321 = mpimg.imread(output_path + '/15_90_10.jpg')
image322 = mpimg.imread(output_path + '/15_90_15.jpg')
image323 = mpimg.imread(output_path + '/15_90_20.jpg')

image331 = mpimg.imread(output_path + '/15_95_10.jpg')
image332 = mpimg.imread(output_path + '/15_95_15.jpg')
image333 = mpimg.imread(output_path + '/15_95_20.jpg')

images1 = [
    image111,
    image112,
    image113,
    image121,
    image122,
    image123,
    image131,
    image132,
    image133
]

images2 = [
    image211,
    image212,
    image213,
    image221,
    image222,
    image223,
    image231,
    image232,
    image233
]

images3 = [
    image311,
    image312,
    image313,
    image321,
    image322,
    image323,
    image331,
    image332,
    image333
]

fig1, axs1 = plt.subplots(3, 3, figsize=(15, 15))
fig2, axs2 = plt.subplots(3, 3, figsize=(15, 15))
fig3, axs3 = plt.subplots(3, 3, figsize=(15, 15))

for i, ax in enumerate(axs1.flat):
    ax.imshow(images1[i])
    ax.axis('off')

for i, ax in enumerate(axs2.flat):
    ax.imshow(images2[i])
    ax.axis('off')

for i, ax in enumerate(axs3.flat):
    ax.imshow(images3[i])
    ax.axis('off')

plt.tight_layout()

plt.show()
