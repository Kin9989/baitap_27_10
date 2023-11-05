from flask import Flask, render_template, url_for
import numpy as np
from PIL import Image
import imageio.v2 as iio
import matplotlib.pylab as plt
import cv2
import scipy

app = Flask(__name__)


@app.route("/")
def index():
    img_path = url_for("static", filename="images/image.jpg")
    img = Image.open("C:/Users/kinn/Desktop/baitap1_2/static/images/image.jpg").convert(
        "L"
    )
    im_1 = np.asarray(img)
    im_2 = 255 - im_1
    new_img = Image.fromarray(im_2)
    new_img.save("static/images/inverse_image.jpg")

    gamma = 0.5
    b1 = im_1.astype(float)
    b2 = np.max(b1)
    b3 = b2 / b1
    b2 = np.log(b3) * gamma
    c = np.exp(b2) * 255.0
    c1 = c.astype(int)
    gamma_corrected_img = Image.fromarray(c1.astype(np.uint8))
    gamma_corrected_img.save("static/images/gamma_corrected_image.jpg")

    b1 = im_1.astype(float)
    b2 = np.max(b1)
    c = 128.0 * np.log(1 + b1) / np.log(1 + b2)
    c1 = c.astype(int)
    log_transformed_img = Image.fromarray(c1.astype(np.uint8))
    log_transformed_img.save("static/images/log_transformed_image.jpg")

    iml = np.asarray(img)
    bl = iml.flatten()
    hist, bins = np.histogram(iml, 256, [0, 255])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    num_cdf_m = (cdf_m - cdf_m.min()) * 255
    den_cdf_m = (cdf.max() - cdf_m.min()) * 255
    cdf_m = num_cdf_m / den_cdf_m
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")
    im2 = cdf[bl]
    im3 = np.reshape(im2, iml.shape)
    histogram_equalized_img = Image.fromarray(im3)
    histogram_equalized_img.save("static/images/histogram_equalized_image.jpg")

    a, b = iml.min(), iml.max()
    c = iml.astype(float)
    im2 = 255 * (c - a) / (b - a)
    contrast_stretched_img = Image.fromarray(im2.astype(np.uint8))
    contrast_stretched_img.save("static/images/contrast_stretched_image.jpg")

    c = abs(scipy.fftpack.fft2(iml))
    d = scipy.fftpack.fftshift(c)
    d = d.astype(float)

    # Convert to 8-bit image before saving as JPEG
    d_normalized = (d - np.min(d)) / (np.max(d) - np.min(d)) * 255.0
    d_8bit = d_normalized.astype(np.uint8)

    fft_img = Image.fromarray(d_8bit)
    fft_img.save("static/images/fft_image.jpg")

    # Unique filenames for each processed image
    converted_fft_img = fft_img.convert("L")
    converted_fft_img.save("static/images/converted_fft_image.jpg")

    img = Image.open("static/images/image.jpg").convert("L")
    processed_img = 255 - np.asarray(img)
    processed_img = Image.fromarray(processed_img)
    processed_img.save("static/images/processed_image.jpg")

    return render_template("index.html", processed_image="inverse_image.jpg")


if __name__ == "__main__":
    app.run(debug=True)
