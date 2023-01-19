import os
from os.path import join
import argparse
import yaml
import numpy as np
import pdf2image
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import cv2
from PIL import Image

from vidocp.utils.deskew import deskew_histbased#, deskew_linebased
from vidocp.utils.display import show_mpl
from vidocp.utils.draw import draw_stats
from vidocp.table_parsing import parse_table


def open_pdf(pdf_path: str, page_index=None) -> list[np.array]:
    if page_index:
        pages = pdf2image.convert_from_path(pdf_path, first_page=page_index + 1, last_page=page_index + 1)
    else:
        pages = pdf2image.convert_from_path(pdf_path)
    def convert(page):
        page = np.array(page)
        page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        page = cv2.fastNlMeansDenoising(page, h=3)
        return page
    return [convert(page) for page in pages]
    

def rename(base_path, folder_name):
    names = os.listdir(join(base_path, folder_name))
    name_dict = dict(enumerate(names))
    reflist = []
    for k, v in name_dict.items():
        os.rename(join(base_path, folder_name, v), join(base_path, folder_name, f"{k:02}.pdf"))
        reflist.append(f"{k:02}\t{v}")
    with open(join(base_path, f"{folder_name}_key.txt"), "w") as f:
        f.write('\n'.join(reflist))


def save_array_as_pdf():
    pass


def read_document(pdf_path):
    pass


def make_histogram(array, nbins, show=False):
    histogram = np.histogram(array, bins=nbins)
    if show:
        _ = plt.hist(array, bins=nbins)
        plt.show()
    return histogram


def split_rowmean_diff(page):
    width = page.shape[1]
    cutpoint = int(width / 2)
    left = page[:, :cutpoint]
    right = page[:, cutpoint:]
    leftmeans = np.mean(left, axis=1)
    rightmeans = np.mean(right, axis=1)
    return rightmeans - leftmeans


def split4_rowmean_diff(page):
    height, width = page.shape
    output = np.zeros((height, 4))
    cutpoint = int(width / 4)
    output[:, 0] = np.mean(page[:, :cutpoint], axis=1)
    output[:, 1] = np.mean(page[:, cutpoint:2*cutpoint], axis=1)
    output[:, 2] = np.mean(page[:, 2*cutpoint:3*cutpoint], axis=1)
    output[:, 3] = np.mean(page[:, 3*cutpoint:], axis=1)
    return output


def bar(array):
    plt.bar(range(len(array)), array)
    plt.show()


def rotate(page, angle):
    rotated = rotate(page, angle, reshape=False, order=0, mode="nearest")
    return rotated


def preprocess():
    pass




if __name__ == "__main__":
    with open("default.yml") as f:
        cfg = yaml.load(f)
    rename(join(cfg.pdfdir, "scanned"))
    rename(join(cfg.pdfdir, "unscanned"))
    rename(join(cfg.pdfdir, "partial"))

