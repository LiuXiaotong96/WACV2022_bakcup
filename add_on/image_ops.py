import cv2
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def register_colorbar():
    ncolors = 256
    color_array = plt.get_cmap('gist_rainbow')(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(1.0,0.0,ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

    plt.register_cmap(cmap=map_object)

def add_green_border(img):
    img[:10,:,0] = 0.
    img[:10,:,1] = 255.
    img[:10,:,2] = 0.
    img[-10:,:,0] = 0.
    img[-10:,:,1] = 255.
    img[-10:,:,2] = 0.
    img[:,:10,0] = 0.
    img[:,:10,1] = 255.
    img[:,:10,2] = 0.
    img[:,-10:,0] = 0.
    img[:,-10:,1] = 255.
    img[:,-10:,2] = 0.
    return img

def add_red_border(img):
    img[:10,:,0] = 255.
    img[:10,:,1] = 0.
    img[:10,:,2] = 0.
    img[-10:,:,0] = 255.
    img[-10:,:,1] = 0.
    img[-10:,:,2] = 0.
    img[:,:10,0] = 255.
    img[:,:10,1] = 0.
    img[:,:10,2] = 0.
    img[:,-10:,0] = 255.
    img[:,-10:,1] = 0.
    img[:,-10:,2] = 0.
    return img

def load_and_resize(im_path):
    """
    Loads an image and resizes to 256x256.
    """
    bgr_img = cv2.imread(im_path)
    bgr_img = cv2.resize(bgr_img, (256,256))
    return bgr_img

def preprocess_im(im_path,mean_im_path):
    """
    Takes in an the path to an image and a numpy array of the mean image for the dataset.
    Resizes the image to 224x224, subtracts off the mean image.
    Everything stays in BGR.
    """
    bgr_img = load_and_resize(im_path)
    mean_img = np.load(mean_im_path)
    img = bgr_img - mean_img
    return img

def pil_bgr_to_rgb(img):
    b, g, r = img.split()
    return Image.merge("RGB", (r, g, b))

# def combine_image_and_heatmap(img,heatmap):
#     """
#     Takes in a numpy array for an image and the similarity heatmap.
#     Blends the two images together and returns a np array of the blended image.
#     """
#     cmap = plt.get_cmap('jet') # colormap for the heatmap
#     heatmap = heatmap - np.min(heatmap)
#     heatmap /= np.max(heatmap)
#     heatmap = cmap(np.max(heatmap)-heatmap)
#     if np.max(heatmap) < 255.:
#         heatmap *= 255

#     heatmap_img = cv2.resize(heatmap,(256,256))
#     bg = Image.fromarray(img.astype('uint8')).convert('RGBA')
#     fg = Image.fromarray(heatmap_img.astype('uint8')).convert('RGBA')
#     outIm = np.array(Image.blend(bg,fg,alpha=0.5))
#     return outIm

def combine_image_and_heatmap(img,heatmap,lower_bound,upper_bound):
    """
    Takes in a numpy array for an image and the similarity heatmap.
    Blends the two images together and returns a np array of the blended image.
    """
    register_colorbar()
    #cmap = plt.get_cmap('jet') # colormap for the heatmap
    cmap = plt.get_cmap('rainbow_alpha')
    heatmap = heatmap - lower_bound
    heatmap /= upper_bound
    heatmap = cmap(np.max(heatmap)-heatmap)
    if np.max(heatmap) < 255.:
        heatmap *= 255

    heatmap_img = cv2.resize(heatmap,(256,256))
    bg = Image.fromarray(img.astype('uint8')).convert('RGBA')
    fg = Image.fromarray(heatmap_img.astype('uint8')).convert('RGBA')
    outIm = np.array(Image.blend(bg,fg,alpha=0.5))
    return outIm

def combine_horz(images):
    """
    Combines images into a single side-by-side PIL image object.
    """
    images = [Image.fromarray(img.astype('uint8')) for img in images]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

def combine_vert(imagelist):
    """
    Combines images into a single stacked PIL image object.
    """
    images_ = []
    for img in imagelist:
        images_.append(img.astype('uint8'))
    images = []
    for img in images_:
        images.append(Image.fromarray(img))
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0,y_offset))
        y_offset += im.size[1]
    return new_im