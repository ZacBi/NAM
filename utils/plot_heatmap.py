import cv2
from PIL import Image
import numpy as np

# shape of img_tensor: (256, 256, 3)
# shape of activation: (1, 256)
image = Image.fromarray(np.uint8(img_tensor.numpy())).convert('RGBA')
new_activation = bilinear_interpolation(activation.reshape(16, 16, 1), img_tensor.shape[0], img_tensor.shape[1]) # (16, 16, 1) to (256, 256, 1)

# plot heatmap
new_activation_ = (new_activation - new_activation.min()) / (new_activation.max() - new_activation.min())
heatmap = cv2.applyColorMap(np.uint8(255 * new_activation_), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
heatmap = Image.fromarray(heatmap).convert('RGBA')
heatmap.putalpha(int(0.5 * 255))
new_image = Image.alpha_composite(image, heatmap).convert('RGB')
new_image.save(f'{save_dir}/heatmap}.png')

# plot binary mask
thres = np.percentile(new_activation, 95)
pos = np.where(new_activation > thres)
mask = Image.new('RGBA', (img_tensor.shape[0], img_tensor.shape[1]), (0, 0, 0, 200))
mask_pixels = mask.load()
for i, j in zip(pos[0], pos[1]):
    mask_pixels[j, i] = (255, 255, 255, 0)
new_image = Image.alpha_composite(image, mask).convert('RGB')
new_image.save(f'{save_dir}/binary_mask.png')