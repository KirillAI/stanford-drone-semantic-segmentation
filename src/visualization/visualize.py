def draw_segmentation_on_image(image, mask):
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    masked_image = np.copy(image)
    masked_image[(mask > 0).all(-1)] = [50, 20, 200]
    mask[..., 0] *= 50
    mask[..., 1] *= 100
    mask[..., 2] *= 20
    new_mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    masked_image += new_mask
    masked_image_w = cv2.addWeighted(masked_image, 0.5, image, 0.5, 0, masked_image)
    _ = plt.imshow(masked_image_w)
    plt.show()