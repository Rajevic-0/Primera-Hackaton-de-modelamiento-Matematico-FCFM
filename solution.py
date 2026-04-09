"""
Ofuscation of fingerprints:
- Elastical deformation 
- Random local swirls
- Ridge noise
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, convolve
import cv2  

# ----------------------------------------------------------------------
# 1.Applies n=passes layers of elastical deformation
# ----------------------------------------------------------------------
def multi_scale_elastic_deform(image, alpha=15, sigma=2.8, passes=3, seed=None):

    rng = np.random.RandomState(seed)
    img = image.astype(np.float32)
    h, w = img.shape

    for _ in range(passes):
        dx = rng.rand(h, w) * 2 - 1
        dy = rng.rand(h, w) * 2 - 1
        dx = gaussian_filter(dx, sigma, mode='constant') * alpha
        dy = gaussian_filter(dy, sigma, mode='constant') * alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        img = map_coordinates(img, indices, order=3, mode='nearest').reshape(h, w)

    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# 2. Applies a local swirl arround a point.
# ----------------------------------------------------------------------
def local_swirl(image, center, radius, strength, seed=None):

    h, w = image.shape
    cx, cy = center
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)

    factor = np.clip(1.0 - r / radius, 0.0, 1.0)

    angle = strength * factor

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = cx + cos_a * dx - sin_a * dy
    y_rot = cy + sin_a * dx + cos_a * dy

    mask = r < radius
    x_new = np.where(mask, x_rot, x)
    y_new = np.where(mask, y_rot, y)

    deformed = map_coordinates(image.astype(np.float32), [y_new, x_new], order=3, mode='nearest')

    return np.clip(deformed, 0, 255).astype(image.dtype)


# ----------------------------------------------------------------------
# 3. Applies num_swirls random swirls around the fingerprint. 
# ----------------------------------------------------------------------
def apply_random_swirls(image, num_swirls=3, radius_range=(30, 60),
                        strength_range=(0.8, 1.5), seed=None):
    rng = np.random.RandomState(seed)
    img = image.copy()
    h, w = img.shape

    margin = int(min(radius_range[1], min(h, w) * 0.1))

    for _ in range(num_swirls):
        cx = rng.randint(margin, w - margin)
        cy = rng.randint(margin, h - margin)
        radius = rng.uniform(*radius_range)
        strength = rng.uniform(*strength_range)
        img = local_swirl(img, (cx, cy), radius, strength, seed=None)

    return img


# ----------------------------------------------------------------------
# 4. Adds random Gabor noise.
# ----------------------------------------------------------------------
def add_ridge_noise(image, strength=2.5, kernel_size=7, seed=None):
    rng = np.random.RandomState(seed)
    angle = rng.uniform(0, np.pi)

    sigma = 2.0
    lambd = 10.0
    gamma = 0.5
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, angle, lambd, gamma, ktype=cv2.CV_32F)
    if kernel is None:
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)

    noise = rng.randn(*image.shape).astype(np.float32) * strength
    filtered_noise = convolve(noise, kernel, mode='reflect')

    noisy = image.astype(np.float32) + filtered_noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# 5. Full pipeline
# ----------------------------------------------------------------------
def obfuscate_fingerprint(image,
                          alpha=15, sigma=2.8, passes=3,
                          use_swirls=True, num_swirls=3,
                          swirl_radius_min=30, swirl_radius_max=60,
                          swirl_strength_min=0.8, swirl_strength_max=1.5,
                          ridge_noise_strength=2.0,
                          seed=None):
    """
    Obfuscates a fingerprint using:
      1. Elastical deformation  (breaks Minutiae globally)
      2. Local swirls (perturbates deltas)
      3. Ridge noise (adds texture to fingerprint lines)
    """
    # Rng 
    rng = np.random.RandomState(seed)

    # 1: Elastical deformation
    deformed = multi_scale_elastic_deform(image, alpha=alpha, sigma=sigma,
                                          passes=passes, seed=rng.randint(0, 2**31))

    # 2: Local swirls
    if use_swirls:
        deformed = apply_random_swirls(deformed,
                                       num_swirls=num_swirls,
                                       radius_range=(swirl_radius_min, swirl_radius_max),
                                       strength_range=(swirl_strength_min, swirl_strength_max),
                                       seed=rng.randint(0, 2**31))

    # 3: Ridge noise 
    if ridge_noise_strength > 0:
        deformed = add_ridge_noise(deformed, strength=ridge_noise_strength,
                                   seed=rng.randint(0, 2**31))

    return deformed

# ----------------------------------------------------------------------
# 6 Main
# ----------------------------------------------------------------------

def main(image):
    return obfuscate_fingerprint(
        image,
        alpha=20, sigma=5, passes=3,
        use_swirls=True, num_swirls=9,
        swirl_radius_min=5, swirl_radius_max=20,
        swirl_strength_min=0.6, swirl_strength_max=1.3,
        ridge_noise_strength=2.5,
        seed=42
    )


# ----------------------------------------------------------------------
# 7 Testing.
# ----------------------------------------------------------------------
"""
def test()
    img = cv2.imread('a.tif',0)
    img = main(img)
    cv2.imwrite('b.tif',img)

test()
"""

