import matplotlib.pyplot as plt
from align_image_code import align_images

# First load images

# # high sf
# im2 = plt.imread('./DerekPicture.jpg')/255.

# # low sf
# im1 = plt.imread('./nutmeg.jpg')/255

im1 = plt.imread('./14.jpg')
im2 = plt.imread('./13.jpg')

# im1 = plt.imread('./9.png')
# im2 = plt.imread('./10.png')

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

print(im1.min(), im1.max())

plt.imsave('im1_aligned.png', im1_aligned)
plt.imsave('im2_aligned.png', im2_aligned)

# ## You will provide the code below. Sigma1 and sigma2 are arbitrary 
# ## cutoff values for the high and low frequencies

# sigma1 = arbitrary_value_1
# sigma2 = arbitrary_value_2
# hybrid = hybrid_image(im1, im2, sigma1, sigma2)

# plt.imshow(hybrid)
# plt.show

# ## Compute and display Gaussian and Laplacian Pyramids
# ## You also need to supply this function
# N = 5 # suggested number of pyramid levels (your choice)
# pyramids(hybrid, N)