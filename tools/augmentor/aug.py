import Augmentor
p = Augmentor.Pipeline("C:/Users/User/Desktop/augmentor/original")
p.ground_truth("C:/Users/User/Desktop/augmentor/mask")
p.flip_left_right(probability=0.3)
p.gaussian_distortion(probability=0.5, grid_width=5, grid_height=5, magnitude=5, corner="bell", method="in", mex=0.5, mey=0.5, sdx=0.05, sdy=0.05)

#p.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)
#p.random_color(probability=0.5, min_factor=0.5, max_factor=1.5)
#p.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)

p.random_distortion(probability=0.5, grid_width=5, grid_height=5, magnitude=5)
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.skew(probability=0.5, magnitude=0.5)
p.zoom_random(probability=0.3, percentage_area=0.5, randomise_percentage_area=False)
p.resize(probability=1, width=256, height=256, resample_filter=u'BICUBIC')
p.sample(3999)

p = Augmentor.Pipeline("C:/Users/User/Desktop/augmentor/original/output")
p.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)
p.random_color(probability=0.5, min_factor=0.5, max_factor=1.5)
p.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)
p.process()