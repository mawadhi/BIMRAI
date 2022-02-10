import Augmentor
p = Augmentor.Pipeline("C:/Users/User/Desktop/augmentor/original/output")
p.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)
p.random_color(probability=0.5, min_factor=0.5, max_factor=1.5)
p.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)
p.process()
