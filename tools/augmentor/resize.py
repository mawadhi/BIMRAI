import Augmentor
p = Augmentor.Pipeline("C:/Users/User/Desktop/augmentor/original")
p.ground_truth("C:/Users/User/Desktop/augmentor/mask")
p.resize(probability=1, width=256, height=256, resample_filter=u'BICUBIC')
p.process()