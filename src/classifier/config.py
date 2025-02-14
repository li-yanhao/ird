import os 

class Config:
    def __init__(self):
        self.img_size = 256

        self.model = "efficient-b0"

        # online data creation
        self.r_range_list = [
            (0.52, 0.66),
            (0.66, 1.0),
            (1.0, 2.0),
            (2.0, 4.0),
        ]
        
        # self.unaltered_image = True
        self.num_class = len(self.r_range_list)

        # self.antialias_options = ["none", "kernel_stretch"]  # useless
        # self.interp_options = ["bicubic", "bilinear"]  # useless
        # self.post_jpeg_Q_range = [95, 100]


        # self.library_options = None
        # self.antialias_options = [True, False]
        # self.fixed_resampling_factor = None
        # self.interp_options = None

        # preprocess        
        self.preprocess = "none"

        # self.ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt/best-epoch=60-val_loss=0.49.ckpt")
        self.ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt/best-epoch.ckpt")