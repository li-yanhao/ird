# OPTION 1 : Add noise to the input image and then denoise with dynamic lambda
    
    option=1    
    input=traffic.png
    sigma=10.0
    lambda=0.0
    
    ./chambolle_ipol $option $input $sigma $lambda noisy.png denoised.png
    ./imdiff_ipol $input denoised.png difference.png $sigma

# OPTION 2 : Add noise to the input image and then denoise with fixed lambda
    
    option=2    
    input=traffic.png
    sigma=10.0
    lambda=0.08
    
    ./chambolle_ipol $option $input $sigma $lambda noisy.png denoised.png
    ./imdiff_ipol $input denoised.png difference.png $sigma

# OPTION 3 : Denoise noisy image with dynamic lambda
    
    option=3   
    input=traffic_noisy.png
    sigma=10.0
    lambda=0.0
    
    ./chambolle_ipol $option $input $sigma $lambda noisy.png denoised.png
    ./imdiff_ipol traffic.png denoised.png difference.png $sigma

# OPTION 4 : Denoise noisy image with fixed lambda
    
    option=4    
    input=traffic_noisy.png
    sigma=10.0
    lambda=0.08
    
    ./chambolle_ipol $option $input $sigma $lambda noisy.png denoised.png
    ./imdiff_ipol traffic.png denoised.png difference.png $sigma
