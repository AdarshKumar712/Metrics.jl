# Image Quality Metrics

"""
    PSNR(img1, img2)

Computes peak-signal-to-noise ratio, in decibels, between two images `img1` and `img2`. The higher the PSNR, the better the quality of the compressed, or reconstructed image.
"""
function PSNR(img1, img2)
    mse = mean((img1 .- img2).^2)
    psnr = 255 * 255 / mse
    return 10 * log10(psnr)
end

# TODO: Add following functions
# Inception Score        
# Frechet Inception Distance
