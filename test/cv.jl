using Metrics

@testset "CV_Metrics" begin
    
    img1, img2 = rand(10,10), rand(10, 10)
    
    @test PSNR(img1, img2) ≈ 55.842475032823806
    
    box1 = Dict("x1" => 0, "x2" => 6, "y1" => 6, "y2" => 0)
    box2 = Dict("x1" => 2, "x2" => 8, "y1" => 8, "y2" => 2)

    @test IoU(box1, box2) == 0.2857142857142857
    
end
