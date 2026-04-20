using MRIRealign
using Test
using ImagePhantoms
using Interpolations
using StaticArrays
using FiniteDifferences
using LinearAlgebra

## create phantom
shape = (64, 64, 64)
ob = ellipsoid(ellipsoid_parameters(fovs=(64,64,50)))
image = phantom(-shape[1]÷2+1:shape[1]÷2, -shape[2]÷2+1:shape[2]÷2, -shape[3]÷2+1:shape[3]÷2, ob, 3)
image = min.(image, 1.1)
image .-= 0.9
image = max.(image, 0)

center = (20, 10, 5)
width = (25, 35, 15)
angles = (π/6, 0, 0)
ob1 = gauss3(center, width, angles, 1.0f0)

center = (-15, 0, 5)
width = (5, 5, 15)
angles = (0, π/2, 0)
ob2 = gauss3(center, width, angles, 1.0f0)

center = (0, 0, 15)
width = (5, 5, 15)
angles = (0, 0, 0)
ob3 = gauss3(center, width, angles, 1.0f0)

image .+= 0.5 .* phantom(-shape[1]÷2+1:shape[1]÷2, -shape[2]÷2+1:shape[2]÷2, -shape[3]÷2+1:shape[3]÷2, [ob1, ob2, ob3], 3)

# Interpolant on mm axes (voxel_size=1mm, so mm == voxel indices)
test_voxel_size = (1.0, 1.0, 1.0)
img_itp = MRIRealign._interpolate(image, test_voxel_size)

# Helper: create a moved image from known parameters
# p = [rx, ry, rz, tx, ty, tz] with rotations in radians and translations in mm
function make_moved_image(img_itp, p, img_shape; voxel_size=(1.0,1.0,1.0))
    vx, vy, vz = voxel_size
    center_mm = (img_shape .÷ 2) .* voxel_size
    A = MRIRealign.create_affine_matrix(p, center_mm)
    img_moved = similar(Array{Float64}, img_shape)
    for i ∈ CartesianIndices(img_moved)
        coord_mm = SVector{4,Float64}(i[1]*vx, i[2]*vy, i[3]*vz, 1)
        v = A \ coord_mm
        img_moved[i] = abs(img_itp(v[1], v[2], v[3]))
    end
    return img_moved
end


@testset "MRIRealign.jl" begin

@testset "Utility round-trips" begin
    # create_rotation_matrix: orthogonality and determinant
    for rx ∈ (-0.3, 0.0, 0.2), ry ∈ (-0.1, 0.0, 0.15), rz ∈ (0.0, 0.25)
        R = MRIRealign.create_rotation_matrix(rx, ry, rz)
        @test R' * R ≈ I(3) atol = 1e-12
        @test det(R) ≈ 1.0 atol = 1e-12
    end

    # create_affine_matrix / params_from_rigid_affine round-trip
    for c ∈ ((0,0,0), (32,32,32), (10,20,30))
        for p ∈ ([0.1, -0.05, 0.2, 3.0, -2.0, 1.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [-0.2, 0.1, -0.15, -1.0, 0.5, 2.0])
            A = MRIRealign.create_affine_matrix(p, c)
            p_recovered = MRIRealign.params_from_rigid_affine(A, c)
            @test collect(p_recovered) ≈ p atol = 1e-10
        end
    end

    # Quaternion round-trips
    for rx ∈ (-0.3, 0.0, 0.5), ry ∈ (-0.2, 0.0, 0.3), rz ∈ (0.0, -0.4, 0.1)
        R = MRIRealign.create_rotation_matrix(rx, ry, rz)
        q = MRIRealign._rotmat_to_quat(R)
        @test norm(q) ≈ 1.0 atol = 1e-12
        R_back = MRIRealign._quat_to_rotmat(q)
        @test R_back ≈ R atol = 1e-12
    end
end

@testset "Gradient vs. finite differences (at p=0)" begin
    # At p=0 the analytical gradient (which uses grad_field evaluated at reference
    # coordinates) is exact because the transform is the identity.
    # All coordinates are in mm (with voxel_size = 1mm, mm == voxel indices)
    voxel_size = (1.0, 1.0, 1.0)
    inds = [SVector{3,Float64}((Tuple(idx) .+ (rand(), rand(), rand()) .- (0.5, 0.5, 0.5)) .* voxel_size) for idx ∈ CartesianIndices(image)]
    reference = [img_itp(idx[1], idx[2], idx[3]) for idx in inds]
    grad_field = [SVector{3,Float64}(Interpolations.gradient(img_itp, idx[1], idx[2], idx[3])) for idx ∈ inds]
    hess_field = nothing

    c_mm = SVector{3,Float64}((size(image) .÷ 2) .* voxel_size)
    xyz_centered = [ind - c_mm for ind in inds]
    
    radius = 64.0

    img_mov = MRIRealign._interpolate(circshift(image, (10, 10, 10)), voxel_size)
    diff_vals = similar(inds, Float64)
    fgh! = MRIRealign.make_fgh_function(vec(reference), img_mov, c_mm, inds, xyz_centered, grad_field, hess_field, diff_vals, radius)

    G = zeros(6)
    H = zeros(6, 6)
    p0 = zeros(6)
    G_fd = grad(central_fdm(5, 1; factor=1e6), p -> fgh!(nothing, nothing, nothing, p), p0)[1]
    fgh!(nothing, G, H, p0)
    @test G ≈ G_fd rtol = 0.2
end

@testset "Gauss-Newton Hessian properties" begin
    # The Gauss-Newton Hessian is J'J (positive semi-definite and symmetric),
    # not the true Hessian, so we test structural properties rather than
    # comparing to finite differences.
    voxel_size = (1.0, 1.0, 1.0)
    inds = [SVector{3,Float64}((Tuple(idx) .+ (rand(), rand(), rand()) .- (0.5, 0.5, 0.5)) .* voxel_size) for idx ∈ CartesianIndices(image)]
    reference = [img_itp(idx[1], idx[2], idx[3]) for idx in inds]
    grad_field = [SVector{3,Float64}(Interpolations.gradient(img_itp, idx[1], idx[2], idx[3])) for idx ∈ inds]
    hess_field = nothing

    c_mm = SVector{3,Float64}((size(image) .÷ 2) .* voxel_size)
    xyz_centered = [ind - c_mm for ind in inds]

    radius = 64.0

    img_mov = MRIRealign._interpolate(circshift(image, (10, 10, 10)), voxel_size)
    diff_vals = similar(inds, Float64)
    fgh! = MRIRealign.make_fgh_function(vec(reference), img_mov, c_mm, inds, xyz_centered, grad_field, hess_field, diff_vals, radius)

    for p_test ∈ (zeros(6), [0.05, -0.03, 0.04, 1.0, -0.5, 0.5])
        G = zeros(6)
        H = zeros(6, 6)
        fgh!(nothing, G, H, p_test)
        @test H ≈ H' atol = 1e-10           # symmetric
        @test minimum(eigvals(H)) ≥ -1e-8    # positive semi-definite
    end
end

@testset "Translation estimates" begin
    for (ref_mode, shifts) ∈ [(1, [ 0  1  0  0;  0  0  1  0;  0  0  0  1]),
                               (2, [-1  0 -1 -1;  0  0  1  0;  0  0  0  1]),
                               (3, [ 0  1  0  0; -1 -1  0 -1;  0  0  0  1]),
                               (4, [ 0  1  0  0;  0  0  1  0; -1 -1 -1  0])]
        imgs = cat(image, circshift(image, (1,0,0)), circshift(image, (0,1,0)), circshift(image, (0,0,1)); dims=4)
        p_ref = zeros(6, 4)
        p_ref[4, :] .= shifts[1, :]
        p_ref[5, :] .= shifts[2, :]
        p_ref[6, :] .= shifts[3, :]
        @test realign!(imgs; ref_mode=ref_mode, realign=false) ≈ p_ref rtol = 1e-1
    end
end

@testset "Single-axis motion: p=$p (ref_mode=$ref_mode)" for ref_mode ∈ (1, :consensus), p ∈ (
        [0.1, 0, 0, 0, 0, 0],
        [0, 0.1, 0, 0, 0, 0],
        [0, 0, 0.1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1])

    img_moved = make_moved_image(img_itp, p, shape)
    img_series = cat(image, img_moved; dims=4)

    # real-valued
    img_series_r = copy(img_series)
    p_est = realign!(img_series_r; ref_mode=ref_mode, realign=true)
    if ref_mode == 1
        @test p_est[:, 2] ≈ p atol = 1e-1
    end
    p_residual = realign!(img_series_r; ref_mode=ref_mode, realign=false)
    @test p_residual ≈ zeros(6, 2) atol = 5e-2
    @test img_series_r[:,:,:,1] ≈ img_series_r[:,:,:,2] rtol = 1e-1

    # complex-valued
    img_series_phase = exp.(1im .* axes(img_series, 1) .* π/size(img_series, 1))
    img_series_c = img_series .* img_series_phase
    p_est_c = realign!(img_series_c; ref_mode=ref_mode, realign=true)
    if ref_mode == 1
        @test p_est_c[:, 2] ≈ p atol = 1e-1
    end
    p_residual_c = realign!(img_series_c; ref_mode=ref_mode, realign=false)
    @test p_residual_c ≈ zeros(6, 2) atol = 5e-2
    img_series_c ./= img_series_phase
    @test img_series_c[:,:,:,1] ≈ img_series_c[:,:,:,2] rtol = 1e-1
end

@testset "Combined rotation + translation" begin
    p = [0.08, -0.05, 0.06, 1.5, -1.0, 0.8]
    img_moved = make_moved_image(img_itp, p, shape)
    img_series = cat(image, img_moved; dims=4)

    img_series_r = copy(img_series)
    p_est = realign!(img_series_r; ref_mode=1, realign=true)
    @test p_est[:, 2] ≈ p atol = 1e-1

    p_residual = realign!(img_series_r; ref_mode=1, realign=false)
    @test p_residual[:, 2] ≈ zeros(6) atol = 5e-2
    @test img_series_r[:,:,:,1] ≈ img_series_r[:,:,:,2] rtol = 1e-1
end

@testset "ref_mode=:mean" begin
    p = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    img_moved = make_moved_image(img_itp, p, shape)
    img_series = cat(image, img_moved; dims=4)

    img_series_r = copy(img_series)
    p_est = realign!(img_series_r; ref_mode=:mean, realign=true)
    @test size(p_est) == (6, 2)

    p_residual = realign!(img_series_r; ref_mode=:mean, realign=false)
    @test p_residual ≈ zeros(6, 2) atol = 1e-1
end

@testset "Providing known motion_params" begin
    p = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    img_moved = make_moved_image(img_itp, p, shape)
    img_series = cat(image, img_moved; dims=4)

    motion_params = zeros(6, 2)
    motion_params[:, 2] .= p

    img_series_r = copy(img_series)
    realign!(img_series_r, motion_params; center=size(image) .÷ 2)

    p_residual = realign!(img_series_r; ref_mode=1, realign=false)
    @test p_residual ≈ zeros(6, 2) atol = 5e-2
end

end # top-level testset
