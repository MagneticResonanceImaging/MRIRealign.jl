module MRIRealign

using Images
using NLSolversBase
using Optim
using StaticArrays
using LinearAlgebra
using Interpolations
using Interpolations: gradient, hessian
using ImageFiltering
using Statistics
using OhMyThreads

export realign!, create_rotation_matrix, create_affine_matrix, params_from_rigid_affine


# --- Top-level functions ---
"""
    realign!(img; center, ref_mode, mask, fwhm, realign, voxel_size, radius, x_abstol) -> motion_params

Estimate rigid-body (6-DOF) motion parameters from a 4-D MRI time series
and, optionally, reslice the volumes to undo the estimated motion.

The algorithm minimizes the sum of squared intensity differences between
each volume and a reference, using a Gauss–Newton trust-region optimizer
with exact analytic Jacobians of the rotation matrix.

# Arguments
- `img::AbstractArray{T,4}`: image array with dimensions `(x, y, z, t)`.
  `T` may be real- or complex-valued. For complex data the motion
  estimation is performed on the magnitude; the phase is resliced along
  with the magnitude when `realign=true`.

# Keyword arguments
- `center=size(img)[1:3] .÷ 2`: rotation center in voxel coordinates.
  Only affects the *parameterization* of the motion (i.e., the returned
  translation values); the aligned images are identical regardless of
  `center`.
- `ref_mode=:consensus`: reference strategy. One of
  - `:consensus` — align every pair of time frames and compute a
    robust weighted consensus (IRLS with geodesic rotation distance).
    Most accurate, but `t`-fold slower than a single reference.
  - `:mean` — align each frame to the temporal mean. Fast, but the
    mean may be blurred when motion is large.
  - an `Integer` — align every frame to the given time-frame index.
    Fast; works well when that frame has good image quality.
- `mask=trues(size(img)[1:3])`: `BitArray` or `Bool` array selecting the
  voxels over which the cost function is evaluated.
- `fwhm=nothing`: optional Gaussian smoothing kernel specified as a
  3-tuple of full-width-at-half-maximum values `(σx, σy, σz)` in voxel
  units. `nothing` (default) applies no smoothing. Smoothing can improve
  robustness for noisy data.
- `realign=true`: if `true`, `img` is overwritten in-place with the
  motion-corrected volumes. If `false`, only the parameters are
  estimated.
- `voxel_size::NTuple{3}=(1.0, 1.0, 1.0)`: voxel dimensions in
  **millimeters**, e.g. `(1.5, 1.5, 3.0)`.  Used to define the
  interpolant on mm-spaced axes so that translations, the affine
  matrix, and all spatial derivatives are natively in mm.
- `radius::Real=64.0`: characteristic head radius in **millimeters**.
  Rotation angles (in radians) are multiplied by `radius` to obtain
  arc-length displacements in mm, placing rotations on the same footing
  as translations inside the optimizer.
- `x_abstol::Real=1e-3`: absolute convergence tolerance for the
  optimizer, in **millimeters**.  The optimizer terminates when the
  parameter step falls below this value.
  The default of `1e-3` mm (1 μm) is far below any practical MRI
  resolution.

# Returns
- `motion_params::Matrix{T}` of size `(6, t)`. Each column holds
  `[rx, ry, rz, tx, ty, tz]` — three rotation angles in **radians** and
  three translations in **millimeters** — for the corresponding time frame.

Note that `voxel_size`, `radius`, and `x_abstol` can in any unit of translation,
as long as all three are the same. The translations in the return `motion_params`
are in the units of the input parameters.


# Examples
```julia
# Estimate and apply motion correction (1 mm isotropic voxels)
params = realign!(img)

# Specify voxel size and head radius
params = realign!(img; voxel_size=(1.5, 1.5, 3.0), radius=70.0)

# Estimate only (no reslicing), using frame 1 as reference
params = realign!(img; ref_mode=1, realign=false)

# With a brain mask, smoothing, and a coarser tolerance (10 μm)
params = realign!(img; mask=brain_mask, fwhm=(5.0, 5.0, 5.0), x_abstol=0.01)
```

See also [`realign!(img, motion_params)`](@ref), [`create_affine_matrix`](@ref).

---

    realign!(img, motion_params; center, voxel_size) -> motion_params

Reslice `img` in-place using pre-computed `motion_params` (e.g., from a
previous call to [`realign!`](@ref)).

# Arguments
- `img::AbstractArray{T,4}`: image array with dimensions `(x, y, z, t)`.
- `motion_params::AbstractMatrix`: `(6, t)` matrix of motion parameters
  in the format `[rx, ry, rz, tx, ty, tz]` per column, with rotations in
  radians and translations in millimeters.

# Keyword arguments
- `center=size(img)[1:3] .÷ 2`: rotation center that was used when
  `motion_params` was estimated.
- `voxel_size::NTuple{3}=(1.0,1.0,1.0)`: voxel dimensions in mm.

# Returns
- `motion_params` (the same matrix that was passed in).

# Examples
```julia
# Two-step workflow: estimate, then apply later
params = realign!(img; realign=false)
# ... inspect params ...
realign!(img, params)
```
"""
function realign!(img::AbstractArray{Tin,4};
    center=size(img)[1:3] .÷ 2,
    ref_mode=:consensus,
    mask=trues(size(img)[1:3]),
    fwhm=nothing::Union{Nothing,NTuple{3}},
    realign=true,
    voxel_size::NTuple{3}=(1.0, 1.0, 1.0),
    radius::Real=64.0,
    x_abstol::Real=1e-3 # stop when displacement step is <1 μm
) where Tin

    if Tin <: Complex
        _img = abs.(img)
        Treal = real(Tin)
    else
        _img = img
        Treal = Tin
    end

    if ref_mode == :consensus
        t_refs = axes(_img, 4)
    elseif ref_mode == :mean
        _img = cat(_img, mean(_img, dims=4); dims=4)
        t_refs = size(_img, 4)
    elseif typeof(ref_mode) <: Integer
        t_refs = ref_mode
    else
        error("ref_mode must either be `:consensus`, `:mean`, or an integer")
    end

    img_s = (fwhm === nothing || all(fwhm .== 0)) ? _img : smooth_image(_img, fwhm)

    voxel_size = Float64.(voxel_size)
    vx, vy, vz = voxel_size
    center = SVector{3,Float64}(center)
    center_mm = center .* voxel_size
    radius = Float64(radius)

    # Interpolate all time frames on mm-spaced axes
    @views Tint = typeof(_interpolate(img_s[:, :, :, 1], voxel_size))
    img_itp = Vector{Tint}(undef, size(img_s, 4))
    @tasks for t ∈ eachindex(img_itp)
        vol = img_s[:, :, :, t]
        vol ./= quantile(vec(vol), 0.9)
        img_itp[t] = _interpolate(vol, voxel_size)
    end

    # Mask positions in mm (with random sub-voxel jitter for convergence, cf. SPM)
    mask_inds = [SVector{3}((idx[1] + rand() - 0.5) * vx,
                            (idx[2] + rand() - 0.5) * vy,
                            (idx[3] + rand() - 0.5) * vz)
                 for idx ∈ CartesianIndices(mask) if mask[idx]]

    # Precompute centered coordinates (mask position minus center in mm)
    xyz_centered = [ind - center_mm for ind ∈ mask_inds]

    _motion_params = Array{Float64}(undef, 6, length(img_itp), length(t_refs))
    for (i_ref, t_ref) ∈ enumerate(t_refs)
        reference = [img_itp[t_ref](ind[1], ind[2], ind[3]) for ind ∈ mask_inds]
        grad_field = [SVector{3}(gradient(img_itp[t_ref], ind[1], ind[2], ind[3])) for ind ∈ mask_inds]
        # hess_field = [hessian(img_itp[t_ref], idx[1], idx[2], idx[3]) for idx ∈ mask_inds]
        hess_field = nothing # using the Gauss-Newton approximation

        p0_mm = zeros(6)
        @tasks for t ∈ axes(img_s, 4)
            @local diff_vals = similar(mask_inds, Float64)

            fgh! = make_fgh_function(reference, img_itp[t], center_mm, mask_inds, xyz_centered,
                                     grad_field, hess_field, diff_vals, radius)
            res = optimize(NLSolversBase.only_fgh!(fgh!), p0_mm, NewtonTrustRegion(),
                           Optim.Options(; x_abstol))
            p_mm = Optim.minimizer(res)
            # Convert arc-lengths back to radians; translations are already in mm
            _motion_params[1:3, t, i_ref] .= p_mm[1:3] ./ radius
            _motion_params[4:6, t, i_ref] .= p_mm[4:6]
        end
    end

    motion_params = if ref_mode == :consensus
        weighted_mean_estimate!(_motion_params; r=mean(size(img)[1:3]))
    elseif ref_mode == :mean
        _motion_params[:, 1:end-1, 1]
    else
        dropdims(_motion_params, dims=3)
    end

    if realign
        realign!(img, motion_params; center, voxel_size)
    end

    return Treal.(motion_params)
end


function realign!(img::AbstractArray{T,4}, motion_params; center=size(img)[1:3] .÷ 2, voxel_size::NTuple{3}=(1.0,1.0,1.0)) where T
    voxel_size = Float64.(voxel_size)
    vx, vy, vz = voxel_size
    center_mm = SVector{3,Float64}(center) .* voxel_size
    @tasks for t ∈ axes(img, 4)
        vol = @view img[:, :, :, t]
        img_itp = _interpolate(vol, voxel_size)
        A = create_affine_matrix(motion_params[:, t], center_mm)
        @inbounds for idx ∈ CartesianIndices(vol)
            v = A * SVector{4,Float64}(idx[1]*vx, idx[2]*vy, idx[3]*vz, 1)
            vol[idx] = img_itp(v[1], v[2], v[3])
        end
    end
    return motion_params
end


# --- Combined f, g, h! for Optim.only_fg! or Newton trust-region ---
function make_fgh_function(reference, mov_itp, center, mask_inds, xyz_centered, grad_field, hess_field, diff_vals, radius)
    function fgh!(F, G, H, p_mm)
        # p_mm = [arc_x, arc_y, arc_z, tx, ty, tz] — all in mm.
        # Convert arc-lengths to radians for the affine matrix.
        p = SVector{6}(p_mm[1]/radius, p_mm[2]/radius, p_mm[3]/radius, p_mm[4], p_mm[5], p_mm[6])

        # Residuals — all coordinates are in mm
        A = create_affine_matrix(p, center)
        @inbounds for (n, ind) ∈ enumerate(mask_inds)
            v = A * SVector{4}(ind[1], ind[2], ind[3], 1.0)
            diff_vals[n] = reference[n] - mov_itp(v[1], v[2], v[3])
        end

        # Objective value
        F = sum(abs2, diff_vals)

        # Initialize gradient and Hessian
        G === nothing || fill!(G, 0)
        H === nothing || fill!(H, 0)

        # Precompute rotation matrix derivatives for exact Jacobian
        rx, ry, rz = p[1], p[2], p[3]
        sr, cr = sincos(rx)
        sp, cp = sincos(ry)
        sy, cy = sincos(rz)

        # ∂R/∂rx
        dRdrx = @SMatrix [
             0                     cy*sp*cr+sy*sr   -cy*sp*sr+sy*cr;
             0                     sy*sp*cr-cy*sr   -sy*sp*sr-cy*cr;
             0                     cp*cr            -cp*sr
        ]
        # ∂R/∂ry
        dRdry = @SMatrix [
            -cy*sp   cy*cp*sr   cy*cp*cr;
            -sy*sp   sy*cp*sr   sy*cp*cr;
            -cp     -sp*sr     -sp*cr
        ]
        # ∂R/∂rz
        dRdrz = @SMatrix [
            -sy*cp  -sy*sp*sr-cy*cr  -sy*sp*cr+cy*sr;
             cy*cp   cy*sp*sr-sy*cr   cy*sp*cr+sy*sr;
             0       0                0
        ]

        # Per-voxel contributions — direct dot-product formulation avoids
        # constructing the 3×6 Jacobian matrix Jx and the matrix-vector
        # product Jx' * gI.  Instead we compute JtgI (the 6-vector) directly.
        #
        # The optimizer variable is p_mm = (arc_x, arc_y, arc_z, tx, ty, tz)
        # where arc_i = angle_i * radius.  By the chain rule:
        #   ∂f/∂arc_i = (∂f/∂angle_i) * (∂angle_i/∂arc_i) = (∂f/∂angle_i) / radius
        # For translations:  ∂f/∂t_mm = gI  (identity, interpolant is in mm)
        @inbounds for i ∈ eachindex(mask_inds)
            xyz = xyz_centered[i]    # precomputed: mask_inds[i] - center (mm)
            gI  = grad_field[i]      # ∂I/∂x_mm
            r   = diff_vals[i]

            # Rotation part: dot(dR * xyz, gI) for each of rx, ry, rz
            drx_xyz = dRdrx * xyz
            dry_xyz = dRdry * xyz
            drz_xyz = dRdrz * xyz

            # Jacobian w.r.t. p_mm.  Rotation components are divided by
            # radius (chain rule: ∂/∂arc = ∂/∂angle / radius).
            JtgI = SVector{6}(
                dot(drx_xyz, gI) / radius,
                dot(dry_xyz, gI) / radius,
                dot(drz_xyz, gI) / radius,
                gI[1], gI[2], gI[3]
            )

            if G !== nothing
                G .+= (-2r) .* JtgI
            end

            if H !== nothing
                H .+= 2 .* (JtgI * JtgI')   # Gauss-Newton approx (rank-1 update)
                if hess_field !== nothing
                    # Full Hessian correction requires the actual Jacobian matrix
                    Jx = hcat(drx_xyz, dry_xyz, drz_xyz,
                              SMatrix{3,3,Float64}(1,0,0, 0,1,0, 0,0,1))
                    HI = hess_field[i]       # ∂²I/∂x², shape (3×3)
                    H .-= (2r) .* (Jx' * HI * Jx)
                end
            end
        end
        return F
    end

    return fgh!
end


## --- utilities ---
function smooth_image(img::AbstractArray{T,3}, fwhm::NTuple{3}) where {T<:Real}
    σ = T.(fwhm ./ (2√(2log(2))))
    img = imfilter(img, Kernel.gaussian(σ))
    return img
end

function smooth_image(img::AbstractArray{T,4}, fwhm::NTuple{3}) where {T<:Real}
    img_s = similar(img)
    @tasks for it ∈ axes(img, 4)
        @views img_s[:, :, :, it] .= smooth_image(img[:, :, :, it], fwhm)
    end
    return img_s
end

_interpolate(x) = extrapolate(interpolate(x, BSpline(Cubic())), Interpolations.Flat())

function _interpolate(x, voxel_size::NTuple{3})
    itp = interpolate(x, BSpline(Cubic()))
    nx, ny, nz = size(x)
    vx, vy, vz = voxel_size
    ax = range(vx, step=vx, length=nx)
    ay = range(vy, step=vy, length=ny)
    az = range(vz, step=vz, length=nz)
    return extrapolate(Interpolations.scale(itp, ax, ay, az), Interpolations.Flat())
end


"""
    create_rotation_matrix(rx, ry, rz) -> SMatrix{3,3}
    create_rotation_matrix(p)          -> SMatrix{3,3}

Build a 3×3 rotation matrix from three Euler angles `rx`, `ry`, `rz`
(in radians), using the ZYX intrinsic convention: `R = Rz * Ry * Rx`.

The single-argument form extracts `(p[1], p[2], p[3])`, so any indexable
collection (vector, tuple, `SVector`, …) works.

# Examples
```jldoctest
julia> R = create_rotation_matrix(0.0, 0.0, 0.0)
3×3 StaticArraysCore.SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```

See also [`create_affine_matrix`](@ref).
"""
create_rotation_matrix(p) = create_rotation_matrix(p[1], p[2], p[3])

function create_rotation_matrix(rx, ry, rz)
    Rx = @SMatrix [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)]
    Ry = @SMatrix [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)]
    Rz = @SMatrix [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1]
    R = Rz * Ry * Rx
    return R
end

"""
    create_affine_matrix(p, center) -> SMatrix{4,4}

Build a 4×4 homogeneous rigid-body transformation matrix from the
6-element parameter vector `p = [rx, ry, rz, tx, ty, tz]`.

- `rx, ry, rz`: rotation angles in **radians** (ZYX convention).
- `tx, ty, tz`: translations in the same spatial units as `center`
  (millimeters when used with scaled interpolants).
- `center`: 3-element rotation center in the same spatial units as
  the translations.

The transformation is  `x′ = R * (x - center) + center + t`, so that
rotations are applied about `center` and translations are added
afterward.  The matrix operates in whatever coordinate system `center`
and the translations share.

# Examples
```jldoctest
julia> A = create_affine_matrix([0, 0, 0, 1, 2, 3], (0, 0, 0))
4×4 StaticArraysCore.SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
 1.0  0.0  0.0  1.0
 0.0  1.0  0.0  2.0
 0.0  0.0  1.0  3.0
 0.0  0.0  0.0  1.0
```

See also [`params_from_rigid_affine`](@ref), [`create_rotation_matrix`](@ref).
"""
function create_affine_matrix(p, center)
    rx, ry, rz, tx, ty, tz = p
    R = create_rotation_matrix(rx, ry, rz)
    c = SVector{3}(center[1], center[2], center[3])
    t = SVector{3}(tx, ty, tz) + c - R * c  # = translation + center - R * center
    @SMatrix [
        R[1,1] R[1,2] R[1,3] t[1];
        R[2,1] R[2,2] R[2,3] t[2];
        R[3,1] R[3,2] R[3,3] t[3];
        0      0      0      1
    ]
end

"""
    params_from_rigid_affine(A, center) -> SVector{6}

Extract the 6-DOF parameter vector `[rx, ry, rz, tx, ty, tz]` from a
4×4 homogeneous rigid-body matrix `A` and the rotation `center` that was
used to construct it.  Translations are returned in the same spatial
units as `center`.

This is the inverse of [`create_affine_matrix`](@ref):

```jldoctest
julia> p = [0.1, -0.05, 0.2, 3.0, -2.0, 1.0];

julia> A = create_affine_matrix(p, (32, 32, 32));

julia> collect(params_from_rigid_affine(A, (32, 32, 32))) ≈ p
true
```

See also [`create_affine_matrix`](@ref).
"""
function params_from_rigid_affine(A, center)
    T = eltype(A)
    R = @SMatrix [A[1, 1] A[1, 2] A[1, 3];
        A[2, 1] A[2, 2] A[2, 3];
        A[3, 1] A[3, 2] A[3, 3]]
    t_total = @SVector [A[1, 4], A[2, 4], A[3, 4]]

    # Extract Euler angles for Rz * Ry * Rx
    ry = asin(-R[3, 1])
    if abs(R[3, 1]) < 1 - T(1e-8)
        rx = atan(R[3, 2], R[3, 3])
        rz = atan(R[2, 1], R[1, 1])
    else # Gimbal lock case
        rx = zero(T)
        if R[3, 1] < 0
            rz = atan(-R[1, 2], -R[1, 3])
        else
            rz = atan(R[1, 2], R[1, 3])
        end
    end

    # Compute explicit translation parameters
    t = t_total - ((I(3) - R) * SVector(center))

    return @SVector [rx, ry, rz, t[1], t[2], t[3]]
end

function weighted_mean_estimate!(estimates; r=1, max_iter=100, tol=1e-6)
    # bring all estimates in the same frame of reference (iframe = end÷2)
    for iref ∈ axes(estimates, 3)
        @views A0 = create_affine_matrix(estimates[:, end÷2, iref], (0, 0, 0))
        for iframe ∈ axes(estimates, 2)
            @views A = create_affine_matrix(estimates[:, iframe, iref], (0, 0, 0))
            estimates[:, iframe, iref] = params_from_rigid_affine(A / A0, (0, 0, 0))
        end
    end

    nframes = size(estimates, 2)
    nrefs   = size(estimates, 3)
    consensus = Matrix{Float64}(undef, 6, nframes)

    for iframe ∈ 1:nframes
        # Convert all reference estimates for this frame to quaternions + translations
        quats = Vector{SVector{4,Float64}}(undef, nrefs)
        trans = Vector{SVector{3,Float64}}(undef, nrefs)
        for iref ∈ 1:nrefs
            p = @view estimates[:, iframe, iref]
            R = create_rotation_matrix(p[1], p[2], p[3])
            quats[iref] = _rotmat_to_quat(R)
            trans[iref] = SVector{3,Float64}(p[4], p[5], p[6])
        end

        # Initial consensus: unweighted mean
        q_mean, t_mean = _quat_weighted_mean(quats, trans, ones(nrefs))

        for _ = 1:max_iter
            q_old = q_mean
            t_old = t_mean

            # Compute weights based on geodesic rotation distance + translation distance
            w = Vector{Float64}(undef, nrefs)
            for iref ∈ 1:nrefs
                dot_val = clamp(abs(dot(quats[iref], q_mean)), 0.0, 1.0)
                rot_dist = 2 * acos(dot_val) * r  # geodesic distance scaled by radius
                trans_dist = norm(trans[iref] - t_mean)
                w[iref] = 1 / (1 + sqrt(rot_dist^2 + trans_dist^2))
            end

            q_mean, t_mean = _quat_weighted_mean(quats, trans, w)

            if norm(t_mean - t_old) + 2 * acos(clamp(abs(dot(q_mean, q_old)), 0.0, 1.0)) < tol
                break
            end
        end

        # Convert quaternion back to Euler angles + translation
        R_mean = _quat_to_rotmat(q_mean)
        A_mean = SMatrix{4,4,Float64}(
            R_mean[1,1], R_mean[2,1], R_mean[3,1], 0,
            R_mean[1,2], R_mean[2,2], R_mean[3,2], 0,
            R_mean[1,3], R_mean[2,3], R_mean[3,3], 0,
            t_mean[1],   t_mean[2],   t_mean[3],   1)
        consensus[:, iframe] .= params_from_rigid_affine(A_mean, (0, 0, 0))
    end
    return consensus
end

function _quat_weighted_mean(quats, trans, weights)
    # Weighted quaternion mean via dominant eigenvector of ∑ wᵢ qᵢ qᵢᵀ
    M = zeros(MMatrix{4,4,Float64})
    t_mean = zero(MVector{3,Float64})
    w_sum = zero(Float64)
    for i ∈ eachindex(quats)
        q = quats[i]
        M .+= weights[i] .* (q * q')
        t_mean .+= weights[i] .* trans[i]
        w_sum += weights[i]
    end
    t_mean ./= w_sum

    E = eigen(Symmetric(Matrix(M)))
    q_mean = SVector{4,Float64}(E.vectors[:, end])
    q_mean = q_mean[1] < 0 ? -q_mean : q_mean  # consistent sign
    return q_mean, SVector{3,Float64}(t_mean)
end

function _rotmat_to_quat(R)
    # Shepperd's method: convert 3×3 rotation matrix to unit quaternion [w, x, y, z]
    tr = R[1,1] + R[2,2] + R[3,3]
    if tr > 0
        s = 2 * sqrt(tr + 1)
        w = s / 4
        x = (R[3,2] - R[2,3]) / s
        y = (R[1,3] - R[3,1]) / s
        z = (R[2,1] - R[1,2]) / s
    elseif R[1,1] > R[2,2] && R[1,1] > R[3,3]
        s = 2 * sqrt(1 + R[1,1] - R[2,2] - R[3,3])
        w = (R[3,2] - R[2,3]) / s
        x = s / 4
        y = (R[1,2] + R[2,1]) / s
        z = (R[1,3] + R[3,1]) / s
    elseif R[2,2] > R[3,3]
        s = 2 * sqrt(1 + R[2,2] - R[1,1] - R[3,3])
        w = (R[1,3] - R[3,1]) / s
        x = (R[1,2] + R[2,1]) / s
        y = s / 4
        z = (R[2,3] + R[3,2]) / s
    else
        s = 2 * sqrt(1 + R[3,3] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = (R[1,3] + R[3,1]) / s
        y = (R[2,3] + R[3,2]) / s
        z = s / 4
    end
    q = SVector{4,Float64}(w, x, y, z)
    return q / norm(q)
end

function _quat_to_rotmat(q)
    # Convert unit quaternion [w, x, y, z] to 3×3 rotation matrix
    w, x, y, z = q
    @SMatrix [
        1-2(y^2+z^2)  2(x*y-z*w)    2(x*z+y*w);
        2(x*y+z*w)    1-2(x^2+z^2)  2(y*z-x*w);
        2(x*z-y*w)    2(y*z+x*w)    1-2(x^2+y^2)
    ]
end

end