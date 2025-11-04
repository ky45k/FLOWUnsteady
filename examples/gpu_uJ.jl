# gpu_uJ.jl

using CUDA, LinearAlgebra    # GPU 커널과 벡터 연산용

# 1) CUDA 커널 정의: Biot–Savart N² 연산
function biot_savart_kernel!(X, Γ, U, N)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= N
        xi = @view X[1:3, i]
        vi = zero(eltype(xi)).*(1:3)

        @inbounds for j in 1:N
            if j != i
                xj = @view X[1:3, j]
                γj = @view Γ[1:3, j]
                r = xi .- xj
                r3 = norm(r)^3 + 1e-12
                vi .+= cross(γj, r) ./ r3
            end
        end

        @inbounds U[1,i], U[2,i], U[3,i] = vi
    end
    return
end

# 2) GPU용 UJ 함수 정의
function UJ_gpu!(pfield)
    N = get_np(pfield)
    X_cpu = hcat([p.X for p in pfield.particles]...)
    Γ_cpu = hcat([p.Gamma for p in pfield.particles]...)

    Xg = CuArray(X_cpu)
    Γg = CuArray(Γ_cpu)
    Ug = similar(Xg)

    threads, blocks = 256, cld(N,256)
    @cuda threads=threads blocks=blocks biot_savart_kernel!(Xg, Γg, Ug, N)

    U_cpu = Array(Ug)
    @inbounds for i in 1:N
        pfield.particles[i].U .= U_cpu[:,i]
    end

    return nothing
end
