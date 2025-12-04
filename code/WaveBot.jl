module WaveBot

using CSV
using DataFrames
using Unitful
using Unitful: Hz
using DimensionfulAngles.DefaultSymbols
using DimensionfulAngles: θ₀, Periodic
using PhysicalConstants.CODATA2018: g_n as g

const Ndof = 1

function preprocessing(bemdir="."; freq_wave=nothing, scaling_factor=1)
    # directory
    ω = Matrix(CSV.read(bemdir * "/omega.csv", DataFrame, header=false))[:,1] * rad/s

    # frequency
    Nfreq = length(ω)
    ωₙ = ω / θ₀
    ωₙₘ = reshape(ωₙ, (1,1,Nfreq))
    !isnothing(freq_wave) && check_frequencies(bemdir, freq_wave)

    # radiation
    Bᵣ = reshape(transpose(Matrix(CSV.read(bemdir * "/wavebot_radiationdamping.csv", DataFrame, header=false))), (Ndof, Ndof, Nfreq)) * kg/s
    Mₐ = reshape(transpose(Matrix(CSV.read(bemdir * "/wavebot_addedmass.csv", DataFrame, header=false))), (Ndof, Ndof, Nfreq)) * kg
    Zᵣ =  Bᵣ + im * (ωₙₘ .* Mₐ)

    # inertia and hydrostatics
    ρ = 1025kg/m^3
    H, h, R, r = 0.16m, 0.53m-0.16m, 0.88m, 0.35m
    volume = (1/3 * π * h * (R^2 + R*r + r^2)) + (H * π * R^2)
    M = volume * ρ * scaling_factor
    K = π*R^2 * ρ * g

    Zₘ = im * ωₙₘ * M
    Zₖ = -im ./ ωₙₘ * K

    # intrinsic mechanical impedance
    Zᵢ = Zᵣ + Zₘ + Zₖ

    # PTO impedance
    N = kg*m*s^-2
    H = *(kg*m^2*s^-2*A^-2)
    Ω = *(kg*m^2*s^-3*A^-2)
    PTO_gear_ratio = 12.0*rad/m                             # N
    PTO_torque_constant = sqrt(3/2)*6.7*N*m/A               # k_r
    PTO_winding_resistance = 0.5*Ω                          # R_w
    PTO_winding_inductance = 0.0*H                          # L_w
    PTO_drivetrain_inertia = 2.0*kg*m^2/rad#24*kg*m^2/rad   # J_d
    PTO_drivetrain_friction = 1.0*N*m*s/rad                 # b_d
    PTO_drivetrain_stiffness = 0.0*N*m/rad#-10*N*m/rad#     # k_d

    PTO_drivetrain_impedance =                  # Z_d
        PTO_drivetrain_friction  .+             # b_d
        im*ωₙ*PTO_drivetrain_inertia .+         # j*ω*J_d
        (-im ./ ωₙ) * PTO_drivetrain_stiffness  # j*k_d/ω

    PTO_winding_impedance =                     # Z_w
        PTO_winding_resistance .+               # R_w
        im*ωₙ*PTO_winding_inductance            # j*ω*L_w

    off_diag = PTO_torque_constant * PTO_gear_ratio / θ₀
    pto_impedance_11 = PTO_drivetrain_impedance * PTO_gear_ratio^2 / θ₀
    pto_impedance_12 = -complex(off_diag) * ones(size(ω))
    pto_impedance_21 = complex(off_diag) * ones(size(ω))
    pto_impedance_22 = PTO_winding_impedance

    Zₚₜₒ = [
        reshape(pto_impedance_11, (Ndof,Ndof,Nfreq)) reshape(pto_impedance_12, (Ndof,Ndof,Nfreq))
        reshape(pto_impedance_21, (Ndof,Ndof,Nfreq)) reshape(pto_impedance_22, (Ndof,Ndof,Nfreq))
        ]

    # Thèvenin equivalent circuit
    Zₜₕ = @. Zₚₜₒ[2:2,2:2,:] - (Zₚₜₒ[2:2,1:1,:] * Zₚₜₒ[1:1,2:2,:]) / (Zᵢ + Zₚₜₒ[1:1,1:1,:])

    return ω, Zᵢ, Zₚₜₒ, Zₜₕ, Nfreq, M, N
end

# Thèvenin equivalent
function thevenin_equivalent(bemdir="."; freq_wave=nothing, scaling_factor=1)
    ω, Zᵢ, Zₚₜₒ, Zₜₕ, Nfreq, M, N = preprocessing(bemdir; freq_wave, scaling_factor)
    # Thèvenin equivalent circuit
    Vₜₕ_factor =  Zₚₜₒ[2:2,1:1,:] ./ (Zᵢ .+ Zₚₜₒ[1:1,1:1,:])
    # excitation coefficients
    diffraction = reshape(transpose(
            Matrix(CSV.read(bemdir * "/wavebot_diffraction_real.csv", DataFrame, header=false)) .+
            im * Matrix(CSV.read(bemdir * "/wavebot_diffraction_imag.csv", DataFrame, header=false))
    ), (Ndof, Ndof, Nfreq)) * N/m

    froude_krylov = reshape(transpose(
            Matrix(CSV.read(bemdir * "/wavebot_FK_real.csv", DataFrame, header=false)) .+
            im * Matrix(CSV.read(bemdir * "/wavebot_FK_imag.csv", DataFrame, header=false))
    ), (Ndof, Ndof, Nfreq)) * N/m

    Fₑ_coeff = diffraction + froude_krylov

    return (Fₑ_coeff, Vₜₕ_factor, Zₜₕ)
end

function power(thevenin, waves)
    Fₑ_coeff, Vₜₕ_factor, Zₜₕ = thevenin
    Ndof, _, Nfreq = size(Zₜₕ)
    Fₑ = reshape(waves, (Ndof,Ndof,Nfreq)) .* Fₑ_coeff
    Vₜₕ = Fₑ .* Vₜₕ_factor
    return abs.(Vₜₕ).^2 ./ (8*real(Zₜₕ))
end

power(thevenin) = waves->power(thevenin, waves)

function average_power(thevenin, waves)
    return sum(power(thevenin, waves), dims=3)[1,1,1]
end

average_power(thevenin) = (waves->average_power(thevenin, waves))

function check_frequencies(bemdir, freq_wave)
    ω_bem = Matrix(CSV.read(bemdir * "/omega.csv", DataFrame, header=false))[:,1] * rad/s
    freq_bem = uconvert.(Hz, ω_bem, Periodic())
    if !(ustrip.(freq_bem) ≈ ustrip.(freq_wave))
        error("frequency arrays are not equal")
    end
    return nothing
end

end
