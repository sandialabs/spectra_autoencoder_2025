
module Waves

import Base: nextfloat
using Unitful: m, Hz, s, ustrip
using Unitful: AbstractQuantity, Quantity, unit  # TODO: Delete
using Unitful: upreferred, uconvert, m, Hz, unit, Length, Time, Frequency, ustrip
using StatsBase
# using Plots
using DataFrames
using Dates
using HTTP
using TranscodingStreams, CodecZlib
using DelimitedFiles
using DimensionfulAngles: rad, °ᵃ as °
using AxisArrays
using PhysicalConstants.CODATA2018: g_n as g
using Integrals: SampledIntegralProblem, TrapezoidalRule, IntegralProblem, QuadGKJL, solve
using SciMLBase: AbstractIntegralAlgorithm, ReturnCode
using Distributions: Chisq, quantile
using Interpolations: interpolate, extrapolate, scale, Gridded, Linear
# using DimensionfulAngles: rad
# using MAT: matread
# using NaNMath: sqrt as sqrt_nan
using JLD2

# spectral statistics
function integrate(y::AbstractVector, x::AbstractVector, method::AbstractIntegralAlgorithm=TrapezoidalRule())
    sol = solve(SampledIntegralProblem(y, x), method)
    sol.retcode ≠ ReturnCode.Success && error("solution unsuccessful with code: $(sol.retcode)")
    return sol.u
end

function integrate(y::AbstractMatrix, x::AbstractVector, method::AbstractIntegralAlgorithm=TrapezoidalRule())
    sol = solve(SampledIntegralProblem(y, x; dim=2), method)
    sol.retcode ≠ ReturnCode.Success && error("solution unsuccessful with code: $(sol.retcode)")
    return sol.u
end

function integrate(y::AbstractVector, x::AbstractVector, a::Number, b::Number, method::AbstractIntegralAlgorithm=QuadGKJL(); abstol=nothing)
    f = extrapolate(interpolate((x,), y, Gridded(Linear())), 0.0)
    isnothing(abstol) && (abstol=1.0e-8*unit(eltype(x))*unit(eltype(y)))
    sol = solve(IntegralProblem((x, p)->f(x), a, b), method; abstol)
    sol.retcode ≠ ReturnCode.Success && error("solution unsuccessful with code: $(sol.retcode)")
    return sol.u
end

function integrate(y::AbstractMatrix, x::AbstractVector, a::Number, b::Number, method::AbstractIntegralAlgorithm=QuadGKJL(); abstol=nothing)
    isnothing(abstol) && (abstol=1.0e-8*unit(eltype(x))*unit(eltype(y)))
    sol = ones(eltype(x[1]*y[1,1]), size(y)[1])
    for (i, yᵢ) in enumerate(eachrow(y))
        fᵢ = extrapolate(interpolate((x,), yᵢ, Gridded(Linear())), 0.0)
        solᵢ = solve(IntegralProblem((x, p)->fᵢ(x), a, b), method; abstol)
        solᵢ.retcode ≠ ReturnCode.Success && error("solution $i unsuccessful with code: $(sol.retcode)")
        sol[i] = solᵢ.u
    end
    return sol
end

function integrate(y::AbstractMatrix, x::AbstractVector, a::Union{Number, AbstractVector}, b::Union{Number, AbstractVector}, method::AbstractIntegralAlgorithm=QuadGKJL(); abstol=nothing)
    isnothing(abstol) && (abstol=1.0e-8*unit(eltype(x))*unit(eltype(y)))
    sol = ones(eltype(x[1]*y[1,1]), size(y)[1])
    a = (length(a)≠1) ? a : ones(size(y)[1])*a
    b = (length(b)≠1) ? b : ones(size(y)[1])*b
    for (i, yᵢ) in enumerate(eachrow(y))
        fᵢ = extrapolate(interpolate((x,), yᵢ, Gridded(Linear())), 0.0)
        solᵢ = solve(IntegralProblem((x, p)->fᵢ(x), a[i], b[i]), method; abstol)
        solᵢ.retcode ≠ ReturnCode.Success && error("solution $i unsuccessful with code: $(sol.retcode)")
        sol[i] = solᵢ.u
    end
    return sol
end


∫ = integrate

function spectral_moment(S::AbstractVector, f::AbstractVector, n::Integer; method::AbstractIntegralAlgorithm=TrapezoidalRule())
    return ∫((S.*(f.^n)), f, method)
end

function spectral_moment(S::AbstractMatrix, f::AbstractVector, n::Integer; method::AbstractIntegralAlgorithm=TrapezoidalRule())
    return ∫((S.*(f.^n)'), f, method)
end

function spectral_moment(S::AbstractVector, f::AbstractVector, n::Integer, a::Number, b::Number; method::AbstractIntegralAlgorithm=QuadGKJL())
    return ∫((S.*(f.^n)), f, a, b, method)
end

function spectral_moment(S::AbstractMatrix, f::AbstractVector, n::Integer, a::Union{Number, AbstractVector}, b::Union{Number, AbstractVector}; method::AbstractIntegralAlgorithm=QuadGKJL())
    return ∫((S.*(f.^n)'), f, a, b, method)
end

function energy_period(S::AbstractVecOrMat, f::AbstractVector; method::AbstractIntegralAlgorithm=TrapezoidalRule())
    m_n1 = spectral_moment(S, f, -1; method)
    m_0 = spectral_moment(S, f, 0; method)
    return upreferred.(m_n1./m_0)
end

function significant_waveheight(S::AbstractVecOrMat, f::AbstractVector; method::AbstractIntegralAlgorithm=TrapezoidalRule())
	m_0 = spectral_moment(S, f, 0; method)
    return 4(.√m_0)
end

function steepness(S::AbstractVecOrMat, f::AbstractVector; method::AbstractIntegralAlgorithm=TrapezoidalRule())
    Hs = significant_waveheight(S, f; method)
    Te = energy_period(S, f; method)
    return 2π*Hs ./ (g*Te.^2)
end

function spectral_statistics(S::AbstractVecOrMat, f::AbstractVector; method::AbstractIntegralAlgorithm=TrapezoidalRule())
    m_n1 = spectral_moment(S, f, -1; method)
    m_0 = spectral_moment(S, f, 0; method)
    Te =  upreferred.(m_n1./m_0)
    Hs = 4(.√m_0)
    Se = 2π*Hs ./ (g*Te.^2)
    return Te, Hs, Se
end

function mackay_parameters(S::AbstractVecOrMat, f::AbstractVector; method_discrete::AbstractIntegralAlgorithm=TrapezoidalRule(), method_continuous::AbstractIntegralAlgorithm=QuadGKJL())
    fe = 1.0 ./ energy_period(S, f; method=method_discrete)
    m_0 = spectral_moment(S, f, 0; method=method_discrete)
    m_n1 = spectral_moment(S, f, -1; method=method_discrete)
    Te =  upreferred.(m_n1./m_0)
    Hs = 4(.√m_0)

    a, b = 0.0, Inf # 0.0Hz, Inf*Hz
    m_0_a = spectral_moment(S, f, 0, a, fe; method=method_continuous)
    m_n1_a = spectral_moment(S, f, -1, a, fe; method=method_continuous)
    m_0_b = spectral_moment(S, f, 0, fe, b; method=method_continuous)
    m_n1_b = spectral_moment(S, f, -1, fe, b; method=method_continuous)
    H̅sₐ² = m_0_a ./ m_0
    Te_a =  upreferred.(m_n1_a./m_0_a)
    Te_b =  upreferred.(m_n1_b./m_0_b)
    dT̅ = (Te_a - Te_b) ./ Te
    return (Te, Hs, H̅sₐ², dT̅)
end

# normalization
function normalize(S::AbstractVecOrMat, f::AbstractVector; method::AbstractIntegralAlgorithm=TrapezoidalRule())
    te = energy_period(S, f; method)
    hs = significant_waveheight(S, f; method)
    return upreferred.(f.*te), upreferred.(S./(hs.^2 .* te))
end

# spectra
te_to_tp(Te::Number, γ::Number) = Te / (0.8255 + 0.03852*γ - 0.005537*γ^2 + 0.0003154*γ^3)
te_to_tp(Te::Number) = Te / 0.858

function pm_spectrum(Hs::Length, Tp::Time, f::Frequency)
    fp = 1/Tp
    spectrum = f==0 ? 0 : (Hs^2 /4) * (1.057* fp)^4 * (f)^(-5) * exp((-5/4)*(fp/f)^4)
    return uconvert(m^2/Hz, spectrum)
end

pm_spectrum(Hs, Tp) = f->pm_spectrum(Hs, Tp, f)

function jonswap_spectrum(Hs, Tp, f; γ=nothing)
    fp = 1/Tp
    σ = f <= fp ? 0.07 : 0.09
    α = exp(-((f/fp - 1)/(√(2)*σ))^2)
    isnothing(γ) && (
        γ = (Tp/√Hs ≤ 3.6) ? 5 : (
            (Tp/√Hs > 5) ? 1 : (
            exp(5.75-1.15(Tp/√Hs))
    )))
    Cws = 1 - 0.287*log(γ)
    spectrum = f==0 ? 0 : Cws * pm_spectrum(Hs, Tp, f) *(γ ^α)
    return spectrum
end

jonswap_spectrum(Hs, Tp; γ=nothing) = f->jonswap_spectrum(Hs, Tp, f; γ)

function mackay_interp(dTn, Han2; data) #datafile="unified_model_parameters.mat")
    # data = matread(datafile)

    # Interpolation object (caches coefficients and such)
    function interpolate_2d(z; x=vec(data["dtn"]), y=vec(data["han2"]), xi=dTn, yi=Han2)
        local f = extrapolate(interpolate((x,y), z, Gridded(Linear())), 0.0)
        zi = f.(xi, yi)
    end

    hs1_f = interpolate_2d(data["Hs1"]')
    hs2_f = interpolate_2d(data["Hs2"]')
    te1_f = interpolate_2d(data["Te1"]')
    te2_f = interpolate_2d(data["Te2"]')
    γ1 = interpolate_2d(data["gam1"]')
    γ2 = interpolate_2d(data["gam2"]')



    return (hs1_f, hs2_f, te1_f, te2_f, γ1, γ2)
end

function mackay_spectrum(Hs, Te, dTn, Han2, f; mat_data) # datafile="unified_model_parameters.mat")
        # Computes the Unified Model spectrum described in the paper
        # "A Unified Model for Unimodal and Bimodal Wave Spectra" by Ed Mackay,
        # published at the European Wave and Tidal Energy Conference (EWTEC), September 2015, Nantes, France.
        #
        # Parameters:
        #     Han2 - normalized swell height squared (see definition in paper)
        #     dTn - bimodality parameter (see definition in paper)

        (hs1_f, hs2_f, te1_f, te2_f, γ1, γ2) = mackay_interp(dTn, Han2; data=mat_data) # datafile)

        hs1 = Hs * hs1_f/4.0
        hs2 = Hs * hs2_f/4.0
        te1 = Te * te1_f
        te2 = Te * te2_f
        tp1 = te_to_tp(te1, γ1)
        tp2 = te_to_tp(te2, γ2)

        E1 = jonswap_spectrum.(hs1, tp1, f; γ=γ1)
        E2 = jonswap_spectrum.(hs2, tp2, f; γ=γ2)

        return (E1 + E2)
end

mackay_spectrum(Hs, Te, Han2, dTn; mat_data) = f->mackay_spectrum(Hs, Te, Han2, dTn, f; mat_data)

rand_freq(n) = (rand(n)*2 .- 1)*π*rad


function wave_amplitudes(S::AbstractVector, f::AbstractVector, θ=nothing)
    Δf = diff(f);
    Δf = cat(Δf, Δf[end], dims=1) # bit of a hack. Las frequency should have no energy anyways...
    if !isnothing(θ)
        length(θ)≠length(f) && error("`θ` and `f` must have same length")
    else
        θ = rand_freq(length(f))
    end
    return (.√(2S.*Δf) .* exp.(im*θ))
    # return (sqrt_nan.(ustrip.(2S.*Δf))*m .* exp.(im*θ))
end

function wave_amplitudes(S::AbstractMatrix, f::AbstractVector, θ=nothing)
    Δf = diff(f);
    Δf = cat(Δf, Δf[end], dims=1) # bit of a hack. Las frequency should have no energy anyways...
    if !isnothing(θ)
        length(θ)≠length(f) && error("`θ` and `f` must have same length")
    else
        θ = rand_freq(length(f))
    end
    return (.√(2S.*Δf') .* (exp.(im*θ))')
end

function sampling_error(DOF, quant)
    χ² = Chisq(DOF)
    quant_low, quant_high = 1-(1+quant)/2, 1-(1-quant)/2
    return quantile(χ², quant_low)/DOF, quantile(χ², quant_high)/DOF
end

function _available(parameter::AbstractString)
    # scrape website
    url = "https://www.ndbc.noaa.gov/data/historical/" * parameter * "/"
    raw = filter(x -> occursin(".txt.gz", x), split(String(HTTP.get(url).body)))
    # save_object("./data/cache/"*parameter*".jld2", raw)
    # parse
    filenames = map(x -> String(split(x, "\"")[2]), raw)
    buoys = map(x -> x[1:5], filenames)
    years = map(x -> String(split(x, ".")[1][7:end]), filenames)
    # create DataFrame
    data = DataFrame("buoy" => buoys, "year" => years, "b_file" => false)
    # remove entries with bad file names, currently only "42002w2008_old.txt.gz"
    regular_file(y::String) = length(y) == 4
    b_file(y::String) = length(y) == 5 && y[1] == 'b'
    filter!(:year => y -> regular_file(y) || b_file(y), data)
    # b-files
    b_files = filter(:year => b_file, data)
    for row in eachrow(b_files)
        ibuoy = row.buoy
        iyear = row.year[2:end]
        # TODO: there's probably a more efficient way of doing this:
        data.b_file = @. ifelse(data.buoy == ibuoy && data.year == iyear, true, data.b_file)
    end
    filter!(:year => y -> regular_file(y), data)
    data[!, :year] = parse.(Int, data[!, :year])
    sort!(data)
end


function _available(parameter::AbstractString, buoy::Union{AbstractString,Int})
    data = _available(parameter)
    _filterbuoy(data, buoy)
end


function _request(parameter::AbstractString, buoy::Union{AbstractString,Int}, year::Int, b_file::Bool=false)
    # get data
    sep_dict = Dict("swden" => "w", "swdir" => "d", "swdir2" => "i", "swr1" => "j", "swr2" => "k")
    sep = b_file ? sep_dict[parameter] * "b" : sep_dict[parameter]
    filename = string(buoy) * sep * string(year) * ".txt.gz"
    url = "https://www.ndbc.noaa.gov/data/historical/" * parameter * "/" * filename
    raw = transcode(GzipDecompressor, HTTP.get(url).body)
    _read(raw, parameter)
end


function _filterbuoy(data::DataFrame, buoy::Union{AbstractString,Int})
    filter!(row -> row.buoy == string(buoy), data)
    select!(data, Not(:buoy))
end


function _read(file::Union{AbstractString,Vector{UInt8}}, parameter::AbstractString)
    # parse data
    data, header = DelimitedFiles.readdlm(file, header=true)
    header[1] = strip(header[1], '#')
    # datetime
    ncol_date = header[5] == "mm" ? 5 : 4
    datevec = string.(Int.(data[:, 1:ncol_date]), pad=2)
    two_digit_year = length(datevec[1, 1]) == 2
    fmt = two_digit_year ? "yy" : "yyyy"
    fmt *= "mmddHH"
    ncol_date == 5 && (fmt *= "MM")
    dates = DateTime[]
    for row in eachrow(datevec)
        push!(dates, DateTime(join(row), fmt))
    end
    # data
    unit_dict = Dict("swden" => m * m / Hz, "swdir" => °, "swdir2" => °, "swr1" => 1, "swr2" => 1)
    data = data[:, ncol_date+1:end] * unit_dict[parameter]
    # frequency
    frequency = parse.(Float64, header[ncol_date+1:end]) * Hz
    # AxisArray
    AxisArray(data; time=dates, frequency=frequency)
end


function read(file::AbstractString, parameter::Union{AbstractString,Nothing}=nothing)
    param_dict = Dict("w" => "swden", "d"  => "swdir", "i" => "swdir2", "j" => "swr1", "k" => "swr2")
    isnothing(parameter) && (parameter = param_dict[string(basename(file)[6])])
    file[end-2:end] == ".gz" && (file = transcode(GzipDecompressor, Base.read(file, String)))
    _read(file, parameter)
end


function available_omnidirectional()
    _available("swden")
end


function available_omnidirectional(buoy::Union{AbstractString,Int})
    _available("swden", buoy)
end


function request_omnidirectional(buoy::Union{AbstractString,Int}, year::Int, b_file::Bool=false)
    _request("swden", buoy, year, b_file)
end


function available()
    den = _available("swden")
    dir = _available("swdir")
    dir2 = _available("swdir2")
    r1 = _available("swr2")
    r2 = _available("swr1")

    # buoy-year combinations for which all 5 files exist
    innerjoin(den, dir, dir2, r1, r2, on=[:buoy, :year, :b_file])
end


function available(buoy::Union{AbstractString,Int})
    data = available()
    _filterbuoy(data, buoy)
end


function request(buoy::Union{AbstractString,Int}, year::Int, b_file::Bool=false)
    buoy = string(buoy)
    den = _request("swden", buoy, year, b_file)
    dir = _request("swdir", buoy, year, b_file)
    dir2 = _request("swdir2", buoy, year, b_file)
    r1 = _request("swr1", buoy, year, b_file)
    r2 = _request("swr2", buoy, year, b_file)

    time, frequency = den.axes
    parameter = AxisArrays.Axis{:parameter}([:den, :dir, :dir2, :r1, :r2])
    AxisArray(cat(den, dir, dir2, r1, r2; dims=3), time, frequency, parameter)
end


function metadata(buoy::Union{AbstractString,Int})
    keys = ["Water depth", "Watch circle radius"]
    url = "https://www.ndbc.noaa.gov/station_page.php?station=" * string(buoy)
    raw = split(String(HTTP.get(url, status_exception=false).body), '\n')
    dict = Dict{String, Union{Nothing, Quantity}}()
    for key in keys
        data = filter(x -> occursin(key, x), raw)
        if length(data) == 0
            value =  NaN * 1m
        else
            value = replace(replace(data[1], "\t" => ""), r"<.+?(?=>)>" => "")
            value = strip(split(value, ':')[2])
            if key == "Water depth"
                @assert split(value)[2] == "m"
                value = parse(Float64, split(value)[1]) * 1m
            elseif key == "Watch circle radius"
                @assert split(value)[2] == "yards"
                value = parse(Float64, split(value)[1]) * 0.9144m
            end
        end
        dict[key] = value
    end
    # coordinates
    pattern = r"([+-]?([0-9]*[.])?[0-9]+) +([NS]) +([+-]?([0-9]*[.])?[0-9]+) +([EW])"
    loc = filter(x -> occursin(pattern, x), raw)
    if length(loc) == 0
        lat = NaN
        lon = NaN
    else
        loc = match(pattern, loc[1])
        lat = parse(Float64, loc[1])
        (loc[3] == "S") && (lat *= -1)
        lon = parse(Float64, loc[4])
        (loc[6] == "W") && (lon *= -1)
    end
    dict["Latitude"] = lat * 1°
    dict["Longitude"] = lon * 1°
    return dict
end

function get_data()
    buoy = 46050
    bfile = false

    # available data
    years = (available(buoy)).year
    deleteat!(years, findall(x->(x==2008 || x==2011 || x==2015), years)) # significant amount of data missing in 2008, 2011, 2015
    deleteat!(years, findall(x->(x==2022), years)) # data frequency changes at some point in 2022
    nyears = length(years)

    # frequency
    freq = request_omnidirectional(buoy, years[1], bfile).axes[2].val

    # yearly data
    nsamples = []
    waves = Vector{Any}(nothing, nyears)
    for i = 1:nyears
        # println("$i, $(years[i])")
        # get data
        idata = request_omnidirectional(buoy, years[i], bfile)

        # remove corrupted data
        non_zero_rows = .!vec([x==0 for x in sum(ustrip.(idata); dims=2)])
        steep_thresh = .!vec([x>0.1 for x in steepness(idata, freq)])
        f_energy_thresh = .!vec([any(x.>0.01m^2*Hz^-1) for x in eachrow(idata[:, freq.≤0.0325Hz])])
        filter = non_zero_rows .& steep_thresh .& f_energy_thresh
        idata = idata[filter, :]

        # append
        waves[i] = idata
        append!(nsamples, size(idata)[1])
    end

    return years, freq, waves, nsamples
end

function get_data_hs_te(freq, waves)
    Te, Hs, _ = spectral_statistics(waves, freq)
    data = DataFrame()
    data.te = Te
    data.hs = Hs
    return data
end

function get_data_mackay(freq, waves)
    Te, Hs, H̅sₐ², dT̅ = mackay_parameters(ustrip.(waves), ustrip.(freq))
    data = DataFrame()
    data.te = Te * s
    data.hs = Hs * m
    data.dtn = dT̅
    data.han2 = H̅sₐ²
    return data
end

end
