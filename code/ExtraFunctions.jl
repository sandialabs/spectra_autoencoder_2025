module ExtraFunctions

using Unitful
using StatsBase
# using Plots

include("Waves.jl")
include("WaveBot.jl")
include("AutoEncoder.jl")

function bins(data::AbstractVector, nbins::AbstractVector; nfloats::Integer=100_000)
    # bins
    ndims = length(data)
    length(nbins) == ndims || error("`data` and `nbins` must be same length")
    edges = Vector{Any}(nothing, ndims)
    centers = Vector{Any}(nothing, ndims)
    for (i, (idata, inbins)) in enumerate(zip(data, nbins))
        imin, imax = minimum(idata), nextfloat(maximum(idata), nfloats)
        iΔ = (imax - imin) / inbins
        edges[i] = (imin:iΔ:nextfloat(imax, nfloats))
        centers[i] =(imin+iΔ/2:iΔ:nextfloat(imax-iΔ/2, nfloats))
    end
    centers = Iterators.product(centers...)

    # counts and weights
    hist_data = Tuple(d for d in ustrip.(data))
    hist_edges = Tuple(e for e in ustrip.(collect.(edges)))
    hist = fit(Histogram, hist_data, hist_edges, closed=:left)
    ndata = length(data[1])
    counts = hist.weights
    weights = counts / ndata
    if sum(weights) ≉ 1.0
        error("weights do not add up to 1")
    end

    return (centers, edges), (weights, counts)
end

# function plot_bins_2D(data, centers, edges, type=:data; heat=nothing, plot_edges=true, plot_centers=true, title=nothing, xlabel="Tₑ", ylabel="Hₛ")
#     length(data)==length(size(centers))==length(edges) || error("inputs must be 2D")
#     xlim = (minimum(data[1]), maximum(data[1]))
#     ylim = (minimum(data[2]), maximum(data[2]))
#     if type==:data
#         plt = scatter(data[1], data[2]; xlim=xlim, ylim=ylim, label="data", alpha=0.1, title)
#     elseif type==:heatmap
#         plt = heatmap(centers.iterators[1], centers.iterators[2], heat'; c=cgrad(:bilbao, rev=true), xlim=xlim, ylim=ylim, title)
#     else
#         error("type not recognized")
#     end
#     nbins = size(collect(centers))[1]
#     if plot_edges
#         hline!(edges[2], color=:black, label=nothing, linewidth=10/nbins)
#         vline!(edges[1], color=:black, label=nothing, linewidth=10/nbins)
#     end
#     if plot_centers
#         scatter!(vec(collect(centers)); label="centers", color=:red, markersize=20/nbins, xlabel, ylabel)
#     end
#     return plt
# end

function solve_2d(f, model, data, nbins)
    (ndims = length(data))==2 || error("inputs must be 2D")
    (centers, _), (weights, _) = bins(data, [nbins for _ in 1:ndims])
    centers = collect(centers)
    powers = zeros(nbins^ndims)*u"W"
    for i=1:(nbins^ndims)
        w = weights[i]
        if w != 0
            te, hs = centers[i]
            spectrum = Waves.pm_spectrum(hs, Waves.te_to_tp(te))
            waves = Waves.wave_amplitudes(spectrum.(f), f)
            powers[i] = w * WaveBot.average_power(model, waves)
        end
    end
    return powers, sum(powers)
end

function bin_power(frequency, model, data_params, bin_vec)
    mats = []
    bins = []
    for bin in bin_vec
        mat, power = solve_2d(frequency, model, data_params, bin)
        append!(mats, mat)
        append!(bins, power)
    end
    return mats, bins
end

function clean_data(data, mat_data)
    tmp = Waves.mackay_interp.(data.dtn, data.han2; data=mat_data)
    tmp = reinterpret(reshape, Float64, tmp)'
    idx_nan = isnan.(tmp[:,1])
    # let
    #     plt = scatter(data.dtn, data.han2, label="good", alpha=0.1)
    #     scatter!(data[idx_nan, :].dtn, data[idx_nan, :].han2, label="bad", alpha=0.5)
    #     ylabel!("Han^2")
    #     xlabel!("dTen")
    #     display(plt)
    # end
    amt = sum(idx_nan)
    total = size(data, 1)
    # print(amt)
    # print(total)
    println("bad: $(amt)/$(total) = $(amt/total*100) %")
    return data[.!idx_nan, :], idx_nan
end

function solve_4d(f, model, data, nbins, mat_data)
    (ndims = length(data))==4 || error("inputs must be 4D")
    (centers, edges), (weights, _) = bins(data, [nbins for _ in 1:ndims])
    centers = collect(centers)
    powers = zeros(nbins^ndims)*u"W"
    for i=1:(nbins^ndims)
        w = weights[i]
        te, hs, han2, dtn = centers[i]
        tmp = Waves.mackay_interp.(dtn, han2; data=mat_data)
        if (w!=0) && !any(isnan.(tmp))
            spectrum = Waves.mackay_spectrum(hs, te, dtn, han2; mat_data)
            waves = Waves.wave_amplitudes(spectrum.(f), f)
            powers[i] = w * WaveBot.average_power(model, waves)
        end
        # println("te $te, hs $hs, han2 $han2, dtn $dtn: $(powers[i])")
    end
    return powers, sum(powers)
end

function solve_4d_AE(f, model, data, nbins, decoder, orig_freqs, new_freqs)
    (ndims = length(data))==4 || error("inputs must be 4D")
    (centers, edges), (weights, _) = bins(data, [nbins for _ in 1:ndims])
    centers = collect(centers)
    powers = zeros(nbins^ndims)*u"W"
    for i=1:(nbins^ndims)
        w = weights[i]
        te, hs, θ1, θ2 = centers[i]  # TODO
        if (w!=0)
            decoder_input = transpose(reduce(hcat, [θ1, θ2]))
            decoder_output = transpose(decoder(decoder_input))
            spectrum = transpose(AutoEncoder.unnormalize_data(decoder_output, [ustrip(te)], [ustrip(hs)], orig_freqs, new_freqs)[1, begin+2:end-2])
            waves = Waves.wave_amplitudes(spectrum, f)
            powers[i] = (w * ustrip(WaveBot.average_power(model, waves)))*u"W"
        end
        # println("te $te, hs $hs, θ1 $θ1, θ2 $θ2: $(powers[i])")
    end
    return powers, sum(powers)
end

end