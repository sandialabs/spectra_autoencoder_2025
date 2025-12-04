### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 707f2b47-c200-408c-8b23-984ca0c18dc6
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	
	using PlutoUI
	using CairoMakie
	
	using Dates
	using Unitful
	using Statistics
	using StatsBase
	using CSV
	using DimensionfulAngles
	using PhysicalConstants.CODATA2018
	using HTTP
	using TranscodingStreams
	using CodecZlib
	using AxisArrays
	using Integrals
	using SciMLBase
	using DataFrames
	using MAT: matread
	using Distributions
	using LaTeXStrings
	using LinearAlgebra
	
	using Flux
	using JLD2
	using Interpolations
	using DelimitedFiles
	md"""
	#### Imports & Includes
	"""
end

# ╔═╡ 8ca16461-40af-4b1e-a24c-5b01d23e18a3
let
	include("code/AutoEncoder.jl")
	include("code/Waves.jl")
	include("code/WaveBot.jl")
	include("code/ExtraFunctions.jl")
	nothing
end

# ╔═╡ b0e4b31d-965f-4d21-98d0-e411dad86052
md"""
# Site-Specific Parameterization of Ocean Spectra for Power Estimates of Wave Energy Converters
"""

# ╔═╡ 48651dbe-4218-11f0-3f20-393151cdf520
# Pluto Resize & Misc.
html"""<style>
main {
    max-width: 50%;
}
.dont-panic{ display: none }
</style>
"""

# ╔═╡ a957144e-22ad-4f88-943d-17cad7087b66
TableOfContents()

# ╔═╡ ccf45dd8-debc-40cb-b31d-576a52f9cf54
begin
	set_theme!(theme_latexfonts();
			   fontsize=10, figure_padding = 5)
	inToPt = 72.353#72
	pt_per_unit = 1
	line_width = 5.148#3.4#4.79
	text_width = 7.058
	# target_size = (line_width*0.95, line_width*0.75)
	
	ax_kwargs = (spinewidth=0.5, xtickwidth=0.5, ytickwidth=0.5)
	legend_kwargs = (framewidth=0.5)
	nothing
end

# ╔═╡ 6c021cbd-ca5d-46ac-9b9f-ea9479cdb749
begin
	pm_label = "P-M"
	ma_label = "Mackay"
	ae_label = "AE"
	
	pm_color = :red
	ma_color = Cycled(2)
	ae_color = Cycled(5)

	pm_style = (:solid)
	ma_style = (:dashdot, :dense)
	ae_style = (:dash, :dense)

	pm_marker = :diamond
	ma_marker = :xcross
	ae_marker = :star5

	bright_marker = :lightgreen
	nothing
end

# ╔═╡ 6f89c696-fefc-4804-a130-fdea2bbccf86
md"""
## Introduction
"""

# ╔═╡ 3121e9e0-f667-4d77-9f0f-a59f4a3ed2a0
md"""
#### Data

If block doesn't work on first pass, try again. Otherwise make sure that [NDBC Data Website](https://www.ndbc.noaa.gov/data/historical/swden/) is reachable.
"""

# ╔═╡ 56bb2601-450f-4f2e-8293-69c2fba111d1
begin
	years, freq, wave_spectra, nsamples, all_data = 0, 0, 0, 0, 0
	if isfile("./data/cache/all_data.jld2")
		jldopen("./data/cache/all_data.jld2", "r") do f
			global years, freq, wave_spectra, nsamples, all_data
			years = f["years"]
			freq = f["freq"]
			wave_spectra = f["wave_spectra"]
			nsamples = f["nsamples"]
		end
		println("Loaded raw data from file!")
	else
		try
			global years, freq, wave_spectra, nsamples, all_data
			years, freq, wave_spectra, nsamples = Waves.get_data()
			jldsave("./data/cache/all_data.jld2"; years, freq, wave_spectra, nsamples)
		catch e
			if isa(e, HTTP.ConnectError)
				error("Please try running the cell block again")
			else
				error(e)
			end
		end
		println("Saved raw data to file!")
	end
	start_idx = findfirst(years.==2009)
	end_idx = findfirst(years.==2021)
	years = years[start_idx:end_idx]
	wave_spectra = wave_spectra[start_idx:end_idx]
	nsamples = nsamples[start_idx:end_idx]
	nothing
end

# ╔═╡ a009dda3-ef87-4192-b049-8b418af94b49
md"""
Using the same filtering method from Mackay in his paper *A unified model for unimodal and bimodal spectra* [DOI 10.1016/j.ijome.2016.04.015](https://doi.org/10.1016/j.ijome.2016.04.015), also described in our paper, we remove spectra of excessive low energy and unrealistic steepness.

Mackay's unified\_model\_parameters.mat can be found [here](https://www.mathworks.com/matlabcentral/fileexchange/55015-unified-model-for-unimodal-and-bimodal-ocean-wave-spectra) and is used in both processing and comparisons.

*NOTE*: Due to a slight discrepancy, the file is currently included.
"""

# ╔═╡ 889a8863-fbe7-414e-ae4c-a6bef388d116
md"""
Multiple segments of this code often took several tens of minutes to run, for convenience, these segments save their results in the ./data/cache folder and load these files when available. These files are *NOT* included. The only exception to this is the trained autoencoder which takes multiple hours to run. The trained model and the saved model after each epoch are included.
"""

# ╔═╡ 92e3352f-94e7-4d68-ab23-6da8dc5e74c4
begin
	# Data Combining and Filtering
	datafile="./data/unified_model_parameters.mat"
	mat_data = matread(datafile)
	
	all_wave_data = cat(wave_spectra..., dims=1)
	all_mackay_data, mackay_data, idx_nan, filtered_spectra = 0, 0, 0, 0
	
	if isfile("./data/cache/filteredData.jld2")
		jldopen("./data/cache/filteredData.jld2", "r") do f
			global all_mackay_data, mackay_data, idx_nan, filtered_spectra
			all_mackay_data = f["all_mackay_data"]
			mackay_data = f["mackay_data"]
			idx_nan = f["idx_nan"]
			filtered_spectra = f["filtered_spectra"]
		end
		println("Loaded preprocessed data from file!")
	else
		global all_mackay_data, mackay_data, idx_nan, filtered_spectra
		all_mackay_data = Waves.get_data_mackay(freq, all_wave_data)
		mackay_data, idx_nan = ExtraFunctions.clean_data(all_mackay_data, mat_data)
		filtered_spectra = all_wave_data[0 .== (idx_nan), :]
		jldsave("./data/cache/filteredData.jld2"; all_mackay_data, mackay_data, idx_nan, filtered_spectra)
		println("Saved preprocessed data to file!")
	end
	
	nyears = length(years)
	hoursinyear(year) = (DateTime(year+1,1,1) - DateTime(year,1,1)).value/1000/60/60
	nsamples_perc = nsamples ./ hoursinyear.(years)
	md"""
	#### Preprocessing data
	"""
end

# ╔═╡ c47fe519-4613-491e-87a1-255bee058027
begin
	freqs = ustrip.(freq)
	# Integrate doesn't correctly use Unitful units, removing Hz that should have dissappeared
	hs_vector = Waves.significant_waveheight(filtered_spectra.*u"Hz", freqs) 
	te_vector = Waves.energy_period(filtered_spectra, freqs).*u"s"

	freqs, hs_vector, te_vector
end

# ╔═╡ 783aeecc-b04b-4903-b7c0-1e20442918d6
function get_bound_axis(val, bins)
	upper_idx = findfirst(val .< bins)
	return bins[upper_idx - 1:upper_idx]
end

# ╔═╡ 24423cf9-7caa-4f28-837e-c7781be61d37
filter(data, h_bound, v_bound) = (data[1] .>= h_bound[1] .&& data[1] .< h_bound[2]) .&& (data[2] .>= v_bound[1] .&& data[2] .< v_bound[2])

# ╔═╡ e9ea5c05-6ed6-4c1e-80f1-27187965540a
roundedUnitful(value; digits=2) = "$(round(ustrip(value); digits)) $(unit(value))"

# ╔═╡ d08e79ef-4c38-437a-8c16-154f3b466215
roundedValue(value; digits=2) = "$(round(value; digits))"

# ╔═╡ cef36d47-771e-40e6-a104-97213fd67ce5
perc_error_full(truth, compare) = ((compare - truth)/truth)*100

# ╔═╡ 15a73043-d988-434f-b07c-ed7c482caa1d
perc_error(truth, compare) = roundedValue(perc_error_full(truth, compare))

# ╔═╡ 427cf560-b282-43b2-bf2b-87b1f443c2c1
abs_error(truth, compare) = ustrip(compare - truth)

# ╔═╡ 5e6ea2a8-2043-4655-8df8-60d89fb3f34c
function plot_bins(data, bin_size; scatter_alpha=1, target_size = (4, 3), xlab=L"Energy Period ($s$)", ylab=L"Significant Waveheight ($m$)")
	(centers, edges), (weights, _) = ExtraFunctions.bins(data, [bin_size for _ in 1:length(data)])
	figsize = target_size .* (inToPt/pt_per_unit)
	f = Figure(
		size = figsize
	)
	ax = CairoMakie.Axis(f[1,1],
		xlabel = xlab,
		ylabel = ylab;
		ax_kwargs...,
		xgridvisible = false, ygridvisible = false,
	)
    xlims!(ax, ustrip.(minimum(data[1])), ustrip(maximum(data[1])))
    ylims!(ax, minimum(data[2]), maximum(data[2]))
	scatter!(ax,
		data[1], data[2],
		alpha=scatter_alpha, label="Data",
		rasterize=3, markersize=5
	)
    nbins = size(collect(centers))[1]
	hlines!(ax, 
		edges[2], color=:gray, 
		label=nothing, linewidth=0.2
	)
	vlines!(ax, 
		edges[1], color=:gray, 
		label=nothing, linewidth=0.2
	)
    scatter!(ax, 
		vec(collect(centers)),
		label="Bin Centers", color=:black, 
		markersize=2
	)
	f, ax
end

# ╔═╡ 8679f1fc-7a29-4fb5-8272-4d75c1daae85
function plot_bins_new(data, bin_size; scatter_alpha=1, target_size = (4, 3), xlab=L"Energy Period ($s$)", ylab=L"Significant Waveheight ($m$)")
	(centers, edges), (weights, _) = ExtraFunctions.bins(data, [bin_size for _ in 1:length(data)])
	figsize = target_size .* (inToPt/pt_per_unit)
	f = Figure(
		size = figsize
	)
	ax = CairoMakie.Axis(f[1,1],
		xlabel = xlab,
		ylabel = ylab;
		ax_kwargs...,
		xgridvisible = false, ygridvisible = false,
	)
    xlims!(ax, ustrip.(minimum(data[1])), ustrip(maximum(data[1])))
    ylims!(ax, minimum(data[2]), maximum(data[2]))
	scatter!(ax,
		data[1], data[2],
		alpha=scatter_alpha, label="Data",
		rasterize=3, markersize=5
	)

	f, ax
end

# ╔═╡ 43e5208c-ca93-41b7-ab83-4fde49f9f46a
# Makie.wong_colors()

# ╔═╡ 6059d13f-783d-4aa4-b13c-8eb189420a7f
md"""
## Methodology
"""

# ╔═╡ ca630b45-a5a1-4565-aa0a-9f928085ea5c
md"""
### Wave Resource Data
"""

# ╔═╡ ace2aa30-e927-4a78-ba39-d6a1002aa751
begin
	# WaveBot
	fullsize_bemdir = "./data/bem_fullsize_waves"
	fs_thevenin = WaveBot.thevenin_equivalent(fullsize_bemdir; freq_wave=freq)
	fs_linearmodel = WaveBot.average_power(fs_thevenin)
end

# ╔═╡ 3df3733c-fc52-467c-a418-99ecc46f5538
begin
	year_sizes = [t[1] for t in size.(wave_spectra)]
	year_idxs = [sum(year_sizes[1:i]) for i in 1:length(year_sizes)]
	filtered_idxs = zeros(Int32, length(year_sizes))
	filtered_spectra_idxs = findall(idx_nan)
	f_idx, s_idx = 1, 1
	while ((f_idx <= length(year_sizes)) && (s_idx <= length(filtered_spectra_idxs)))
		global f_idx, s_idx
		if (filtered_spectra_idxs[s_idx] < year_idxs[f_idx])
			filtered_idxs[f_idx] += 1
		else
			filtered_idxs[f_idx+1] += 1
			f_idx += 1
		end
		s_idx += 1
	end

	fs_power_yearly = Vector{typeof(1.0u"W")}()
	for spectrum in wave_spectra
	    wave = Waves.wave_amplitudes(spectrum, freq)
	    append!(fs_power_yearly, mean(fs_linearmodel.(eachrow(wave))))
	end
	fs_power_mean = mean(fs_power_yearly)	
	nothing
	# md"""
	# ```math
	# $$\begin{aligned}
	# & \text {Table 1}\\
	# &\begin{array}{cccc}
	# \hline \hline \text { Year } & \text { No. Spectra } & \text { \% Spectra } & \text { Avg. Power } \\
	# \hline 
	# A & B & C & D\\
	# \hline
	# \end{array}
	# \end{aligned}$$
	# ```
	# """
end

# ╔═╡ 08f935b2-0aa1-4990-9f53-bc3f058ba6d5
let
	table_data_str = "
		\`\`\`math\n
		\$\$\\begin{aligned}\n& \\text {Table 1}\\\\\n
		&\\begin{array}{cccc}\n
		\\hline \\hline \\text { Year } & \\text { No. Spectra } & \\text { \\% Spectra } & \\text { Avg. Power } \\\\\n\\hline\n"
	spectraUsed = size.(wave_spectra,1).-filtered_idxs
	runningTotal = 0
	for (i, e) in enumerate(years)
		hoursInYear = Dates.value(Dates.Hour(Date(Dates.Year(e+1))-Date(Dates.Year(e))))
		percOfSpectra = roundedValue(spectraUsed[i]/hoursInYear*100)
		runningTotal += hoursInYear
		str = "$e & $(spectraUsed[i]) & $(percOfSpectra)\\% & $(roundedUnitful(fs_power_yearly[i]; digits=0)) \\\\\n"
		table_data_str *= str
	end
	percAll = size(filtered_spectra, 1)/runningTotal*100
	table_data_str *= "& $(size(filtered_spectra, 1)) & $(roundedValue(percAll))\\% & $(roundedUnitful(fs_power_mean; digits=0)) \\\\" 
	table_data_str *= "
		\\hline\n
		\\end{array}\n
		\\end{aligned}\$\$\n\`\`\`\nUsable wave spectra from NDBC Buoy 46050 per year, the amount retained expressed as a percentage, and direct mean annual power estimates (based on all spectra) used as a synthetic ground truth."
	# print(table_data_str)
	Markdown.parse(table_data_str)
end

# ╔═╡ a1073f30-312c-4627-97d1-43aa00da4e31
md"""
### Mean Annual Power
"""

# ╔═╡ 011c0dda-4996-4956-bedf-811043876bc8
let
	power_mean = mean(fs_power_yearly)
	Norm_samples = Normal(ustrip(mean(fs_power_yearly)), ustrip(std(fs_power_yearly)))
	Norm_mean = Normal(ustrip(mean(fs_power_yearly)), std(ustrip.(fs_power_yearly))/√nyears)

	norm_ppf2(N, p=0.95) = quantile(N, 1-(1+p)/2), quantile(N, 1-(1-p)/2)
	# CI_min, CI_max = norm_ppf2(ustrip.(fs_power_yearly))
	CI_min, CI_max = norm_ppf2(Norm_mean)
	CI_delta = ((CI_max - CI_min)/2)u"W"
	
	#Ground Truth
	power_delta = CI_delta#108.11542037210347u"kg*m^2*s^-3"
	CI_min, CI_max = (fs_power_mean - power_delta), (power_mean + power_delta)

	target_size = (line_width*0.4, line_width*0.5)
	# target_size = (line_width*0.5, line_width*0.6)
	figsize =  target_size .* (inToPt/pt_per_unit)
	f = Figure(
		size = figsize
	)
	ax = CairoMakie.Axis(f[1,1], 
				xlabel="Year", xticks=2010:5:2021, 
				ylabel=L"Mean Annual Power ($W$)", yticks=750:250:1751; ax_kwargs...)
	scatter!(years, ustrip(fs_power_yearly))
	ylims!(ax, 900, 1800)

	hlines!(ustrip(fs_power_mean), linestyle=:solid, color=:black)
	hlines!(ustrip(CI_min), linestyle=:dash, color=:black)
	hlines!(ustrip(CI_max), linestyle=:dash, color=:black)
	save("./data/figures/figure4.pdf", f; pt_per_unit)
	md"""
	#### Figure 4
	$(f)\

	Average annual power: $(roundedUnitful(fs_power_mean)) +/- $(roundedUnitful(CI_delta)) (95% CI))
	"""
end

# ╔═╡ ee0b7559-ea07-45b3-b1a9-0c7b912f3379
md"""
### Spectral Shape and Scaling
"""

# ╔═╡ 3136fd1a-2533-4ba6-a02b-61386e82376b
begin
	α = 1/5
	n_freq = freq/α
	n_freqs = freqs/α
end

# ╔═╡ 0e839f6c-9dc7-4dab-9922-9eb505b6d597
function plot_bin_spectra(data, center; target_size = (4, 3), subset=0, spec=false, f=n_freqs)
	rand_subset = subset > 0 ? rand(1:size(data, 1), subset) : (1:size(data, 1))
	# special_subset = [508, 4746, 105] # 16 Bins
	# special_subset = [165, 1272, 1208] # 32 Bins
	special_subset = [1104, 670, 856]
	figsize = target_size .* (inToPt/pt_per_unit)
	fig = Figure(
		size = figsize
	)
	ax = CairoMakie.Axis(fig[1,1],
		xlabel = L"Frequency $(Hz)$",
		ylabel = L"Wave Spectrum $(m^2/Hz)$";
		ax_kwargs...,
	)
	ylims!(ax, 0, maximum(data))
    xlims!(ax, 0, maximum(f))
	y = reduce(vcat, [[a; NaN] for a in eachrow(data[rand_subset, :])])
	x = reduce(vcat, [[a; NaN] for a in [f for _ in 1:length(rand_subset)]])
	
	lines!(ax,
		x, y,
		# label="Measured Spectrum",
		color=:lightgray, linewidth=0.5,
		# color=:gray, linewidth=2,
		rasterize=5
	)
	if spec
		spec_data = eachrow(data[special_subset, :])
		linestyles = [(:dot, :dense), (:dashdot, :dense), (:dash, :dense)]
		markerShapes = [:vline, :hline, :xcross]
		colors = [1:7...]
		for i in 1:length(special_subset)
			lines!(ax,
				f, spec_data[i], alpha=1, 
				color=Cycled(i),
				linestyle=linestyles[i]
			)
		end
	elseif subset > 0
		println(rand_subset)
		rand_data = eachrow(data[rand_subset, :])
		for i in 1:subset
			lines!(ax,
				f, rand_data[i], alpha=1, label="$(rand_subset[i])"
			)
		end
	else
		pm = ustrip.(Waves.pm_spectrum(ustrip(center[2])*u"m", ustrip(Waves.te_to_tp(center[1])).*u"s").(f.*u"Hz"))
		lines!(ax,
			f, pm,
			color=pm_color, linestyle=pm_style, label="P-M"
		)
	end

	dof = 24
	norm_ppf2(N, p=0.95) = quantile(N, 1-(1+p)/2), quantile(N, 1-(1-p)/2)
	dist = [norm_ppf2(Chisq(dof))...]
	mean_spectrum = mean(eachrow(data))
	CI_bounds = transpose(dist)/dof .* mean_spectrum
	lines!(ax,
		f, mean_spectrum,
		color=:black, label="Mean"
	)
	if !spec
		lines!.(ax,
			[f, f], eachcol(CI_bounds),
			color=:black, label="95% CI",
			linestyle=:dash
		)
	end
	return fig, ax
end

# ╔═╡ 0def8a11-3054-4c09-ab55-e8823753330b
let
	# Lines, Labels, remove border for centers
	# idx = [37, 117] # 16 Bins
	idx = [168+32*-2, 136+32*6] #32 Bins
	α = 1
	bins = 32
	
	data = [ustrip.(te_vector), ustrip.(hs_vector)]
	(centers, edges), (weights, _) = ExtraFunctions.bins(data, [bins for _ in 1:length(data)])

	target_size = (line_width*0.8, line_width*0.8*0.75)
	# target_size = (line_width*0.95, line_width*0.95*0.75)
	f1, ax = plot_bins(data, bins; target_size)
	scatter!(ax,
		collect(centers)[idx], 
		color=:red, label=nothing, 
		markersize = 7,
	)

	steepness_limit(x) = (0.05 * x^2 * 9.81) / (2 * π)
	steepness_limit_2(x) = (0.0625 * x^2 * 9.81) / (2 * π)
	steep_xs = 4:0.1:12
	steep_ys = steepness_limit.(steep_xs)

	steep_ys_2 = steepness_limit_2.(steep_xs)
	lines!(ax, steep_xs, steep_ys, color=Cycled(2), linestyle=:dash)
	lines!(ax, steep_xs, steep_ys_2, color=Cycled(2), linestyle=:solid)
	# f1
	center_idx = reduce(hcat, [t...] for t in collect(centers)[idx])
	x_bins = collect(edges[1])
	y_bins = collect(edges[2])

	h_bounds = get_bound_axis.(center_idx[1, :], [x_bins for _ in length(idx)])
	v_bounds = get_bound_axis.(center_idx[2, :], [y_bins for _ in length(idx)])

	bin_filter = [filter(data, h[1], v[1]) for (h, v) in zip(eachrow(h_bounds), eachrow(v_bounds))]
	mini_binned_data = [ustrip.(filtered_spectra*α^5)[bf, :] for bf in bin_filter]

	bin_idx = 1
	target_size = (line_width*0.47, line_width*0.47)
	# target_size = (line_width*0.75, line_width*0.75*0.75)
	f2, ax = plot_bin_spectra(mini_binned_data[bin_idx], center_idx[:, bin_idx]; target_size, subset=0, f=freqs)
	axislegend(ax, 
		position=:rt, halign=:right, valign=:top, 
		framewidth=0.5, rowgap = -8, patchlabelgap = 2,
		padding=4, merge=true
	)
	
	bin_idx = 2
	f3, ax = plot_bin_spectra(mini_binned_data[bin_idx], center_idx[:, bin_idx]; target_size, subset=0, f=freqs)
	# f1,f2,f3
	bin_idx = 1
	target_size = (line_width*0.7, line_width*0.7*0.75)
	# target_size = (line_width*0.95, line_width*0.95*0.75)
	f4, ax = plot_bin_spectra(mini_binned_data[bin_idx], center_idx[:, bin_idx]; target_size, subset=0, spec=true, f=freqs)
	# axislegend(ax)
	p1 = roundedValue.([center_idx[1, 1], center_idx[2, 1]])
	p2 = roundedValue.([center_idx[1, 2], center_idx[2, 2]])
	# println("Te: $(center_idx[1, :])")
	# println("Tp: $(Waves.te_to_tp.(center_idx[1, :]))")
	# println("Hs: $(center_idx[2, :])")

	b1 = (collect(centers)[1])
	b2 = (collect(centers)[34])

	println(b1)
	println("Bin Sizes (Tₑ, Hₛ): $(b2 .- b1)")
	
	f4
	save("./data/figures/figure1a.pdf", f1; pt_per_unit)
	save("./data/figures/figure1b.pdf", f2; pt_per_unit)
	save("./data/figures/figure1c.pdf", f3; pt_per_unit)
	save("./data/figures/figure2.pdf", f4; pt_per_unit)
	md"""
	#### Figure 1
	$(f1)\
	$(f2) $(f3)\
	Bottom left figure is the lower bin Tₑ = $(p1[1]), Hₛ = $(p1[2]). \
	Bottom right figure is the upper bin Tₑ = $(p2[1]), Hₛ = $(p2[2]).\
	
	#### Figure 2
	$(f4)
	"""
end

# ╔═╡ dbd9685e-17ff-4223-9f5f-0b3cf6777f0c
let
	target_size = (line_width*0.7, line_width*0.7*0.75)
	figsize = target_size .* (inToPt/pt_per_unit)
	f = Figure(
		size = figsize
	)
	ax = CairoMakie.Axis(f[1,1],
		xlabel = "Period (s)";
		xticks = 0:2:16,
		yticks = 0:0.2:1.0,
		ax_kwargs...,
	)
	xlims!(0, 15)
	ylims!(0, 1.1)
	ω, Zᵢ, Zₚₜₒ, Zₜₕ, _, _, _ = WaveBot.preprocessing("./data/bem_scaled_waves"; freq_wave=n_freq)
	bins=15

	origTeHist = normalize(fit(Histogram, ustrip.(te_vector)); mode=:pdf)
	y1 = origTeHist.weights
	aux_te = ustrip.(te_vector)*α
	scaledTeHist = normalize(fit(Histogram, aux_te); mode=:pdf)
	y2 = scaledTeHist.weights

	x1_step = step(origTeHist.edges[1])
	x1 = origTeHist.edges[1] .+ x1_step/2
	x2 = x1./5
	
	Zₜₕ = vec(Zₜₕ)
	Zᵢ = vec(Zᵢ)
	Z_l = conj(Zₜₕ)
	Z_fu, Z_fi, Z_vu, Z_vi = vec(Zₚₜₒ[1, 1, :]), 
							vec(Zₚₜₒ[1, 2, :]), 
							vec(Zₚₜₒ[2, 1, :]), 
							vec(Zₚₜₒ[2, 2, :])

	Ga = @. abs(Z_vu / (Zᵢ + Z_fu))^2 * (real(Zᵢ) / real(Zₜₕ))
	Ga = @. ustrip(Ga)

	lines!(ax,
		x1[begin:end-1], y1, label="Original Tₑ",
		linestyle=:dashdot
	)
	lines!(ax,
		x2[begin:end-1], y2, label="Scaled Tₑ",
		linestyle=:dash
	)
	x3 = ustrip.((2π*DimensionfulAngles.radᵃ)./ω)
	lines!(ax,
		x3,	Ga, label=L"G"
	)
	axislegend(ax, 
		position=:rt, halign=:right, valign=:top, 
		framewidth=0.5, rowgap = -8, patchlabelgap = 1,
		padding=2
	)
	save("./data/figures/figure5.pdf", f; pt_per_unit)
	md"""
	#### Figure 5
	$(f)
	"""
end

# ╔═╡ a6bd37ca-ee05-40f7-adf0-951a9010ac5a
md"""
## Results
"""

# ╔═╡ a9a28f60-e571-4967-b25f-610f777fea6d
function my_linear_interpolation(x, x_vals, y_vals)
    # Ensure that x is within the range of x_vals
    if !(minimum(x_vals) <= x <= maximum(x_vals)) 
        return 0
    end
    # Find the two nearest values in x_vals
    idx_right = searchsortedfirst(x_vals, x)
    idx_left = idx_right - 1
    # Get the corresponding y values
    y_left = y_vals[idx_left]
    y_right = y_vals[idx_right]
    # Perform linear interpolation
    y_interp = y_left + (y_right - y_left) * (x - x_vals[idx_left]) / (x_vals[idx_right] - x_vals[idx_left])
    return y_interp
end

# ╔═╡ 884c37c2-4caa-4f63-a03c-30a1242383df
function plot_w_params(S, f, ax; color=3)
	Hs = Waves.significant_waveheight(S, f)
	Te = Waves.energy_period(S,f)
	p = lines!(ax, f, S, color=Makie.wong_colors()[color])
	p, Hs, Te
end

# ╔═╡ 8feca466-305f-477d-a654-3a9358279402
# let
# 	idx = [54904]
# 	# idx = rand(1:size(padded_data,2), 5)
# 	if size(idx, 1)==1
# 		f = process_of_idx(idx[1])
# 		save("figures/figure8.pdf", f; pt_per_unit)
# 		md"""
# 		#### Figure 8
# 		$(f)\

# 		Process of sample $(idx)
# 		"""
# 	else
# 		println(idx)
# 		processes = [process_of_idx(i) for i in idx]
# 		processes
# 	end
# end

# ╔═╡ b3d56270-b7fc-4ff7-bef4-7fdc0cec02f6
md"""
### Autoencoder Training
"""

# ╔═╡ 48d3ce5b-90e7-4e2d-8125-1e4498819ba9
begin
	# Autoencoder
	
	# Normalization
	modified_orig_freqs = Float32.([1e-5, freqs[1]-eps(Float32), freqs..., freqs[end]+eps(Float32), 2.1])
	x_new = range(0.01, 7.5, 200)
	n_samples = size(filtered_spectra, 1)
	
	padded_data = transpose(hcat(zeros(n_samples), zeros(n_samples), ustrip.(filtered_spectra), zeros(n_samples), zeros(n_samples)))
	normalized_data = AutoEncoder.new_modified_softmax(padded_data; orig_freq = modified_orig_freqs, new_freqs=x_new)
	
	# Initialization
	features = size(normalized_data, 1)
	layers = [features, 32, 16, 2]
	model = AutoEncoder.init_model(features, layers, x_new, x_new)
	loss(sample) = sqrt.(Flux.Losses.mse(sample, model(sample)))
	# Training (Optional).
	# begin
	#     η = 4e-3
	#     opt = Flux.setup(Flux.Adam(), model)
	#     history = AutoEncoder.train!(model, opt, loss, transpose(normalized_data), 0; epochs=20, batchsize=512, eps=1e-5, filename="model_checkpoint_new")
	# end

	# Model Loading
	m_state = JLD2.load("data/models/trained_autoencoder.jld2", "m_state");
	
	Flux.loadmodel!(model, m_state);
	
	encoder = model[1:length(layers)-1]
	decoder = model[length(layers):end]
	display(model)
end

# ╔═╡ 800de494-fd21-4c78-b617-12c348b4e27f
function process_of_idx(idx)
	target_size = (text_width*0.9, text_width*0.9*0.75)
	figsize = target_size .* (inToPt/pt_per_unit)
	fig = Figure(;size=figsize)
	ax_vec = Vector{}()
	for i in 1:6
		# println((i-1)÷3+1,",",(i-1)%3+1)
		if ((i-1)%3+1 == 3) && ((i-1)÷3+1 == 1)
			ax = CairoMakie.Axis(fig[(i-1)÷3+1,(i-1)%3+1]; 
					xticksvisible=false, xticklabelsvisible=false,
					yticksvisible=false, yticklabelsvisible=false, ax_kwargs...)
		elseif ((i-1)÷3+1 == 1)
			ax = CairoMakie.Axis(fig[(i-1)÷3+1,(i-1)%3+1]; 
					xticksvisible=false, xticklabelsvisible=false, ax_kwargs...)
		elseif ((i-1)%3+1 == 3)
			ax = CairoMakie.Axis(fig[(i-1)÷3+1,(i-1)%3+1]; 
					yticksvisible=false, yticklabelsvisible=false, ax_kwargs...)
		else
			ax = CairoMakie.Axis(fig[(i-1)÷3+1,(i-1)%3+1]; ax_kwargs...)
		end
		append!(ax_vec, [ax])
	end
	
	f = modified_orig_freqs
	S = padded_data[:, idx]
	Hs = Waves.significant_waveheight(S, f)
	Te = Waves.energy_period(S,f)
	

	f̂ = f.*Te
	Ŝ = S/(Hs^2 * Te)

	f̂ₛ = x_new
	Ŝint = [my_linear_interpolation(x, f̂, Ŝ) for x in x_new]
	
	f̂ₛ = f̂ₛ
	Ŝdint = vec(model(reshape(Ŝint, (200, 1))))
	
	f̂ = f̂#f̂dense#
	extp = extrapolate(interpolate((f̂ₛ,), Ŝdint, Gridded(Linear())), 0)
	Ŝd = extp.(f̂)
	
	f = f#fdense#
	Sd = Ŝd*(Hs^2*Te)

	xlimits = (0, 0.5)
	ylimits = (0-maximum(S)*0.05, maximum(S)*1.2)
	xlims!(ax_vec[1], xlimits...)
	ylims!(ax_vec[1], ylimits...)
	xlims!(ax_vec[4], xlimits...)
	ylims!(ax_vec[4], ylimits...)
	fig_gl = fig[1,1] = GridLayout()
	A1, Hs, Te = plot_w_params(S[3:end-2], f[3:end-2], ax_vec[1]; color=1)
	A2, Hs2, Te2 = plot_w_params(Sd[3:end-2], f[3:end-2], ax_vec[4]; color=2)

	xlimits = (0, 4)
	ylimits = (0-maximum(Ŝ)*0.05, maximum(Ŝ)*1.2)
	xlims!(xlimits...)
	ylims!(ylimits...)
	xlims!(ax_vec[2], xlimits...)
	ylims!(ax_vec[2], ylimits...)
	xlims!(ax_vec[3], xlimits...)
	ylims!(ax_vec[3], ylimits...)
	xlims!(ax_vec[5], xlimits...)
	ylims!(ax_vec[5], ylimits...)
	xlims!(ax_vec[6], xlimits...)
	ylims!(ax_vec[6], ylimits...)
	B1, _, _ = plot_w_params(Ŝ[3:end-2], f̂[3:end-2], ax_vec[2]; color=1)
	C1, _, _ = plot_w_params(Ŝint, f̂ₛ, ax_vec[3]; color=1)
	B2, _, _ = plot_w_params(Ŝd[3:end-2], f̂[3:end-2], ax_vec[5];color=2)
	C2, _, _ = plot_w_params(Ŝdint, f̂ₛ, ax_vec[6];color=2)
	text!(ax_vec[1], 1, 1, text = L"S(f)", align = (:right, :top), 
		space = :relative, offset = (-3, -2))
	text!(ax_vec[2], 1, 1, text = L"\tilde{S}(\tilde{f})", align = (:right, :top), 
		space = :relative, offset = (-3, -2))
	text!(ax_vec[3], 1, 1, text = L"\tilde{S}(\tilde{f}')", align = (:right, :top),
		space = :relative, offset = (-3, -2))
	text!(ax_vec[6], 1, 1, text = L"\tilde{S}_{d}(\tilde{f}')", align = (:right, :top),
		space = :relative, offset = (-3, -2))
	text!(ax_vec[5], 1, 1, text = L"\tilde{S}_{d}(\tilde{f})", align = (:right, :top),
		space = :relative, offset = (-3, -2))
	text!(ax_vec[4], 1, 1, text = L"S_{d}(f)", align = (:right, :top),
		space = :relative, offset = (-3, -2))

	rowgap!(fig.layout, 5)
	colgap!(fig.layout, 10)

	# perc_diff = (abs(Hs-Hs2)/Hs*100, abs(Te-Te2)/Te*100)
	# println(perc_diff)
	# println(ax_vec)
	fig#all_plots = [A1, B1, C1, A2, B2, C2]

end

# ╔═╡ ac9630f9-f455-4620-ba0d-2a33fbc1fa0b
md"""
### Autoencoder Spectra
"""

# ╔═╡ 6aa09a4f-f4e9-43a8-8348-336f70954dc6
encoded_spectra = encoder(normalized_data)

# ╔═╡ a0ab4a43-3723-4fbf-a209-f97786ddabc8
function plot_bin_spectra_aux_n(data, center, ax; hs=1, te=1, target_size = (4, 3), subset=0, spec=false, color=:green)
	rand_subset = subset > 0 ? rand(1:size(data, 1), subset) : (1:size(data, 1))
	special_subset = [340, 1504, 4832]
	
	# figsize = target_size .* (inToPt/pt_per_unit)
	# f = Figure(
	# 	size = figsize
	# )
	# ax = CairoMakie.Axis(f[1,1],
	# 	xlabel = L"Frequency $(Hz)$",
	# 	ylabel = L"Wave Spectrum $(m^2/Hz)$";
	# 	ax_kwargs...,
	# )
	# ylims!(ax, 0, 0.0025)
	ylims!(ax, 0, 10)
    xlims!(ax, 0, 0.4)
	y = reduce(vcat, [[a; NaN] for a in eachrow(data[rand_subset, :])])
	x = reduce(vcat, [[a; NaN] for a in [freqs for _ in 1:length(rand_subset)]])
	lines!(ax,
		x, y,
		# label="Measured Spectrum",
		color=:gray, linewidth=0.5,
		rasterize=3
	)
	dof = 24
	norm_ppf2(N, p=0.95) = quantile(N, 1-(1+p)/2), quantile(N, 1-(1-p)/2)
	dist = [norm_ppf2(Chisq(dof))...]
	mean_spectrum = mean(eachrow(data))
	CI_bounds = transpose(dist)/dof .* mean_spectrum
	lines!(ax,
		freqs, mean_spectrum,
		color=:black, label="Mean"
	)
	lines!.(ax,
		[freqs, freqs], eachcol(CI_bounds),
		color=:black,# label="95% CI",
		linestyle=:dash
	)
	decoder_input = transpose(reduce(hcat, vec(center)))
	decoder_output = transpose(decoder(decoder_input))
	ae_spec = transpose(AutoEncoder.unnormalize_data(decoder_output, [ustrip(te)], [ustrip(hs)], freqs, x_new))
	# println(ae_spec)
	lines!(ax,
		freqs, vec(ae_spec),
		color=ae_color, label="$ae_label"
	)
end

# ╔═╡ 22542d59-202d-435d-8587-4199aa75bb27
let
	autoencoder_data = DataFrame([hs_vector, te_vector, encoded_spectra[1,:], encoded_spectra[2,:]], ["hs", "te", "θ1", "θ2"])
	data = [ustrip.(autoencoder_data.te), ustrip.(autoencoder_data.hs)]
	
	bins = 32
	idx = 0
	if bins==16
		idx = [37, 117]
	else
		# idx = [169, 137+32*10]
		idx = [168+32*-2, 136+32*6]
	end
	
	bin_idx = 1
	
	(centers, edges), (weights, _) = ExtraFunctions.bins(data, [bins for _ in 1:length(data)])
	center_idx = reduce(hcat, [t...] for t in collect(centers)[idx])
	x_bins = collect(edges[1])
	y_bins = collect(edges[2])
	# println("Te: $(center_idx[1, bin_idx])")
	# println("Hs: $(center_idx[2, bin_idx])")
	
	h_bounds = get_bound_axis.(center_idx[1, :], [x_bins for _ in length(idx)])
	v_bounds = get_bound_axis.(center_idx[2, :], [y_bins for _ in length(idx)])
	
	hs = center_idx[2, bin_idx]
	te = center_idx[1, bin_idx]
	println("Hs & Te: ",roundedValue(hs), " & ",roundedValue(te))
	pm = ustrip.(Waves.pm_spectrum(ustrip(hs)*u"m", ustrip(Waves.te_to_tp(te)).*u"s").(freq))
	
	filter(data, h_bound, v_bound) = (data[1] .> h_bound[1] .&& data[1] .< h_bound[2]) .&& (data[2] .> v_bound[1] .&& data[2] .< v_bound[2])
	bin_filter = [filter(data, h[1], v[1]) for (h, v) in zip(eachrow(h_bounds), eachrow(v_bounds))]
	mini_binned_data = [ustrip.(filtered_spectra)[bf, :] for bf in bin_filter]
	mini_binned_encoded = [encoded_spectra[:, bf] for bf in bin_filter]
	data = eachrow(mini_binned_encoded[1])

	if bins==16
		idx = [102, 104, 106]
		idx = [e.+idx for e in [0, 32, 64]]
	else
		bl = 493+64
		stp = 6
		idx = [e for e in bl:stp:(bl+stp*2)]
		idx = [e.+idx for e in [0, 32*stp, 64*stp]]
	end
	
	idx = vcat(idx...)
	(centers, edges), (weights, _) = ExtraFunctions.bins(data, [bins for _ in 1:length(data)])

	target_size = (line_width*0.6, line_width*0.6)
	# target_size = (line_width*0.95, line_width*0.95)
	f1, ax1 = plot_bins(data, bins; scatter_alpha=1, target_size, xlab=L"Shape Parameter $θ_1$", ylab=L"Shape Parameter $θ_2$")
	scatter!(ax1,
		collect(centers)[idx], 
		color=:red, label=nothing, 
		markersize = 9,
	)
	f1
	center_idx = reduce(hcat, [t...] for t in collect(centers)[idx])
	x_bins = collect(edges[1])
	y_bins = collect(edges[2])
	println("Θ1: $(center_idx[1, :])")
	println("Θ2: $(center_idx[2, :])")
	h_bounds = get_bound_axis.(center_idx[1, :], [x_bins for _ in length(idx)])
	v_bounds = get_bound_axis.(center_idx[2, :], [y_bins for _ in length(idx)])

	bin_filter = [filter(data, h[1], v[1]) for (h, v) in zip(eachrow(h_bounds), eachrow(v_bounds))]
	mini_binned_data = [mini_binned_data[1][bf, :] for bf in bin_filter]

	target_size = (line_width*0.97, line_width*0.97*0.75)
	# target_size = (line_width*0.95, line_width*0.75)
	figsize = target_size .* (inToPt/pt_per_unit)
	f2 = Figure(size=figsize; figure_padding = 1)
	for r in 0:2
		for c in 1:3
			# println(4-r,",", c)
			if (c == 1 && r==0)
				ax = CairoMakie.Axis(f2[3-r, c], 
					yticks=0:3:15, xticks=0:0.1:0.3,
					xticksvisible=true, yticksvisible=true,
					xticklabelsvisible=true, yticklabelsvisible=true; ax_kwargs...
				)
			elseif r == 0
				if c == 2
					ax = CairoMakie.Axis(f2[3-r, c],
						xlabel=L"Frequency ($Hz$)",
						yticks=0:3:15, xticks=0:0.1:0.3,
						xticksvisible=true, yticksvisible=false,
						xticklabelsvisible=true, yticklabelsvisible=false; ax_kwargs...
					)
				else
					ax = CairoMakie.Axis(f2[3-r, c],
						yticks=0:3:15, xticks=0:0.1:0.3,
						xticksvisible=true, yticksvisible=false,
						xticklabelsvisible=true, yticklabelsvisible=false; ax_kwargs...
					)
				end
			elseif c == 1
				if r == 1
					ax = CairoMakie.Axis(f2[3-r, c],
						ylabel=L"Wave Spectrum ($m^2/Hz$)",
						yticks=0:3:15, xticks=0:0.1:0.3,
						xticksvisible=false, yticksvisible=true,
						xticklabelsvisible=false, yticklabelsvisible=true; ax_kwargs...
					)
				else
					ax = CairoMakie.Axis(f2[3-r, c],
						yticks=0:3:15, xticks=0:0.1:0.3,
						xticksvisible=false, yticksvisible=true,
						xticklabelsvisible=false, yticklabelsvisible=true; ax_kwargs...
					)
				end
			else
				ax = CairoMakie.Axis(f2[3-r, c],
					yticks=0:3:15, xticks=0:0.1:0.3,
					xticksvisible=false, yticksvisible=false,
					xticklabelsvisible=false, yticklabelsvisible=false; ax_kwargs...,
				)
			end
			# println("r+c: $(r*3+c)")
			mbd_idx = r*3+c
			plot_bin_spectra_aux_n(mini_binned_data[mbd_idx], center_idx[:, mbd_idx], ax; hs, te, target_size, subset=0, color=Cycled(1))
			xlims!(ax, 0, 0.33)
			ylims!(ax, 0, 6.5)
			lines!(ax, freqs, pm,
				color=pm_color, linestyle=pm_style, label=pm_label
			)
			if (c==3 && r==2)
				axislegend(ax, 
					position=:rt, halign=:right, valign=:top, 
					framewidth=0.5, rowgap = -8, patchlabelgap = 2,
					padding=4, merge=true
				)
			end
		end
	end
	rowgap!(f2.layout, 5)
	colgap!(f2.layout, 5)
	
	save("./data/figures/figure8.pdf", f1; pt_per_unit)
	save("./data/figures/figure9.pdf", f2; pt_per_unit)
	md"""
	#### Figure 8
	$(f1)
	#### Figure 9
	$(f2)
	"""
end

# ╔═╡ 45455ee7-d396-4ace-aaf4-af9f0c68e3fd
md"""
### Mean Annual Power -- Measured Waves
"""

# ╔═╡ fd1771fc-6bfa-49dd-a07f-4b200b3d9d6c
begin
	fs_pm_spectra, fs_ma_spectra, fs_AE_spectra = 0, 0, 0
	if isfile("./data/cache/fs_spectras.jld2")	
		jldopen("./data/cache/fs_spectras.jld2", "r") do f
			global fs_pm_spectra, fs_ma_spectra, fs_AE_spectra
			fs_pm_spectra = f["fs_pm_spectra"]
			fs_ma_spectra = f["fs_ma_spectra"]
			fs_AE_spectra = f["fs_AE_spectra"]
		end
		println("Loaded spectra data from file!")
	else
		global fs_pm_spectra, fs_ma_spectra, fs_AE_spectra
		@info "PM"
		fs_pm_funcs = Waves.pm_spectrum.(hs_vector, Waves.te_to_tp.(te_vector))
		fs_pm_spectra = reduce(hcat, [s.(freq) for s in fs_pm_funcs])
		@info "AE"
		fs_decoder_input = transpose(reduce(hcat, [encoded_spectra[1,:], encoded_spectra[2,:]]))
		fs_decoder_output = transpose(decoder(fs_decoder_input))
		fs_AE_spectra = transpose(AutoEncoder.unnormalize_data(fs_decoder_output, ustrip.(te_vector), ustrip.(hs_vector), freqs, x_new))
		@info "Mackay"
		fs_ma_funcs = Waves.mackay_spectrum.(mackay_data.hs, mackay_data.te, mackay_data.dtn, mackay_data.han2; mat_data)
		fs_ma_spectra = reduce(hcat, [s.(freq) for s in fs_ma_funcs])
		
		JLD2.jldsave("./data/cache/fs_spectras.jld2"; fs_pm_spectra, fs_ma_spectra, fs_AE_spectra)
		println("Saved spectra data to file!")
	end
end

# ╔═╡ 2e058679-8e47-48a8-9b3a-cd1021bcc13a
begin
	bins_big = [2,4,8,16,32,64,128,256,512]
	bins_small = [2,4,8,16,32]
	
	fs_data_2d = [te_vector, hs_vector]
	fs_data_4d_mackay = [mackay_data.te, mackay_data.hs, mackay_data.han2, mackay_data.dtn]
	fs_data_4d_ae = [te_vector, hs_vector, encoded_spectra[1,:], encoded_spectra[2,:]]
	
	fs_bin_powers, fs_ma_powers, fs_ae_powers = 0, 0, 0
	if isfile("./data/cache/bins_fs.jld2")
		jldopen("./data/cache/bins_fs.jld2", "r") do f
			global fs_bin_powers, fs_ma_powers, fs_ae_powers
			fs_bin_powers = f["fs_bin_powers"]
			fs_ma_powers = f["fs_ma_powers"]
			fs_ae_powers = f["fs_ae_powers"]
		end
		println("Loaded fullsize bin data from file!")
	else
		global fs_bin_powers, fs_ma_powers, fs_ae_powers
		fs_mats, fs_bin_powers = ExtraFunctions.bin_power(freq, fs_thevenin, fs_data_2d, bins_big)
		fs_aux_bins1 = [ExtraFunctions.solve_4d(freq, fs_thevenin, fs_data_4d_mackay, bin, mat_data) for bin in bins_small]
		fs_ma_powers = [mat_power[2] for mat_power in fs_aux_bins1]
		fs_aux_bins2 = [ExtraFunctions.solve_4d_AE(freqs, fs_thevenin, fs_data_4d_ae, bin, decoder, modified_orig_freqs, x_new) for bin in bins_small]
		fs_ae_powers = [mat_power[2] for mat_power in fs_aux_bins2]
		
		JLD2.jldsave("./data/cache/bins_fs.jld2"; fs_bin_powers, fs_ma_powers, fs_ae_powers)
		println("Saved fullsize bin data to file!")
	end
end

# ╔═╡ fc0ba65c-15f6-44ee-a1b0-b431a75b2b59
fs_avg_p(spectra_iter) = mean([uconvert(u"W", fs_linearmodel(Waves.wave_amplitudes(spectrum, freq))) for spectrum in spectra_iter])

# ╔═╡ 675f1c6a-eb9c-4020-b2d9-787e943d0379
norm_ppf2(N, p=0.95) = quantile(N, 1-(1+p)/2), quantile(N, 1-(1-p)/2)

# ╔═╡ 2100390f-0ef9-4e19-bb3d-86e9331cf187
function plot_baseline(gt_limit, base_limit, base_bins, ci_min, ci_max; 
						axis=false, target_size=(text_width*0.65, text_width*0.65*0.75), ylimits=(6,11))
	figsize = target_size .* (inToPt/pt_per_unit)
	f = Figure(
		size = figsize
	)
	ax = CairoMakie.Axis(f[1,1],
		xlabel = "Bins per Dimension", xscale=log2,
		ylabel = L"Mean Annual Power ($W$)";
		ax_kwargs..., xticks=bins_big, 
	)
	
	ylims!(ax, ylimits...)
	y_vals = ustrip.([gt_limit, ci_min, ci_max, base_limit])
	styles = [:solid, :dash, :dash, pm_style]
	colors = [:black, :black, :black, pm_color]
	labels = ["Truth", "95% CI", "95% CI", "P-M"]
	band!(0:1:600, ustrip.(ci_min), ustrip.(ci_max),
			color=:lightgray, alpha=0.5)
	for (val, linestyle, color, label) in zip(y_vals, styles, colors, labels)
		hlines!(ax, val, linestyle=linestyle, 
			color=color, label=label)
	end
	
	scatter!(ax,
		base_bins[1], ustrip.(base_bins[2]),
		color=pm_color, marker=pm_marker, label="P-M"# bins"
	)
	if axis
		Legend(f[1, 2], ax, framewidth=0.5, 
			rowgap = -8, patchlabelgap = 2,
			padding=4, merge=true
		)
		# axislegend(ax[1,2], 
		# 	position=:rt, halign=:right, valign=:top, 
		# 	framewidth=0.5, rowgap = -8, patchlabelgap = 2,
		# 	padding=4, merge=true
		# )
	end
	f, ax
end

# ╔═╡ 1b73a718-93a9-47ea-bdb4-53e463653a50
function plot_limits_all(baseline_arr, mackay_arr, ae_arr; 
							target_size=(text_width*0.6, text_width*0.6*0.75), ylimits=(6,11))
	f, ax = plot_baseline(baseline_arr...; target_size, ylimits)

    hlines!(ax, ustrip(mackay_arr[1]), linestyle=ma_style, color=ma_color, label=ma_label)
    scatter!(bins_small, ustrip.(mackay_arr[2]), color=ma_color, marker=ma_marker, label=ma_label)
    hlines!(ax, ustrip(ae_arr[1]), linestyle=ae_style, color=ae_color, label=ae_label)
    scatter!(bins_small, ustrip.(ae_arr[2]), color=ae_color, marker=ae_marker, label=ae_label)

	Legend(f[1, 2], ax, framewidth=0.5, 
		rowgap = -8, patchlabelgap = 2,
		padding=4, merge=true, patchsize=(30,20)
	)
	colgap!(f.layout, 5)
	# axislegend(ax, 
	# 	position=:rt, halign=:right, valign=:top, 
	# 	framewidth=0.5, rowgap = -8, patchlabelgap = 2,
	# 	padding=4, merge=true
	# )
	f, ax
end

# ╔═╡ db263b34-2bb5-47a4-8638-a44fa0d44dc5
begin
	fs_GT_p∞ = fs_avg_p(eachrow(filtered_spectra))
	fs_PM_p∞ = fs_avg_p(eachcol(fs_pm_spectra))
	fs_M_p∞ = fs_avg_p(eachcol(fs_ma_spectra))
	fs_AE_p∞ = fs_avg_p(eachcol(fs_AE_spectra.*u"m^2/Hz"))

	# println("Powers - Mean of Means")
	# println("Ground Truth Limit: \t\t$(roundedUnitful(fs_power_mean))")
	# println("Pierson-Moskowitz Limit: \t$(roundedUnitful(fs_PM_p∞)), \t$(perc_error(fs_power_mean, fs_PM_p∞))%")
	# println("Mackay Limit: \t\t\t\t$(roundedUnitful(fs_M_p∞)), \t$(perc_error(fs_power_mean, fs_M_p∞))%")
	# println("Autoencoder Limit: \t\t\t$(roundedUnitful(fs_AE_p∞)), \t$(perc_error(fs_power_mean, fs_AE_p∞))%")
	# println("")
	
	println("Fullsize Spectra Errors")
	pm_rmse = ustrip(sqrt(Flux.Losses.mse(transpose(filtered_spectra), fs_pm_spectra)))
	mack_rmse = ustrip(sqrt(Flux.Losses.mse(transpose(filtered_spectra), fs_ma_spectra)))
	ae_rmse = sqrt(Flux.Losses.mse(ustrip(transpose(filtered_spectra)), fs_AE_spectra))
	println("Pierson-Moskowitz \tRMSE: $(roundedValue(pm_rmse))")
	println("Mackay \t\t\t\tRMSE: $(roundedValue(mack_rmse))")
	println("Autoencoder \t\tRMSE: $(roundedValue(ae_rmse))")

	# mean distribution
	# _Norm_samples = Normal(ustrip(mean(_power_yearly)), ustrip(std(_power_yearly)))
	Norm_mean = Normal(ustrip(mean(fs_power_yearly)), std(ustrip.(fs_power_yearly))/√nyears)
	CI_min, CI_max = norm_ppf2(Norm_mean)
	CI_delta = ((CI_max - CI_min)/2)u"W"
	println("Average annual power: $(roundedUnitful(fs_power_mean)) +/- $(roundedUnitful(CI_delta)) (95% CI))")
	
	#Ground Truth
	_CI_min, _CI_max = (fs_power_mean - CI_delta), (fs_power_mean + CI_delta)
	println("Average annual power: $(roundedUnitful(fs_power_mean))")
	
	# fs_baseline_bin_stats = [bins_big, fs_bin_powers]
	# fs_baseline_args = [fs_power_mean, fs_PM_p∞, fs_baseline_bin_stats, CI_min, CI_max]
	# target_size=(text_width*0.8, text_width*0.6*0.75)
	# f1, _ = plot_baseline(fs_baseline_args...; axis=true, 
	# 						target_size, ylimits=(1200, 1550))
end

# ╔═╡ 5fde7fea-e0c9-40e6-9de8-893db91b403a
let
	fs_baseline_bin_stats = [bins_small, fs_bin_powers[begin:length(bins_small)]]
	fs_baseline_args = [fs_power_mean, fs_PM_p∞, fs_baseline_bin_stats, CI_min, CI_max]
	fs_mackay_stats = [fs_M_p∞, fs_ma_powers]
	fs_ae_stats = [fs_AE_p∞, fs_ae_powers]
	target_size=(line_width*0.7, line_width*0.6*0.7)
	f2, _ = plot_limits_all(fs_baseline_args, fs_mackay_stats, fs_ae_stats; 
							target_size, ylimits=(1200, 1550))
	# save("figures/results_unscaled.pdf", f1; pt_per_unit)
	save("./data/figures/figure10.pdf", f2; pt_per_unit)
	md"""
	#### Figure 10
	
	$(f2)
	"""
end

# ╔═╡ 3b1764e2-7f5b-4bd4-9c27-e39725642c39
begin
	res1_data_str = "
		\`\`\`math\n
		\$\$\\begin{aligned}\n& \\text {Table 2}\\\\\n
		&\\begin{array}{cccc}\n
		\\hline \\hline \\text { Method } & \\text { Value } & \\text { Percent Error } \\\\\n\\hline\n"
	res1_data_str *= "\\text{Truth} & $(roundedUnitful(fs_power_mean)) & - \\\\\n"
	res1_data_str *= "\\text{Pierson-Moskowitz} & $(roundedUnitful(fs_PM_p∞)) & $(perc_error(fs_power_mean, fs_PM_p∞))\\\\\n"
	res1_data_str *= "\\text{Mackay} & $(roundedUnitful(fs_M_p∞)) & $(perc_error(fs_power_mean, fs_M_p∞))\\\\\n"
	res1_data_str *= "\\text{Autoencoder} & $(roundedUnitful(fs_AE_p∞)) & $(perc_error(fs_power_mean, fs_AE_p∞))\\\\\n"
	res1_data_str *= "
		\\hline\n
		\\end{array}\n
		\\end{aligned}\$\$\n\`\`\`\n"
	Markdown.parse(res1_data_str)
end

# ╔═╡ b0927717-c196-45c6-b2b1-5cd92365fa69
md"""
### Mean Annual Power -- Scaled Waves
"""

# ╔═╡ e2acbc1d-0f1b-4614-a39d-b034300e23c9
begin
	# WaveBot
	bemdir = "./data/bem_scaled_waves"
	scaled_thevenin = WaveBot.thevenin_equivalent(bemdir; freq_wave=n_freq)
	scaled_linearmodel = WaveBot.average_power(scaled_thevenin)
end

# ╔═╡ 64e7ab00-6f62-4381-a7d5-de6a09211062
begin
	_model = AutoEncoder.init_model(features, layers, x_new, x_new)
	# println("Average annual power: $(round(ustrip(fs_power_mean); digits=2))")

	loss_vec, mean_power_vec = 0, 0
	
	if isfile("data/models/trained_autoencoder.jld2")
		jldopen("data/models/trained_autoencoder.jld2", "r") do f
			global loss_vec, mean_power_vec
			loss_vec = f["loss_vec"]
			mean_power_vec = f["mean_power_vec"]
		end
		println("Loaded loss data from file!")
	else
		global loss_vec, mean_power_vec
		loss_vec = Vector{}()
		mean_power_vec = Vector{}()
		for i in 1:20
			_m_state = JLD2.load("data/models/model_checkpoint_new_$(i).jld2", "m_state");
			Flux.loadmodel!(_model, _m_state);

			_model_data = _model(normalized_data)
			_l = sqrt(Flux.Losses.mse(normalized_data, _model_data))
			append!(loss_vec, _l)
			_temp_spectra = AutoEncoder.unnormalize_data(transpose(_model_data), ustrip.(te_vector)*α, ustrip.(hs_vector)*α^2, n_freqs, x_new)
			wave = Waves.wave_amplitudes(_temp_spectra.*u"m^2/Hz", n_freq)
			append!(mean_power_vec, mean(scaled_linearmodel.(eachrow(wave))))
		end
		jldsave("./cache/lossData.jld2"; m_state, loss_vec, mean_power_vec)
	end
end

# ╔═╡ f2ebce6a-b5fb-459e-aa59-fa4ecaae867f
begin
	if isfile("./data/cache/spectras_scaled.jld2")
		jldopen("./data/cache/spectras_scaled.jld2", "r") do f
			global pm_spectra, ma_spectra, AE_spectra
			pm_spectra = f["pm_spectra"]
			ma_spectra = f["ma_spectra"]
			AE_spectra = f["AE_spectra"]
		end
		println("Loaded scaled bin data from file!")
	else
		@info "PM"
		spectra = Waves.pm_spectrum.(hs_vector*α^2, Waves.te_to_tp.(te_vector*α))
		pm_spectra = reduce(hcat, [s.(n_freq) for s in spectra])
		@info "AE"
		decoder_input = transpose(reduce(hcat, [encoded_spectra[1,:], encoded_spectra[2,:]]))
		decoder_output = transpose(decoder(decoder_input))
		AE_spectra = transpose(AutoEncoder.unnormalize_data(decoder_output, ustrip.(te_vector*α), ustrip.(hs_vector*α^2), n_freqs, x_new))
		@info "Mackay"
		m_funcs = Waves.mackay_spectrum.(mackay_data.hs*α^2, mackay_data.te*α, mackay_data.dtn, mackay_data.han2; mat_data)
		ma_spectra = reduce(hcat, [s.(n_freq) for s in m_funcs])
		JLD2.jldsave("./data/cache/spectras_scaled.jld2"; pm_spectra, ma_spectra, AE_spectra)
		println("Saved scaled bin data to file!")
	end
end

# ╔═╡ a8fb78db-ca0d-4e85-b7cd-570bd620abce
begin
	data_2d = [te_vector*α, hs_vector*α^2]
	data_4d_mackay = [mackay_data.te*α, mackay_data.hs*α^2, mackay_data.han2, mackay_data.dtn]
	data_4d_autoencoder = [te_vector*α, hs_vector*α^2, encoded_spectra[1,:], encoded_spectra[2,:]]
	
	bin_powers, ma_powers, ae_powers = 0, 0, 0
	if isfile("./data/cache/bins_scaled.jld2")
		jldopen("./data/cache/bins_scaled.jld2", "r") do f
			global bin_powers, ma_powers, ae_powers
			bin_powers = f["bin_powers"]
			ma_powers = f["ma_powers"]
			ae_powers = f["ae_powers"]
		end
		println("Loaded scaled bin data from file!")
	else
		@info "Hs-Te Bins"
		mats, bin_powers = ExtraFunctions.bin_power(n_freq, scaled_thevenin, data_2d, bins_big)
		@info "AE Bins"
		aux_bins2 = [ExtraFunctions.solve_4d_AE(n_freqs, scaled_thevenin, data_4d_autoencoder, bin, decoder, modified_orig_freqs/α, x_new) for bin in bins_small]
		ae_powers = [mat_power[2] for mat_power in aux_bins2]
		@info "Mackay Bins"
		aux_bins1 = [ExtraFunctions.solve_4d(n_freq, scaled_thevenin, data_4d_mackay, bin, mat_data) for bin in bins_small]
		ma_powers = [mat_power[2] for mat_power in aux_bins1]

		JLD2.jldsave("./data/cache/bins_scaled.jld2"; bin_powers, ma_powers, ae_powers)
		println("Saved scaled bin data to file!")
	end
end

# ╔═╡ 7812c0d6-5c50-4523-b66c-e1e104a01af2
begin
	power_yearly_scaled = Vector{typeof(1.0u"W")}()
	for spectrum in wave_spectra*α^5
	    wave = Waves.wave_amplitudes(spectrum, n_freq)
	    append!(power_yearly_scaled, mean(scaled_linearmodel.(eachrow(wave))))
	end
	power_mean_scaled = mean(power_yearly_scaled)
	println("Average scaled annual power: $power_mean_scaled")

	# mean distribution
	Norm_samples = Normal(ustrip(mean(power_yearly_scaled)), ustrip(std(power_yearly_scaled)))
	scaled_Norm_mean = Normal(ustrip(mean(power_yearly_scaled)), std(ustrip.(power_yearly_scaled))/√nyears)
	scaled_CI_min, scaled_CI_max = norm_ppf2(scaled_Norm_mean)
	scaled_CI_delta = ((scaled_CI_max - scaled_CI_min)/2)u"W"
	println("Average annual power: $(roundedUnitful(power_mean_scaled)) +/- $(roundedUnitful(CI_delta)) (95% CI))")
	
	# #Ground Truth
	# power_delta = CI_delta#108.11542037210347u"kg*m^2*s^-3"
	# CI_min, CI_max = (power_mean_scaled - power_delta), (power_mean_scaled + power_delta)
	# println("Average annual power: $(roundedUnitful(power_mean_scaled))")
	
	# mean - all together
	# waves_all = Waves.wave_amplitudes(filtered_spectra*α^5, freq)
	# power_mean_all = mean(scaled_linearmodel.(eachrow(waves_all)))
	# println("Average annual power (all): $(roundedUnitful(power_mean_all)) ")
	# pdiff = ((power_mean_all - power_mean_scaled) / power_mean_scaled)
	# println("percent diff: $(roundedUnitful(abs(pdiff*100)))%")
end

# ╔═╡ 8cd01eca-b54c-490f-942c-83e5ab752b34
let
	target_size = (line_width*0.47, line_width*0.47*0.8)
	figsize = target_size .* (inToPt/pt_per_unit)
	f1 = Figure(
		size = figsize
	)
	f2 = Figure(
		size = figsize
	)
	ax1 = CairoMakie.Axis(f1[1, 1],
		xlabel = "Epochs",
		ylabel = "Loss RMSE";
		ax_kwargs..., 
	)
	ax2 = CairoMakie.Axis(f2[1, 1], 
		xlabel = "Epochs",
		ylabel = "MAP Relative Error (%)"; 
		ax_kwargs...
	)
	# ylims!(ax1, 0.000000001, 0.015)
	# ylims!(ax2, 0, 7.5)
	lines!(ax1, 1:20,
		  loss_vec)
	lines!(ax2, 1:20,
		  ustrip.((mean_power_vec .- power_mean_scaled)./power_mean_scaled.*100))
	save("./data/figures/figure7a.pdf", f1; pt_per_unit)
	save("./data/figures/figure7b.pdf", f2; pt_per_unit)
	md"""
	#### Figure 7
	$(f1) $(f2)
	"""
end

# ╔═╡ 68bbb20f-4f70-4884-a4e3-74e8aa80293f
scaled_avg_p(spectra_iter) = mean([uconvert(u"W", scaled_linearmodel(Waves.wave_amplitudes(spectrum, n_freq))) for spectrum in spectra_iter])

# ╔═╡ 6de702d6-96ad-4b11-a6d9-b32522032712
begin
	scaled_original = filtered_spectra*α^5
	GT_p∞ = scaled_avg_p(eachrow(scaled_original))
	PM_p∞ = scaled_avg_p(eachcol(pm_spectra))
	M_p∞ = scaled_avg_p(eachcol(ma_spectra))
	AE_p∞ = scaled_avg_p(eachcol(AE_spectra.*u"m^2/Hz"))
	nothing
end

# ╔═╡ fb5838e9-4502-4391-88e2-5133f6f37d5d
mean(GT_p∞)

# ╔═╡ ef2b8046-b421-460a-8eb3-129e1140d0a5
power_mean_scaled

# ╔═╡ e951539e-d3ca-4f36-8f61-28b752a15633
let	
	baseline_bin_stats = [bins_small, bin_powers[begin:length(bins_small)]]
	baseline_args = [power_mean_scaled, PM_p∞, baseline_bin_stats, scaled_CI_min, scaled_CI_max]
	mackay_stats = [M_p∞, ma_powers]
	ae_stats = [AE_p∞, ae_powers]
	target_size=(line_width*0.7, line_width*0.6*0.75)
	# target_size=(line_width*0.95, line_width*0.95*0.7)
	f, _ = plot_limits_all(baseline_args, mackay_stats, ae_stats; 
							target_size, ylimits=(6.25,8.5))
	save("./data/figures/figure11.pdf", f; pt_per_unit)

	md"""
	#### Figure 11
	$(f)
	"""
end

# ╔═╡ 33f59655-6af0-465c-bbfa-b960e3322469
begin	
	
	# println("Powers vs GT Limit")
	# println("Ground Truth Limit: $(roundedUnitful(GT_p∞))")
	# println("Pierson-Moskowitz Limit: $(roundedUnitful(PM_p∞)), $(perc_error(GT_p∞, PM_p∞))")
	# println("Mackay Limit: $(roundedUnitful(M_p∞)), $(perc_error(GT_p∞, M_p∞))")
	# println("Autoencoder Limit: $(roundedUnitful(AE_p∞)), $(perc_error(GT_p∞, AE_p∞))")

	begin
	res2_data_str = "
		\`\`\`math\n
		\$\$\\begin{aligned}\n& \\text {Table 3}\\\\\n
		&\\begin{array}{cccc}\n
		\\hline \\hline \\text { Method } & \\text { Value } & \\text { Percent Error } \\\\\n\\hline\n"
	res2_data_str *= "\\text{Truth} & $(roundedUnitful(power_mean_scaled)) & - \\\\\n"
	res2_data_str *= "\\text{Pierson-Moskowitz} & $(roundedUnitful(PM_p∞)) & $(perc_error(power_mean_scaled, PM_p∞))\\\\\n"
	res2_data_str *= "\\text{Mackay} & $(roundedUnitful(M_p∞)) & $(perc_error(power_mean_scaled, M_p∞))\\\\\n"
	res2_data_str *= "\\text{Autoencoder} & $(roundedUnitful(AE_p∞)) & $(perc_error(power_mean_scaled, AE_p∞))\\\\\n"
	res2_data_str *= "
		\\hline\n
		\\end{array}\n
		\\end{aligned}\$\$\n\`\`\`\n"
	Markdown.parse(res2_data_str)
end
end

# ╔═╡ f7c4dd98-1b90-440f-8bf6-1d3234c392d7
function plot_ind_bin_errors(error_mat, idxs; ax_title="", target_size = (text_width*0.47, text_width*0.47*0.75), bins=16, colorrange=(-40, 40))
	figsize = target_size .* (inToPt/pt_per_unit)
	f = Figure(
		size = figsize
	)
	ax = CairoMakie.Axis(f[1,1],
		xlabel = L"Energy Period $(s)$",
		ylabel = L"Significant Waveheight $(m)$",
		title=ax_title;
		ax_kwargs...,
	)

	x, y = eachrow(idxs)
	xp, yp = x[begin:bins], y[begin:bins:end]

	if isnothing(colorrange)
		hm = heatmap!(ax, 
				  xp, yp, error_mat, 
				  colormap=:RdBu)
	else
		hm = heatmap!(ax, 
				  xp, yp, error_mat, 
				  colormap=:RdBu, colorrange=colorrange)
	
	end
	Colorbar(f[:, end+1], hm)
	if bins < 17
		errors = string.(roundedValue.(reduce(vcat, (eachrow(transpose(error_mat))))))
		errors = [e!="NaN" ? e :  "" for e in errors]
		text!(
		    x, y,
		    text = errors,
			align=(:center, :center),
			color=:black#, strokewidth=0.4, strokecolor=:white
		)
	end
	f, ax
end

# ╔═╡ bd219d3c-ab2d-4d15-83bc-fbb3dfa8efe9
function _plot_bin_errors(data, spectra_data; 
						  bins=16, α = 1/5, truth=power_mean_scaled,
						  linearmodel= WaveBot.average_power(WaveBot.thevenin_equivalent("./data/bem_scaled_waves"; freq_wave=n_freq)),
						  target_size = (line_width*0.7, line_width*0.7*0.75))
	
	(centers, edges), (weights, _) = ExtraFunctions.bins(data, [bins for _ in 1:length(data)])
	# println(edges)
	center_idx = reduce(hcat, [t...] for t in collect(centers))
	x_bins = collect(edges[1])
	y_bins = collect(edges[2])

	function get_bound_axis(val, bins)
		upper_idx = findfirst(val .< bins)
		return bins[upper_idx - 1:upper_idx]
	end
	h_bounds = get_bound_axis.(center_idx[1, :], [x_bins for _ in 1:bins^2])
	v_bounds = get_bound_axis.(center_idx[2, :], [y_bins for _ in 1:bins^2])

	filter(data, h_bound, v_bound) = (data[1] .>= h_bound[1] .&& data[1] .< h_bound[2]) .&& (data[2] .>= v_bound[1] .&& data[2] .< v_bound[2])
	bin_filter = [filter(data, h[1], v[1]) for (h, v) in zip(eachrow(h_bounds), eachrow(v_bounds))]
	
	power_list(spectra_iter) = [uconvert(u"W", linearmodel(Waves.wave_amplitudes(spectrum, n_freq))) for spectrum in spectra_iter]

	scaled_original = filtered_spectra*α^5
	GT_pl = power_list(eachrow(scaled_original))
	PM_pl = power_list(eachcol(spectra_data[1]))
	M_pl = power_list(eachcol(spectra_data[2]))
	AE_pl = power_list(eachcol(spectra_data[3].*u"m^2/Hz"))

	PM_error_mat = Array{Float64}(undef, bins, bins)
	Ma_error_mat = Array{Float64}(undef, bins, bins)
	AE_error_mat = Array{Float64}(undef, bins, bins)

	for i in 1:bins^2
		bf = bin_filter[i]
		# print(sum(bf), "\n")
		gt_power = mean(GT_pl[bf])
		PM_power = mean(PM_pl[bf])
		M_power = mean(M_pl[bf])
		AE_power = mean(AE_pl[bf])
		# print("bin $i from edges Te:$(round.(h_bounds[i]; digits=2)) and Hs:$(round.(v_bounds[i]; digits=2))\n")
		# print("gt_power: $(gt_power) \n")
		# print("PM_power: $(roundedUnitful(PM_power)) \tPerc Error: $(perc_error(gt_power, PM_power))\n")
		# print("M_power: $(roundedUnitful(M_power)) \tPerc Error: $(perc_error(gt_power, M_power))\n")
		# print("AE_power: $(roundedUnitful(AE_power)) \tPerc Error: $(perc_error(gt_power, AE_power))\n")
		# col = ((i - 1) % bins) + 1
		# row = ((i - 1) ÷ bins)
		# print(row, ",", col, "\n")
		# PM_error_mat[row, col] = perc_error_full(gt_power, PM_power)
		# Ma_error_mat[row, col] = perc_error_full(gt_power, M_power)
		# AE_error_mat[row, col] = perc_error_full(gt_power, AE_power)
		PM_error_mat[i] = perc_error_full(gt_power, PM_power)
		Ma_error_mat[i] = perc_error_full(gt_power, M_power)
		AE_error_mat[i] = perc_error_full(gt_power, AE_power)
	end
	# PM_error_mat, Ma_error_mat, AE_error_mat
	nanmean_arr(mat) = mean(mat[.!(isnan.(mat))])
	nansum_arr(mat) = sum(mat[.!(isnan.(mat))])

	# pm_total_error = roundedValue(mean(perc_error_full.(GT_pl, PM_pl)))
	# ma_total_error = roundedValue(mean(perc_error_full.(M_pl, PM_pl)))
	# ae_total_error = roundedValue(mean(perc_error_full.(AE_pl, PM_pl)))
	# pm_total_error = roundedValue(perc_error_full(mean(GT_pl), mean(PM_pl)))
	# ma_total_error = roundedValue(perc_error_full(mean(M_pl), mean(PM_pl)))
	# ae_total_error = roundedValue(perc_error_full(mean(AE_pl), mean(PM_pl)))
	
	# pm_total_error = roundedValue(nansum_arr(weights .* PM_error_mat))
	# ma_total_error = roundedValue(nansum_arr(weights .* Ma_error_mat))
	# ae_total_error = roundedValue(nansum_arr(weights .* AE_error_mat))

	pm_total_error = perc_error(truth, PM_p∞)
	ma_total_error = perc_error(truth, M_p∞)
	ae_total_error = perc_error(truth, AE_p∞)

	
	println("$(pm_label) Max Error: $(maximum((PM_error_mat[.!(isnan.(PM_error_mat))])))")
	println("$(ma_label) Max Error: $(maximum((Ma_error_mat[.!(isnan.(Ma_error_mat))])))")
	println("$(ae_label) Max Error: $(maximum((AE_error_mat[.!(isnan.(AE_error_mat))])))")
	println("$(pm_label) Min Error: $(minimum((PM_error_mat[.!(isnan.(PM_error_mat))])))")
	println("$(ma_label) Min Error: $(minimum((Ma_error_mat[.!(isnan.(Ma_error_mat))])))")
	println("$(ae_label) Min Error: $(minimum((AE_error_mat[.!(isnan.(AE_error_mat))])))")
	
	
	# # pm_total_error, ma_total_error, ae_total_error
	# f1, ax1 = plot_ind_bin_errors(PM_error_mat, center_idx; 
	# 							  ax_title="PM Bin Errors - Error: $(pm_total_error)", bins,
	# 							  target_size)
	# f2, ax2 = plot_ind_bin_errors(Ma_error_mat, center_idx; 
	# 							  ax_title="Mackay Bin Errors - Error: $(ma_total_error)", bins,
	# 							  target_size)
	# f3, ax3 = plot_ind_bin_errors(AE_error_mat, center_idx; 
	# 							  ax_title="Autoencoder Bin Errors - Error: $(ae_total_error)", bins,
	# 							  target_size)
	# target_size = (text_width*0.8, text_width*0.8*0.75)
	figsize = target_size .* (inToPt/pt_per_unit)
	f1, ax1 = plot_ind_bin_errors(ustrip.(PM_error_mat), center_idx; 
								  ax_title="$(pm_label) Bin Errors", bins, target_size)
	# f2, ax2 = plot_ind_bin_errors(Ma_error_mat, center_idx; 
	# 							  ax_title="Mackay Bin Errors", bins, target_size)
	# f3, ax3 = plot_ind_bin_errors(AE_error_mat, center_idx; 
	# 							  ax_title="Autoencoder Bin Errors", bins, target_size)
	target_size = (line_width*0.95, line_width*0.6)
	figsize = target_size .* (inToPt/pt_per_unit)
	f2 = Figure(
		size = figsize
	)
	ax2 = CairoMakie.Axis(f2[1,1],
		xlabel = L"Energy Period $(s)$",
		ylabel = L"Significiant Waveheight $(m)$",
		title="$(ma_label)\nBin Errors";
		ax_kwargs...,
	)
	ax3 = CairoMakie.Axis(f2[1,2],
		xlabel = L"Energy Period $(s)$",
		title="$(ae_label)\nBin Errors",
		yticksvisible=false, yticklabelsvisible=false;
		ax_kwargs...,
	)

	x, y = eachrow(center_idx)
	xp, yp = x[begin:bins], y[begin:bins:end]
	
	hm = heatmap!(ax2, 
				  xp, yp, ustrip.(Ma_error_mat), 
				  colormap=:RdBu, colorrange=(-40, 40))
	hm = heatmap!(ax3, 
				  xp, yp, AE_error_mat, 
				  colormap=:RdBu, colorrange=(-40, 40))
	
	# Colorbar(f2[:, end+1], hm)
	
	f1, f2
end

# ╔═╡ 9fd403c0-81ca-4bc4-b1b8-584f354fb953
function _plot_bin_contour_weights(data, spectra_data; 
						  bins=16, α = 1/5, truth=power_mean_scaled,
						  linearmodel= WaveBot.average_power(WaveBot.thevenin_equivalent("./data/bem_scaled_waves"; freq_wave=n_freq)),
						  target_size = (line_width*0.7, line_width*0.7*0.75), colorrange=(-2, 1.75))
	
	(centers, edges), (weights, _) = ExtraFunctions.bins(data, [bins for _ in 1:length(data)])
	# println(edges)
	center_idx = reduce(hcat, [t...] for t in collect(centers))
	x_bins = collect(edges[1])
	y_bins = collect(edges[2])

	function get_bound_axis(val, bins)
		upper_idx = findfirst(val .< bins)
		return bins[upper_idx - 1:upper_idx]
	end
	h_bounds = get_bound_axis.(center_idx[1, :], [x_bins for _ in 1:bins^2])
	v_bounds = get_bound_axis.(center_idx[2, :], [y_bins for _ in 1:bins^2])

	filter(data, h_bound, v_bound) = (data[1] .>= h_bound[1] .&& data[1] .< h_bound[2]) .&& (data[2] .>= v_bound[1] .&& data[2] .< v_bound[2])
	bin_filter = [filter(data, h[1], v[1]) for (h, v) in zip(eachrow(h_bounds), eachrow(v_bounds))]

	bin_count_mat = Array{Float64}(undef, bins, bins)
	
	for i in 1:bins^2
		bf = sum(bin_filter[i])
		bin_count_mat[i] = (bf != 0) ? bf/length(data[1])*100 : NaN
	end

	figsize = target_size .* (inToPt/pt_per_unit)
	f1, ax1 = plot_ind_bin_errors(bin_count_mat, center_idx; 
								 bins, target_size, colorrange=(0, 3.0))

	# Colorbar(f2[:, end+1], hm)
	
	f1
end

# ╔═╡ 0de11b26-83a9-4dbc-9b38-8f57c56dff06
# let
# 	α = 1/5
# 	width=0.6
# 	target_size = (text_width*width, text_width*width*0.75)
# 	te_hs_vector = [ustrip.(te_vector)*α, ustrip.(hs_vector*α^2)]
	
# 	f1, f2 = _plot_bin_errors(te_hs_vector, [pm_spectra, ma_spectra, AE_spectra]; bins=32, α, target_size)
# 	save("./data/figures/figure12a.pdf", f1; pt_per_unit)
# 	save("./data/figures/figure12b.pdf", f2; pt_per_unit)
	
# 	md"""
# 	#### Figure 12a
# 	$(f1)\
	
# 	#### Figure 12b
# 	$(f2)
# 	"""
# end

# ╔═╡ 69071870-90ce-4b5d-815d-f9998c611eef
# let
# 	α = 1/5
# 	width=0.4
# 	target_size = (text_width*width, text_width*width*2)
# 	te_hs_vector = [ustrip.(te_vector)*α, ustrip.(hs_vector*α^2)]

# 	f1 = _plot_bin_contour(te_hs_vector, [pm_spectra, ma_spectra, AE_spectra]; bins=32, α, target_size)
# 	save("./data/figures/figure12a.pdf", f1; pt_per_unit)
	
# 	md"""
# 	#### Figure 12a
# 	$(f1)\

# 	"""
# end

# ╔═╡ 596d8f55-75bd-4ba6-a70c-b3d6dd739457
# let
# 	α = 1/5
# 	width=0.4
# 	target_size = (text_width*width, text_width*width*2)
# 	te_hs_vector = [ustrip.(te_vector)*α, ustrip.(hs_vector*α^2)]

# 	f1 = _plot_bin_contour_prop(te_hs_vector, [pm_spectra, ma_spectra, AE_spectra]; bins=32, α, target_size)
# 	save("./data/figures/figure12b.pdf", f1; pt_per_unit)
	
# 	md"""
# 	#### Figure 12b
# 	$(f1)\

# 	"""
# end

# ╔═╡ 0fa16704-9e05-494c-b6db-d9f41a59326c
# let
# 	α = 1/5
# 	width=0.6
# 	target_size = (text_width*width, text_width*width*0.75)
# 	te_hs_vector = [ustrip.(te_vector)*α, ustrip.(hs_vector*α^2)]

# 	f1 = _plot_bin_contour_weights(te_hs_vector, [pm_spectra, ma_spectra, AE_spectra]; bins=32, α, target_size)
# 	save("./data/figures/figure12c.pdf", f1; pt_per_unit)
	
# 	md"""
# 	#### Figure 12c
# 	$(f1)\

# 	"""
# end

# ╔═╡ d1980d03-67d2-483c-98c3-a68c1f0fd15b
begin
	power_list(spectra_iter) = [uconvert(u"W", scaled_linearmodel(Waves.wave_amplitudes(spectrum, n_freq))) for spectrum in spectra_iter]
end

# ╔═╡ 963b3d5a-3593-40ca-9b8a-18380f32328f
function _plot_bin_errors_testing_left(data, spectra_data; 
						  bins=16, α = 1/5, truth=power_mean_scaled,
						  linearmodel= WaveBot.average_power(WaveBot.thevenin_equivalent("./data/bem_scaled_waves"; freq_wave=n_freq)),
						  target_size = (line_width*0.1, line_width*0.1*0.75))
	
	scaled_original = filtered_spectra*α^5
	GT_pl = power_list(eachrow(scaled_original))
	PM_pl = power_list(eachcol(spectra_data[1]))
	M_pl = power_list(eachcol(spectra_data[2]))
	AE_pl = power_list(eachcol(spectra_data[3].*u"m^2/Hz"))

	PM_left_err = (PM_pl .- GT_pl) ./ GT_pl * 100
	M_left_err = (M_pl .- GT_pl) ./ GT_pl * 100
	AE_left_err = (AE_pl .- GT_pl) ./ GT_pl * 100
	
	# PM_right_err = (PM_pl .- GT_pl) / power_mean_scaled
	# M_right_err = (M_pl .- GT_pl) / power_mean_scaled
	# AE_right_err = (AE_pl .- GT_pl) / power_mean_scaled

	xs, ys = data
	figsize = target_size .* (inToPt/pt_per_unit)
	
	f = Figure(
		size = figsize
	)
	
	ax1 = CairoMakie.Axis(f[1,1],
		# xlabel = L"Energy Period $(s)$",
		# ylabel = L"Significant Waveheight $(m)$",
		title="$(pm_label)",
		xticksvisible=false, xticklabelsvisible=false;
		ax_kwargs...,
	)

	ax2 = CairoMakie.Axis(f[2,1],
		# xlabel = L"Energy Period $(s)$",
		ylabel = L"Significant Waveheight $(m)$",
		title="$(ma_label)",
		xticksvisible=false, xticklabelsvisible=false;
		ax_kwargs...,
	)

	ax3 = CairoMakie.Axis(f[3,1],
		xlabel = L"Energy Period $(s)$",
		# ylabel = L"Significant Waveheight $(m)$",
		title="$(ae_label)";
		ax_kwargs...,
	)

	alpha=0.5
	color_max = maximum([maximum(abs.(PM_left_err)), maximum(abs.(M_left_err)), maximum(abs.(AE_left_err))])

	println("PM: $(minimum(PM_left_err)) - $(maximum(PM_left_err))")
	println("M: $(minimum(M_left_err)) - $(maximum(M_left_err))")
	println("AE: $(minimum(AE_left_err)) - $(maximum(AE_left_err))")
	# println()
	# println(maximum(M_err))
	# println(maximum(AE_err))

	# println(minimum(PM_left_err))
	# println(minimum(M_err))
	# println(minimum(AE_err))
	
	sort_idx = sortperm(ustrip.(abs.(PM_left_err)))
	pm_xs = xs[sort_idx]
	pm_ys = ys[sort_idx]
	PM_left_err = PM_left_err[sort_idx]
	
	hm = scatter!(ax1, 
		pm_xs, pm_ys, color=ustrip.(PM_left_err), alpha=alpha,
		colormap=:RdBu, rasterize=2, colorrange=(-color_max, color_max))

	sort_idx = sortperm(ustrip.(abs.(M_left_err)))
	m_xs = xs[sort_idx]
	m_ys = ys[sort_idx]
	M_left_err = M_left_err[sort_idx]
	
	_ = scatter!(ax2, 
		m_xs, m_ys, color=ustrip.(M_left_err), alpha=alpha,
		colormap=:RdBu, rasterize=2, colorrange=(-color_max, color_max))

	sort_idx = sortperm(ustrip.(abs.(AE_left_err)))
	ae_xs = xs[sort_idx]
	ae_ys = ys[sort_idx]
	AE_left_err = AE_left_err[sort_idx]
	
	_ = scatter!(ax3, 
		ae_xs, ae_ys, color=ustrip.(AE_left_err), alpha=alpha,
		colormap=:RdBu, rasterize=2, colorrange=(-color_max, color_max))
	
	Colorbar(f[end+1, :], hm, vertical = false, flipaxis=false, label="Relative Error (%)")

	rowgap!(f.layout, 10)
	
	f
end

# ╔═╡ bb26f5b5-1dfe-4787-9c19-5432faf33df3
function _plot_bin_errors_testing_right(data, spectra_data; 
						  bins=16, α = 1/5, truth=power_mean_scaled,
						  linearmodel= WaveBot.average_power(WaveBot.thevenin_equivalent("./data/bem_scaled_waves"; freq_wave=n_freq)),
						  target_size = (line_width*0.1, line_width*0.1*0.75))
	
	scaled_original = filtered_spectra*α^5
	GT_pl = power_list(eachrow(scaled_original))
	PM_pl = power_list(eachcol(spectra_data[1]))
	M_pl = power_list(eachcol(spectra_data[2]))
	AE_pl = power_list(eachcol(spectra_data[3].*u"m^2/Hz"))
	
	PM_right_err = ustrip.(PM_pl .- GT_pl) * 3600 / 1000
	M_right_err = ustrip.(M_pl .- GT_pl) * 3600 / 1000
	AE_right_err = ustrip.(AE_pl .- GT_pl) * 3600 / 1000

	# GT_joules = ustrip(mean(GT_pl) * length(PM_right_err) * 3600)
	# println("GT Joules: ", GT_joules)

	# PM_joules = ustrip(mean(PM_pl) * length(PM_right_err) * 3600)
	# println("PM Joules: ", PM_joules)

	# println("Diff Joules: ", PM_joules - GT_joules)

	# println("Sum of PM errors: ", sum(PM_right_err))

	# println(length(PM_right_err))

	# println(11 * 365.25 * 24)
	
	xs, ys = data
	figsize = target_size .* (inToPt/pt_per_unit)
	
	f = Figure(
		size = figsize
	)
	
	ax1 = CairoMakie.Axis(f[1,1],
		# xlabel = L"Energy Period $(s)$",
		# ylabel = L"Significant Waveheight $(m)$",
		title="$(pm_label)",
		xticksvisible=false, xticklabelsvisible=false;
		ax_kwargs...,
	)

	ax2 = CairoMakie.Axis(f[2,1],
		# xlabel = L"Energy Period $(s)$",
		ylabel = L"Significant Waveheight $(m)$",
		title="$(ma_label)",
		xticksvisible=false, xticklabelsvisible=false;
		ax_kwargs...,
	)

	ax3 = CairoMakie.Axis(f[3,1],
		xlabel = L"Energy Period $(s)$",
		# ylabel = L"Significant Waveheight $(m)$",
		title="$(ae_label)";
		ax_kwargs...,
	)

	alpha=0.5
	color_max = maximum([maximum(abs.(PM_right_err)), maximum(abs.(M_right_err)), maximum(abs.(AE_right_err))])

	println("PM: $(minimum(PM_right_err)) - $(maximum(PM_right_err))")
	println("M: $(minimum(M_right_err)) - $(maximum(M_right_err))")
	println("AE: $(minimum(AE_right_err)) - $(maximum(AE_right_err))")
	# println(maximum(PM_right_err))
	# println(maximum(M_err))
	# println(maximum(AE_err))

	# println(minimum(PM_right_err))
	# println(minimum(M_err))
	# println(minimum(AE_err))
	
	sort_idx = sortperm(ustrip.(abs.(PM_right_err)))
	pm_xs = xs[sort_idx]
	pm_ys = ys[sort_idx]
	PM_right_err = PM_right_err[sort_idx]
	
	hm = scatter!(ax1, 
		pm_xs, pm_ys, color=ustrip.(PM_right_err), alpha=alpha,
		colormap=:RdBu, rasterize=2, colorrange=(-color_max, color_max))

	sort_idx = sortperm(ustrip.(abs.(M_right_err)))
	m_xs = xs[sort_idx]
	m_ys = ys[sort_idx]
	M_right_err = M_right_err[sort_idx]
	
	_ = scatter!(ax2, 
		m_xs, m_ys, color=ustrip.(M_right_err), alpha=alpha,
		colormap=:RdBu, rasterize=2, colorrange=(-color_max, color_max))

	sort_idx = sortperm(ustrip.(abs.(AE_right_err)))
	ae_xs = xs[sort_idx]
	ae_ys = ys[sort_idx]
	AE_right_err = AE_right_err[sort_idx]
	
	_ = scatter!(ax3, 
		ae_xs, ae_ys, color=ustrip.(AE_right_err), alpha=alpha,
		colormap=:RdBu, rasterize=2, colorrange=(-color_max, color_max))
	
	Colorbar(f[end+1, :], hm, vertical = false, flipaxis=false, label="Error (kJ)")

	rowgap!(f.layout, 10)
	
	f
end

# ╔═╡ 30b14504-4eb9-4d1b-8ac7-f6fcbf7d387e
let
	α = 1/5
	width=0.45
	target_size = (line_width*width, line_width*width*2.5)
	te_hs_vector = [ustrip.(te_vector)*α, ustrip.(hs_vector*α^2)]
	
	f1 = _plot_bin_errors_testing_left(te_hs_vector, [pm_spectra, ma_spectra, AE_spectra]; bins=32, α, target_size)
	f2 = _plot_bin_errors_testing_right(te_hs_vector, [pm_spectra, ma_spectra, AE_spectra]; bins=32, α, target_size)
	save("./data/figures/figure12a.pdf", f1; pt_per_unit)
	save("./data/figures/figure12b.pdf", f2; pt_per_unit)
	
	md"""
	#### Figure 12a
	$(f1)
	
	#### Figure 12b
	$(f2)
	"""
end

# ╔═╡ d491f4b8-6b3e-4866-851d-9b93330ef669
function _plot_bin_contour(data, spectra_data; 
						  bins=16, α = 1/5, truth=power_mean_scaled,
						  linearmodel= WaveBot.average_power(WaveBot.thevenin_equivalent("./data/bem_scaled_waves"; freq_wave=n_freq)),
						  target_size = (line_width*0.7, line_width*0.7*0.75), colorrange=(-0.75, 0.75))
	
	scaled_original = filtered_spectra*α^5
	GT_pl = power_list(eachrow(scaled_original))
	PM_pl = power_list(eachcol(spectra_data[1]))
	M_pl = power_list(eachcol(spectra_data[2]))
	AE_pl = power_list(eachcol(spectra_data[3].*u"m^2/Hz"))

	PM_err = @. (PM_pl - GT_pl) / GT_pl
	M_err = @. (M_pl - GT_pl) / GT_pl
	AE_err = @. (AE_pl - GT_pl) / GT_pl

	println("$(pm_label) Max PM Error: $(maximum(PM_err))")
	println("$(ma_label) Max Mackay Error: $(maximum(M_err))")
	println("$(ae_label) Max Autoencoder Error: $(maximum(AE_err))")
	println("$(pm_label) Min PM Error: $(minimum(PM_err))")
	println("$(ma_label) Min Mackay Error: $(minimum(M_err))")
	println("$(ae_label) Min Autoencoder Error: $(minimum(AE_err))")

	xs, ys = data
	figsize = target_size .* (inToPt/pt_per_unit)
	
	f = Figure(
		size = figsize
	)
	
	ax1 = CairoMakie.Axis(f[1,1],
		# xlabel = L"Energy Period $(s)$",
		ylabel = L"Significant Waveheight $(m)$",
		title="$(pm_label)",
		xticksvisible=false, xticklabelsvisible=false;
		ax_kwargs...,
	)
	
	ax2 = CairoMakie.Axis(f[2,1],
		ylabel = L"Significiant Waveheight $(m)$",
		title="$(ma_label)",
		xticksvisible=false, xticklabelsvisible=false;
		ax_kwargs...,
	)
	ax3 = CairoMakie.Axis(f[3,1],
		xlabel = L"Energy Period $(s)$",
		ylabel = L"Significiant Waveheight $(m)$",
		title="$(ae_label)";
		ax_kwargs...,
	)

	hm = tricontourf!(ax1, 
		xs, ys, ustrip.(PM_err), 
		colormap=:RdBu, levels=colorrange[1]:0.1:colorrange[2], rasterize=2)
	hm = tricontourf!(ax2, 
				  xs, ys, ustrip.(M_err), 
				  colormap=:RdBu, levels=colorrange[1]:0.1:colorrange[2], rasterize=2)
	hm = tricontourf!(ax3, 
				  xs, ys, ustrip.(AE_err), 
				  colormap=:RdBu, levels=colorrange[1]:0.1:colorrange[2], rasterize=2)
	
	Colorbar(f[:, end+1], hm)
	
	f
end

# ╔═╡ 99068c0d-34ea-4972-9611-52a39d17e29a
function _plot_bin_contour_prop(data, spectra_data; 
						  bins=16, α = 1/5, truth=power_mean_scaled,
						  linearmodel= WaveBot.average_power(WaveBot.thevenin_equivalent("./data/bem_scaled_waves"; freq_wave=n_freq)),
						  target_size = (line_width*0.7, line_width*0.7*0.75), colorrange=(-2, 1.75))
	
	scaled_original = filtered_spectra*α^5
	GT_pl = power_list(eachrow(scaled_original))
	PM_pl = power_list(eachcol(spectra_data[1]))
	M_pl = power_list(eachcol(spectra_data[2]))
	AE_pl = power_list(eachcol(spectra_data[3].*u"m^2/Hz"))

	PM_err = @. (PM_pl - GT_pl) / power_mean_scaled
	M_err = @. (M_pl - GT_pl) / power_mean_scaled
	AE_err = @. (AE_pl - GT_pl) / power_mean_scaled

	println("$(pm_label) Max PM Error: $(maximum(PM_err))")
	println("$(ma_label) Max Mackay Error: $(maximum(M_err))")
	println("$(ae_label) Max Autoencoder Error: $(maximum(AE_err))")
	println("$(pm_label) Min PM Error: $(minimum(PM_err))")
	println("$(ma_label) Min Mackay Error: $(minimum(M_err))")
	println("$(ae_label) Min Autoencoder Error: $(minimum(AE_err))")
	
	xs, ys = data
	figsize = target_size .* (inToPt/pt_per_unit)
	
	f = Figure(
		size = figsize
	)
	
	ax1 = CairoMakie.Axis(f[1,1],
		# xlabel = L"Energy Period $(s)$",
		ylabel = L"Significant Waveheight $(m)$",
		title="$(pm_label)",
		xticksvisible=false, xticklabelsvisible=false;
		ax_kwargs...,
	)
	
	ax2 = CairoMakie.Axis(f[2,1],
		ylabel = L"Significiant Waveheight $(m)$",
		title="$(ma_label)",
		xticksvisible=false, xticklabelsvisible=false;
		ax_kwargs...,
	)
	ax3 = CairoMakie.Axis(f[3,1],
		xlabel = L"Energy Period $(s)$",
		ylabel = L"Significiant Waveheight $(m)$",
		title="$(ae_label)";
		ax_kwargs...,
	)

	hm = tricontourf!(ax1, 
		xs, ys, ustrip.(PM_err), 
		colormap=:RdBu, levels=colorrange[1]:0.1:colorrange[2], rasterize=2)
	hm = tricontourf!(ax2, 
				  xs, ys, ustrip.(M_err), 
				  colormap=:RdBu, levels=colorrange[1]:0.1:colorrange[2], rasterize=2)
	hm = tricontourf!(ax3, 
				  xs, ys, ustrip.(AE_err), 
				  colormap=:RdBu, levels=colorrange[1]:0.1:colorrange[2], rasterize=2)
	
	Colorbar(f[:, end+1], hm)
	
	f
end

# ╔═╡ f7fb07ff-ff34-4502-aa8c-379156a6570c
# let
	

# 	width = 0.7
# 	target_size = (line_width*width, line_width*width*0.75)
# 	figsize = target_size .* (inToPt/pt_per_unit)
# 	f = Figure(
# 		size = figsize,
		
# 	)
# 	ax = CairoMakie.Axis(f[1,1],
# 		ylabel="Relative Occurence";
# 		ax_kwargs...,
# 	)
# 	xlims!(-5,5)
# 	# ylims!(1, 100000)
# 	bins=200

# 	# quickIntegral(Δx, y) = sum((y[begin: end-1] .+ y[begin+1:end] ./ 2) .* Δx)
# 	PM_hist = normalize(fit(Histogram, PM_errors, nbins=bins);mode=:none)
# 	y1 = PM_hist.weights
# 	Ma_hist = normalize(fit(Histogram, Ma_errors, nbins=bins);mode=:none)
# 	y2 = Ma_hist.weights
# 	AE_hist = normalize(fit(Histogram, AE_errors, nbins=bins);mode=:none)
# 	y3 = AE_hist.weights

# 	x1 = (PM_hist.edges[1] .+ step(PM_hist.edges[1])/2)[begin:end-1]
# 	x2 = (Ma_hist.edges[1] .+ step(Ma_hist.edges[1])/2)[begin:end-1]
# 	x3 = (AE_hist.edges[1] .+ step(AE_hist.edges[1])/2)[begin:end-1]

# 	# println(length(x1))
# 	# println(length(x2))
# 	# println(length(x3))
# 	# println("PM:", sum(x1 .* y1))
# 	# println("Ma:", sum(x2 .* y2))
# 	# println("AE:", sum(x3 .* y3))

# 	linestyles = [(:solid), (:dashdot, :dense), (:dash, :dense)]
# 	colors = [pm_color, ma_color, ae_color]
# 	N = sum(y1)
# 	lines!(ax,
# 		x1, y1/N, label=pm_label,
# 		linestyle=linestyles[1], color=colors[1],
# 	)
# 	lines!(ax,
# 		x2, y2/N, label=ma_label,
# 		   linestyle=linestyles[2], color=colors[2],
# 	)
# 	lines!(ax,
# 		x3, y3/N, label=ae_label, color=colors[3],
# 		linestyle=linestyles[3]
# 	)
# 	axislegend(ax, 
# 		position=:rt, halign=:right, valign=:top, 
# 		framewidth=0.5, rowgap = -8, patchlabelgap = 1,
# 		padding=2
# 	)
# 	save("./data/figures/figure13.pdf", f; pt_per_unit)
# 	md"""
# 	#### Figure 13
# 	$f
# 	"""
# end

# ╔═╡ f6015f97-2162-45e0-9fd4-3b946a2e2bc5
begin
	GT_pl = power_list(eachrow(scaled_original))
	PM_pl = power_list(eachcol(pm_spectra))
	M_pl = power_list(eachcol(ma_spectra))
	AE_pl = power_list(eachcol(AE_spectra.*u"m^2/Hz"))

	# PM_errors = abs_error.(GT_pl, PM_pl)
	PM_left_errors = ((PM_pl - GT_pl) ./ GT_pl)*100
	Ma_left_errors = ((M_pl - GT_pl) ./ GT_pl)*100
	AE_left_errors = ((AE_pl - GT_pl) ./ GT_pl)*100

	PM_right_errors = (PM_pl .- GT_pl) * 3600 / 1000
	Ma_right_errors = (M_pl .- GT_pl) * 3600 / 1000
	AE_right_errors = (AE_pl .- GT_pl) * 3600 / 1000
end

# ╔═╡ 0f3b2065-832a-4abf-bc35-4fb4a3ba4a93
let

	width = 0.7
	target_size = (line_width*width, line_width*width*0.75)
	figsize = target_size .* (inToPt/pt_per_unit)
	f = Figure(
		size = figsize,
		
	)
	ax = CairoMakie.Axis(f[1,1],
		ylabel="Relative Occurence",
		xlabel="Percent Error (%)",
		yscale=log10;				 
		ax_kwargs...,
	)
	# xlims!(-10,10)
	ylims!(1e-5, 1)
	bins=200

	# quickIntegral(Δx, y) = sum((y[begin: end-1] .+ y[begin+1:end] ./ 2) .* Δx)
	PM_hist = normalize(fit(Histogram, PM_left_errors, nbins=bins);mode=:none)
	y1 = PM_hist.weights
	Ma_hist = normalize(fit(Histogram, Ma_left_errors, nbins=bins);mode=:none)
	y2 = Ma_hist.weights
	AE_hist = normalize(fit(Histogram, AE_left_errors, nbins=bins);mode=:none)
	y3 = AE_hist.weights

	x1 = (PM_hist.edges[1] .+ step(PM_hist.edges[1])/2)[begin:end-1]
	x2 = (Ma_hist.edges[1] .+ step(Ma_hist.edges[1])/2)[begin:end-1]
	x3 = (AE_hist.edges[1] .+ step(AE_hist.edges[1])/2)[begin:end-1]

	# println(length(x1))
	# println(length(x2))
	# println(length(x3))
	# println("PM:", sum(x1 .* y1))
	# println("Ma:", sum(x2 .* y2))
	# println("AE:", sum(x3 .* y3))

	vlines!(ax, 0, color=:black, linewidth=0.5)
	linestyles = [(:solid), (:dashdot, :dense), (:dash, :dense)]
	colors = [pm_color, ma_color, ae_color]
	N = sum(y1)
	lines!(ax,
		x1, y1/N, label=pm_label,
		linestyle=linestyles[1], color=colors[1],
	)
	N = sum(y2)
	lines!(ax,
		x2, y2/N, label=ma_label,
		   linestyle=linestyles[2], color=colors[2],
	)
	N = sum(y3)
	lines!(ax,
		x3, y3/N, label=ae_label, color=colors[3],
		linestyle=linestyles[3]
	)
	axislegend(ax, 
		position=:rt, halign=:right, valign=:top, 
		framewidth=0.5, rowgap = -8, patchlabelgap = 1,
		padding=2
	)
	save("./data/figures/figure13.pdf", f; pt_per_unit)
	md"""
	#### Figure 13
	$f
	"""
end

# ╔═╡ 0f893603-8129-46fb-961a-f200a0d7f8b3
let

	width = 0.7
	target_size = (line_width*width, line_width*width*0.75)
	figsize = target_size .* (inToPt/pt_per_unit)
	f = Figure(
		size = figsize,
		
	)
	ax = CairoMakie.Axis(f[1,1],
		ylabel="Relative Occurence",
		xlabel="Error (kJ)",
		yscale=log10;				 
		ax_kwargs...,
	)
	xlims!(-50,50)
	ylims!(1e-5, 1)
	bins=200

	# quickIntegral(Δx, y) = sum((y[begin: end-1] .+ y[begin+1:end] ./ 2) .* Δx)
	PM_hist = normalize(fit(Histogram, ustrip.(PM_right_errors), nbins=bins);mode=:none)
	y1 = PM_hist.weights
	Ma_hist = normalize(fit(Histogram, ustrip.(Ma_right_errors), nbins=bins);mode=:none)
	y2 = Ma_hist.weights
	AE_hist = normalize(fit(Histogram, ustrip.(AE_right_errors), nbins=bins);mode=:none)
	y3 = AE_hist.weights

	x1 = (PM_hist.edges[1] .+ step(PM_hist.edges[1])/2)[begin:end-1]
	x2 = (Ma_hist.edges[1] .+ step(Ma_hist.edges[1])/2)[begin:end-1]
	x3 = (AE_hist.edges[1] .+ step(AE_hist.edges[1])/2)[begin:end-1]

	# println(length(x1))
	# println(length(x2))
	# println(length(x3))
	# println("PM:", sum(x1 .* y1))
	# println("Ma:", sum(x2 .* y2))
	# println("AE:", sum(x3 .* y3))

	vlines!(ax, 0, color=:black, linewidth=0.5)
	linestyles = [(:solid), (:dashdot, :dense), (:dash, :dense)]
	colors = [pm_color, ma_color, ae_color]
	N = sum(y1)
	lines!(ax,
		x1, y1/N, label=pm_label,
		linestyle=linestyles[1], color=colors[1],
	)
	N = sum(y2)
	lines!(ax,
		x2, y2/N, label=ma_label,
		   linestyle=linestyles[2], color=colors[2],
	)
	N = sum(y3)
	lines!(ax,
		x3, y3/N, label=ae_label, color=colors[3],
		linestyle=linestyles[3]
	)
	axislegend(ax, 
		position=:rt, halign=:right, valign=:top, 
		framewidth=0.5, rowgap = -8, patchlabelgap = 1,
		padding=2
	)
	save("./data/figures/figure14.pdf", f; pt_per_unit)
	md"""
	#### Figure 14
	$f
	"""
end

# ╔═╡ Cell order:
# ╟─b0e4b31d-965f-4d21-98d0-e411dad86052
# ╟─48651dbe-4218-11f0-3f20-393151cdf520
# ╟─a957144e-22ad-4f88-943d-17cad7087b66
# ╟─707f2b47-c200-408c-8b23-984ca0c18dc6
# ╟─8ca16461-40af-4b1e-a24c-5b01d23e18a3
# ╟─ccf45dd8-debc-40cb-b31d-576a52f9cf54
# ╠═6c021cbd-ca5d-46ac-9b9f-ea9479cdb749
# ╟─6f89c696-fefc-4804-a130-fdea2bbccf86
# ╟─3121e9e0-f667-4d77-9f0f-a59f4a3ed2a0
# ╟─56bb2601-450f-4f2e-8293-69c2fba111d1
# ╟─a009dda3-ef87-4192-b049-8b418af94b49
# ╟─889a8863-fbe7-414e-ae4c-a6bef388d116
# ╟─92e3352f-94e7-4d68-ab23-6da8dc5e74c4
# ╟─c47fe519-4613-491e-87a1-255bee058027
# ╟─783aeecc-b04b-4903-b7c0-1e20442918d6
# ╟─24423cf9-7caa-4f28-837e-c7781be61d37
# ╟─e9ea5c05-6ed6-4c1e-80f1-27187965540a
# ╟─d08e79ef-4c38-437a-8c16-154f3b466215
# ╟─cef36d47-771e-40e6-a104-97213fd67ce5
# ╟─15a73043-d988-434f-b07c-ed7c482caa1d
# ╟─427cf560-b282-43b2-bf2b-87b1f443c2c1
# ╟─5e6ea2a8-2043-4655-8df8-60d89fb3f34c
# ╟─0e839f6c-9dc7-4dab-9922-9eb505b6d597
# ╟─8679f1fc-7a29-4fb5-8272-4d75c1daae85
# ╠═43e5208c-ca93-41b7-ab83-4fde49f9f46a
# ╠═0def8a11-3054-4c09-ab55-e8823753330b
# ╟─6059d13f-783d-4aa4-b13c-8eb189420a7f
# ╟─ca630b45-a5a1-4565-aa0a-9f928085ea5c
# ╟─ace2aa30-e927-4a78-ba39-d6a1002aa751
# ╟─3df3733c-fc52-467c-a418-99ecc46f5538
# ╟─08f935b2-0aa1-4990-9f53-bc3f058ba6d5
# ╟─a1073f30-312c-4627-97d1-43aa00da4e31
# ╟─011c0dda-4996-4956-bedf-811043876bc8
# ╟─ee0b7559-ea07-45b3-b1a9-0c7b912f3379
# ╟─3136fd1a-2533-4ba6-a02b-61386e82376b
# ╟─dbd9685e-17ff-4223-9f5f-0b3cf6777f0c
# ╟─a6bd37ca-ee05-40f7-adf0-951a9010ac5a
# ╟─a9a28f60-e571-4967-b25f-610f777fea6d
# ╟─884c37c2-4caa-4f63-a03c-30a1242383df
# ╟─800de494-fd21-4c78-b617-12c348b4e27f
# ╟─8feca466-305f-477d-a654-3a9358279402
# ╟─b3d56270-b7fc-4ff7-bef4-7fdc0cec02f6
# ╟─48d3ce5b-90e7-4e2d-8125-1e4498819ba9
# ╟─64e7ab00-6f62-4381-a7d5-de6a09211062
# ╠═8cd01eca-b54c-490f-942c-83e5ab752b34
# ╟─ac9630f9-f455-4620-ba0d-2a33fbc1fa0b
# ╟─6aa09a4f-f4e9-43a8-8348-336f70954dc6
# ╟─a0ab4a43-3723-4fbf-a209-f97786ddabc8
# ╠═22542d59-202d-435d-8587-4199aa75bb27
# ╟─45455ee7-d396-4ace-aaf4-af9f0c68e3fd
# ╟─fd1771fc-6bfa-49dd-a07f-4b200b3d9d6c
# ╟─2e058679-8e47-48a8-9b3a-cd1021bcc13a
# ╟─fc0ba65c-15f6-44ee-a1b0-b431a75b2b59
# ╟─675f1c6a-eb9c-4020-b2d9-787e943d0379
# ╟─2100390f-0ef9-4e19-bb3d-86e9331cf187
# ╟─1b73a718-93a9-47ea-bdb4-53e463653a50
# ╟─db263b34-2bb5-47a4-8638-a44fa0d44dc5
# ╟─5fde7fea-e0c9-40e6-9de8-893db91b403a
# ╟─3b1764e2-7f5b-4bd4-9c27-e39725642c39
# ╟─b0927717-c196-45c6-b2b1-5cd92365fa69
# ╟─e2acbc1d-0f1b-4614-a39d-b034300e23c9
# ╟─f2ebce6a-b5fb-459e-aa59-fa4ecaae867f
# ╟─a8fb78db-ca0d-4e85-b7cd-570bd620abce
# ╟─7812c0d6-5c50-4523-b66c-e1e104a01af2
# ╟─68bbb20f-4f70-4884-a4e3-74e8aa80293f
# ╠═6de702d6-96ad-4b11-a6d9-b32522032712
# ╠═fb5838e9-4502-4391-88e2-5133f6f37d5d
# ╠═ef2b8046-b421-460a-8eb3-129e1140d0a5
# ╟─e951539e-d3ca-4f36-8f61-28b752a15633
# ╟─33f59655-6af0-465c-bbfa-b960e3322469
# ╠═963b3d5a-3593-40ca-9b8a-18380f32328f
# ╠═bb26f5b5-1dfe-4787-9c19-5432faf33df3
# ╠═30b14504-4eb9-4d1b-8ac7-f6fcbf7d387e
# ╟─f7c4dd98-1b90-440f-8bf6-1d3234c392d7
# ╟─bd219d3c-ab2d-4d15-83bc-fbb3dfa8efe9
# ╟─d491f4b8-6b3e-4866-851d-9b93330ef669
# ╟─99068c0d-34ea-4972-9611-52a39d17e29a
# ╟─9fd403c0-81ca-4bc4-b1b8-584f354fb953
# ╟─0de11b26-83a9-4dbc-9b38-8f57c56dff06
# ╟─69071870-90ce-4b5d-815d-f9998c611eef
# ╟─596d8f55-75bd-4ba6-a70c-b3d6dd739457
# ╟─0fa16704-9e05-494c-b6db-d9f41a59326c
# ╟─d1980d03-67d2-483c-98c3-a68c1f0fd15b
# ╟─f7fb07ff-ff34-4502-aa8c-379156a6570c
# ╠═f6015f97-2162-45e0-9fd4-3b946a2e2bc5
# ╟─0f3b2065-832a-4abf-bc35-4fb4a3ba4a93
# ╟─0f893603-8129-46fb-961a-f200a0d7f8b3
