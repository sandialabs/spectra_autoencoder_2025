module AutoEncoder

using Flux, Unitful, Interpolations, JLD2
# using Optimisers
include("Waves.jl")

function my_linear_interpolation(x, x_vals, y_vals)
    # Ensure that x is within the range of x_vals
    if !(x_vals[begin] <= x) || !(x <= x_vals[end]) 
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
function mat_moment(data, freq, moment)
	delta_f = freq[begin+1:end] .- freq[begin:end-1]
	a1 = data .* (freq.^moment)
	a2 = @. (a1[begin+1:end, :] + a1[begin:end-1, :])/2 * delta_f
	sum(a2; dims=1)
end
function mat_energy_period(data, freq)
	mn1 = mat_moment(data, freq, -1)
	m0 = mat_moment(data, freq, 0)
	mn1 ./ m0
end
function mat_significant_wheight(data, freq)
	m0 = mat_moment(data, freq, 0)
	@. 4sqrt(m0)
end
function new_modified_softmax(x_arr::AbstractMatrix; orig_freq::AbstractVector = modified_orig_freqs, new_freqs::AbstractVector=x_new)
	te_all = mat_energy_period(x_arr, orig_freq)
    hs_all = mat_significant_wheight(x_arr, orig_freq)
	new_freqs_all = orig_freq .* te_all
	
    scaling = @. hs_all^2 * te_all
    scaled_vals = x_arr ./ scaling

	all_new_freqs = eachcol(new_freqs_all)
	new_vals = reduce(hcat, [[my_linear_interpolation(x, A_x, scaled_vals[:, i]) for x in new_freqs] for (i, A_x) in enumerate(all_new_freqs)])
	# all_extp = [extrapolate(interpolate((A_x,), scaled_vals[:, i], Gridded(Linear())), 0) for (i, A_x) in enumerate(all_new_freqs)]
	# new_vals = reduce(hcat, [extp.(new_freqs) for extp in all_extp])
	return Float32.(new_vals)
end
function unnormalize_data(x_arr, orig_te, orig_hs, target_freqs, curr_freqs)
    new_freqs_all = transpose(target_freqs) .* (orig_te)
    all_new_freqs = eachrow(new_freqs_all)
    all_itp = [interpolate((curr_freqs,), x_arr[i, :], Gridded(Linear())) for (i, _) in enumerate(all_new_freqs)]
    all_extp = [extrapolate(itp, 0) for itp in all_itp]
    reinterpolated_data = transpose(reduce(hcat, [extp.(aux_freqs) for (extp, aux_freqs) in zip(all_extp, all_new_freqs)]))
    
    scaling = orig_te .* (orig_hs .^ 2)
    unnormalized = reinterpolated_data .* scaling
end

function init_model(features, layers, modified_orig_freqs, x_new)
    encoders = [Dense(layers[i], layers[i+1], leakyrelu) for i in 1:(length(layers)-2)]
    encoders = hcat(encoders..., Dense(layers[end-1], layers[end], sigmoid))
    decoders = [Dense(layers[i], layers[i-1], leakyrelu) for i in length(layers):-1:3]
    decoders = hcat(decoders..., Dense(layers[2], layers[1], sigmoid))
    custom_activation(sample) = new_modified_softmax(sample; orig_freq=modified_orig_freqs, new_freqs=x_new)
    model = Chain(encoders..., decoders..., custom_activation)
end

"""
	modified_softmax(x_arr; Δx=delx)

Scale matrix of features × samples such that integral of each sample is 1.

`Δx` will use the `delx` created by data.jl if not specified. Will use the trapezoidal rule
to calculate the integral of the current output.

# Example
```julia-repl
julia> domain = [0.0, 0.5, 1.0]
3-element Vector{Float64}:
0.0
0.5
1.0

julia> Δx = domain[begin+1:end] - domain[begin:end-1]
2-element Vector{Float64}:
0.5
0.5

julia> A = [1 1 3 2; 1 2 2 1; 1 3 1 2]
3×4 Matrix{Int64}:
1  1  3  2
1  2  2  1
1  3  1  2

julia> newA = modified_softmax(A; Δx=Δx)
3×4 Matrix{Float64}:
1.0  0.5  1.5  1.33333
1.0  1.0  1.0  0.666667
1.0  1.5  0.5  1.33333

julia> integrate.([domain for _ in 1:4], eachcol(newA))
4-element Vector{Float64}:
1.0
1.0
1.0
1.0
```
"""
function modified_softmax(x_arr::AbstractMatrix; Δx=delx::AbstractVector)
	aux = @. (x_arr[begin:end-1, :] + x_arr[begin+1:end, :])/2
	x_arr./sum((aux).*(Δx), dims=1)
end

"""
	get_data(data, batch_size)

Returns a Flux.DataLoader that will return batches of specificied batch_size until data is 
exhausted. Data should be in shape (samples, features).

# Example
```julia-repl
julia> A = [1 2 3 4; 4 5 6 7; 8 9 10 11]
3×4 Matrix{Int64}:
1  2   3   4
4  5   6   7
8  9  10  11

julia> dl = get_data(A, 2)
2-element DataLoader(transpose(::Matrix{Float32}), shuffle=true, batchsize=2)
  with first element:
  4×2 Matrix{Float32}

julia> first(dl) # <- Because data is shuffled, this is random.
1.0   8.0
2.0   9.0
3.0  10.0
4.0  11.0
```
"""
function get_data(data, batch_size)
	xdata = transpose(Float32.(ustrip.(data)))
	dl = Flux.Data.DataLoader(xdata, batchsize=batch_size, shuffle=true)
end

"""
	training_epoch(m_opt, m_params, m_loss, data)

Training function for a single epoch, works under the assumption that m_loss only requires 
the training data. In order, the required arguments are: the model optimizer, model 
parameters, model loss function, and training data.

This is not meant to be called outside of the train! functions.

See also [`train!`](@ref)
"""
function training_epoch(m_opt, m_params, m_loss, data)
	# grad=nothing
	# println("Params before: , $(m_params[1][1:5, 1:5])")
	# bm_params = deepcopy(m_params)
	for train_batch in data
		# _, back = Flux.pullback(m_params) do
		# 	m_loss(train_batch)
		# end
		# grad = back(1f0)
		grad = Flux.gradient(m_loss, m_params)
		Flux.Optimise.update!(m_opt, m_params, grad)
		# Optimisers.update!(m_opt, m_params, grad)
	end
	# prinln("$(grad.grads[1][1:5, 1:5])")
	# println("Params after: , $(m_params[1][1:5, 1:5])")
	# return (bm_params, deepcopy(grad), deepcopy(m_params))
end

"""
	checkpointing(model, loss, eps, curr_lowest, patience, threshold; filename="model_checkpoint")

Early stopping and checkpoint function. Will take the model, loss, epsilon, lowest loss, 
patience, patience threshold, and a file name. Will save the model as long as its better 
than the lowest loss, otherwise will wait until patience has exceeded the threshold to 
return true to stop training early.

Saves file under using filename argument.

This is not meant to be called outside of the train! functions.

See also [`train!`](@ref)
"""
function checkpointing(model, loss, eps, curr_lowest, patience, threshold, filename)
	m_state = Flux.state(model)
	jldsave(*("models/",filename,".jld2"); m_state)
	if Float32(loss + eps) < curr_lowest
		curr_lowest = loss
		patience = Int32(0)
		return curr_lowest, false
	else
		patience += Int32(1)
		if threshold > 0 && patience >= threshold
			println("Stopping training early because loss has not improved in last $(patience) epochs")
			return curr_lowest, true
		end
		return curr_lowest, false
	end
end

"""
	train!(model, m_opt, m_loss, train_data; epochs=100, batchsize=1024, m_device=cpu)

Generic function for training a model with the optimizer, loss function, and training data. 
Will train the model until it has gone through all epochs.

Returns a vector containing the loss throughout the epochs.

#Arguments
- `epochs=100`: the number of epochs to train a model
- `batchsize=1024`: batch size that the training data will be split to
- `m_device=cpu`: obsolete argument for choosing whether to use cpu/gpu
"""
function train!(model, m_opt, m_loss, train_data; epochs=100, batchsize=1024, m_device=cpu)
	loss_arr = Vector{Float32}(undef, epochs)
	m_params = Flux.params(model)
	data_generator = get_data(train_data, batchsize)
	for epoch in 1:epochs
		training_epoch(m_opt, m_params, m_loss, data_generator)
		loss_arr[epoch] = m_loss(data_generator.data)
		@info "$epoch/$epochs Loss: $(round(loss_arr[epoch]; digits=7))"
	end
	return loss_arr
end

#WOW
"""
	train!(model, m_opt, m_loss, train_data, patience_thres::Int64; epochs=100, batchsize=1024, m_device=cpu, eps=0.001, filename="model_checkpoint")

Generic function for training a model with the optimizer, loss function, and training data. 
Model trains until either the training loss has not improved by at least epsilon for a 
number of epochs equal to a patience threshold or it has gone through all epochs. 

A patience of 0 will make model train for all epochs.

Returns a vector containing the loss throughout the epochs. Loads the model with the lowest 
loss at the end of training.

#Arguments
- `epochs=100`: the number of epochs to train a model
- `batchsize=1024`: batch size that the training data will be split to
- `m_device=cpu`: obsolete argument for choosing whether to use cpu/gpu
- `eps=0.001`: Minimum change required in training loss for patience
- `filename="model_checkpoint"`: Filename used in checkpoints
"""
function train!(model, m_opt, m_loss, train_data, patience_thres::Int64; epochs=100, batchsize=1024, m_device=cpu, eps=0.001, filename="model_checkpoint")
    loss_arr = Vector{Float32}(undef, epochs)
    lowest_loss = Float32(1e8)
	lowest_epoch = 1
    patience = Int32(0)
    m_params = Flux.params(model)
    data_generator = get_data(train_data, batchsize)

    for epoch in 1:epochs
		grads = nothing
		training_loss = 0.0
		# println("Params before: , $(m_params[1][1:5, 1:5])")
        for train_batch in data_generator
			
			loss, grads = Flux.withgradient(model) do m
				y_hat = m(train_batch)
				sqrt(Flux.Losses.mse(y_hat, train_batch))
			end
			# if any(x -> x == NaN || x == Inf, grad)
			# 	error("Gradients contain NaN or Inf values.")
			# end

			Flux.update!(m_opt, model, grads[1])

			

            training_loss += m_loss(train_batch)
			# training_loss += Flux.Losses.msle(train_batch, model(train_batch))
        end
		# print("grad?:",grads)
		# println("Params after: , $(m_params[1][1:5, 1:5])")
        loss_arr[epoch] = training_loss

        lowest_loss, stop = checkpointing(model, training_loss, eps, lowest_loss, patience, patience_thres, "$(filename)_$epoch")
		if loss_arr[epoch] == lowest_loss
			lowest_epoch = epoch
		end
		@info "$epoch/$epochs Loss: $(round(loss_arr[epoch]; digits=7))"

        if stop
            break
        end

    end
    println("Best model had the lowest error of $(lowest_loss)")
    m_state = JLD2.load(*("models/", "$(filename)_$lowest_epoch", ".jld2"), "m_state")
    Flux.loadmodel!(model, m_state)
    return loss_arr
end




"""
	train!(model, m_opt, m_loss, train_data, valid_data; epochs=100, batchsize=1024, m_device=cpu)

Generic function for training a model with the optimizer, loss function, training data, and 
validation data. Will train the model until it has gone through all epochs.

Returns a matrix containing the training and validation loss vectors in each column.

#Arguments
- `epochs=100`: the number of epochs to train a model
- `batchsize=1024`: batch size that the training data will be split to
- `m_device=cpu`: obsolete argument for choosing whether to use cpu/gpu
"""
function train!(model, m_opt, m_loss, train_data, valid_data; epochs=100, batchsize=1024, m_device=cpu)
	train_loss_arr = Vector{Float32}(undef, epochs)
	val_loss_arr = Vector{Float32}(undef, epochs)
	m_params = Flux.params(model)
	data_generator = get_data(train_data, batchsize)

	for epoch in 1:epochs
		training_epoch(m_opt, m_params, m_loss, data_generator)
		train_loss_arr[epoch] = m_loss(data_generator.data)

		validation_loss = m_loss(data_generator.data)
		val_loss_arr[epoch] = validation_loss
		@info "$epoch/$epochs Loss: $(round(loss_arr[epoch]; digits=7))"
	end
	return [train_loss_arr val_loss_arr]
end

"""
	train!(model, m_opt, m_loss, train_data, valid_data, patience_thres::Int64; epochs=100, batchsize=1024, m_device=cpu, eps=0.001, filename="model_checkpoint")

Generic function for training a model with the optimizer, loss function, training data, and 
validation data. Model trains until either the validation loss has not improved by at least 
epsilon for a number of epochs equal to a patience threshold or it has gone through all 
epochs. 

A patience of 0 will make model train for all epochs.

Returns a matrix containing the training and validation loss vectors in each column. Loads 
the model with the lowest loss at the end of training.

#Arguments
- `epochs=100`: the number of epochs to train a model
- `batchsize=1024`: batch size that the training data will be split to
- `m_device=cpu`: obsolete argument for choosing whether to use cpu/gpu
- `eps=0.001`: Minimum change required in training loss for patience
- `filename="model_checkpoint"`: Filename used in checkpoints
"""
function train!(model, m_opt, m_loss, train_data, valid_data, patience_thres::Int64; epochs=100, batchsize=1024, m_device=cpu, eps=0.001, filename="model_checkpoint")
	train_loss_arr = Vector{Float32}(undef, epochs)
	val_loss_arr = Vector{Float32}(undef, epochs)
	lowest_loss = 1e8
	m_params = Flux.params(model)
	data_generator = get_data(train_data, batchsize)
	patience = 0
	for epoch in 1:epochs
		training_epoch(m_opt, m_params, m_loss, data_generator)
		train_loss_arr[epoch] = m_loss(data_generator.data)

		validation_loss = m_loss(data_generator.data)
		val_loss_arr[epoch] = validation_loss

		lowest_loss, stop = checkpointing(model, validation_loss, eps, lowest_loss, patience, patience_thres, filename)
		@info "$epoch/$epochs Loss: $(round(loss_arr[epoch]; digits=7))"
		if stop
			break
		end
	end
	println("Best model had the lowest validation error of $(lowest_loss)")
	m_state = JLD2.load(*("models/",filename,".jld2"), "m_state");
	Flux.loadmodel!(model, m_state);
	return [train_loss_arr val_loss_arr]
end

"""
	train!(model, m_opt, m_loss, train_data, valid_data, patience_thres, lr_scheduler; epochs=100, batchsize=1024, m_device=cpu, filename="model_checkpoint")

Generic function for training a model with the optimizer, loss function, training data, 
validation data, and a learning rate scheduler. Model trains until either the validation 
loss has not improved for a number of epochs equal to a patience threshold or it has gone 
through all epochs. 

A patience of 0 will make model train for all epochs.

Returns a matrix containing the training and validation loss vectors in each column. Loads 
the model with the lowest loss at the end of training.

#Arguments
- `epochs=100`: the number of epochs to train a model
- `batchsize=1024`: batch size that the training data will be split to
- `m_device=cpu`: obsolete argument for choosing whether to use cpu/gpu
- `filename="model_checkpoint"`: Filename used in checkpoints
"""
function train!(model, m_opt, m_loss, train_data, valid_data, patience_thres, lr_scheduler; epochs=100, batchsize=1024, m_device=cpu, filename="model_checkpoint")
	train_loss_arr = Vector{Float32}(undef, epochs)
	val_loss_arr = Vector{Float32}(undef, epochs)
	lowest_loss = 1e8
	m_params = Flux.params(model)
	patience=0
	data_generator = get_data(train_data, batchsize)
	for (eta, epoch) in zip(lr_scheduler, 1:epochs)
		model_opt.eta = eta
		training_epoch(m_opt, m_params, m_loss, data_generator)
		train_loss_arr[epoch] = m_loss(data_generator.data)

		validation_loss = m_loss(data_generator.data)
		val_loss_arr[epoch] = validation_loss
		lowest_loss, stop = checkpointing(model, validation_loss, eps, lowest_loss, patience, patience_thres, filename)
		@info "$epoch/$epochs Loss: $(round(loss_arr[epoch]; digits=7))"
		if stop
			break
		end
	end
	println("Best model had the lowest error of $(lowest_loss)")
	m_state = JLD2.load(*("models/",filename,".jld2"), "m_state");
	Flux.loadmodel!(model, m_state);
	return [train_loss_arr val_loss_arr]
end

"""
	plot_history(hist::AbstractVector)

Generic function for plotting the entire training loss history of a model and the last third 
of it. Used in a qualitative manner to determine the accuracy of the model.
"""
function plot_history(hist::AbstractVector)
	seg = length(hist)÷3
	
	all_loss = plot(1:length(hist), hist, labels="Training Error")
	
	recent_loss = plot(length(hist)-seg:length(hist), hist[end-seg:end], labels="Training Error")
	
	plot(all_loss, recent_loss, size=(1500, 750))
end

"""
	plot_history(hist::AbstractMatrix)

Generic function for plotting the entire training and validation loss history of a model and 
the last third of it. Used in a qualitative manner to determine the accuracy of the model.
"""
function plot_history(hist::AbstractMatrix)
	seg = size(hist)[1]÷3
	
	all_loss = plot(1:size(hist)[1], hist[:,1], labels="Training Error")
	plot!(1:size(hist)[1], hist[:,2], labels="Validation Error")

	
	recent_loss = plot(size(hist)[1]-seg:size(hist)[1], hist[end-seg:end, 1], labels="Training Error")
	plot!(size(hist)[1]-seg:size(hist)[1], hist[end-seg:end, 2], labels="Validation Error")
	
	plot(all_loss, recent_loss, size=(1500, 750))
end

end