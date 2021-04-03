### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ d402633e-8c18-11eb-119d-017ad87927b0
using Flux

# ╔═╡ c70eaa72-90ad-11eb-3600-016807d53697
using StatsFuns: log1pexp #log(1 + exp(x))

# ╔═╡ 76f893fe-928b-11eb-0034-8586ba4e1918
using BSON: @save, @load # To save model

# ╔═╡ 0a761dc4-90bb-11eb-1f6c-fba559ed5f66
using Plots ##

# ╔═╡ 9e368304-8c16-11eb-0417-c3792a4cd8ce
md"""
# Assignment 3: Variational Autoencoders

- Student Name: Tianyi Liu
- Student #: 1005820827
- Collaborators: Lazar Atanackovic, Phil Fradkin

## Background

In this assignment we will implement and investigate a Variational Autoencoder as introduced by Kingma and Welling in [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).


### Data: Binarized MNIST

In this assignment we will consider an  MNIST dataset of $28\times 28$ pixel images where each pixel is **either on or off**.

The binary variable $x_i \in \{0,1\}$ indicates whether the $i$-th pixel is off or on.

Additionally, we also have a digit label $y \in \{0, \dots, 9\}$. Note that we will not use these labels for our generative model. We will, however, use them for our investigation to assist with visualization.

### Tools

In previous assignments you were required to implement a simple neural network and gradient descent manually. In this assignment you are permitted to use a machine learning library for convenience functions such as optimizers, neural network layers, initialization, dataloaders.

However, you **may not use any probabilistic modelling elements** implemented in these frameworks. You cannot use `Distributions.jl` or any similar software. In particular, sampling from and evaluating probability densities under distributions must be written explicitly by code written by you or provided in starter code.
"""

# ╔═╡ 54749c92-8c1d-11eb-2a54-a1ae0b1dc587
# load the original greyscale digits
train_digits = Flux.Data.MNIST.images(:train)

# ╔═╡ 176f0938-8c1e-11eb-1135-a5db6781404d
# convert from tuple of (28,28) digits to vector (784,N) 
greyscale_MNIST = hcat(float.(reshape.(train_digits,:))...)

# ╔═╡ c6fa2a9c-8c1e-11eb-3e3c-9f8f5c218dec
# binarize digits
binarized_MNIST = greyscale_MNIST .> 0.5

# ╔═╡ 9e7e46b0-8e84-11eb-1648-0f033e4e6068
# partition the data into batches of size BS
BS = 200

# ╔═╡ 743d473c-8c1f-11eb-396d-c92cacb0235b
# batch the data into minibatches of size BS
batches = Flux.Data.DataLoader(binarized_MNIST, batchsize=BS)

# ╔═╡ db655546-8e84-11eb-21df-25f7c8e82362
# confirm dimensions are as expected (D,BS)
size(first(batches))

# ╔═╡ 2093080c-8e85-11eb-1cdb-b35eb40e3949
md"""
## Model Definition

Each element in the data $x \in D$ is a vector of $784$ pixels. 
Each pixel $x_d$ is either on, $x_d = 1$ or off $x_d = 0$.

Each element corresponds to a handwritten digit $\{0, \dots, 9\}$.
Note that we do not observe these labels, we are *not* training a supervised classifier.

We will introduce a latent variable $z \in \mathbb{R}^2$ to represent the digit.
The dimensionality of this latent space is chosen so we can easily visualize the learned features. A larger dimensionality would allow a more powerful model. 


- **Prior**: The prior over a digit's latent representation is a multivariate standard normal distribution. $p(z) = \mathcal{N}(z \mid \mathbf{0}, \mathbf{1})$
- **Likelihood**: Given a latent representation $z$ we model the distribution over all 784 pixels as the product of independent Bernoulli distributions parametrized by the output of the "decoder" neural network $f_\theta(z)$.
```math
p_\theta(x \mid z) = \prod_{d=1}^{784} \text{Ber}(x_d \mid f_\theta(z)_d)
```

### Model Parameters

Learning the model will involve optimizing the parameters $\theta$ of the "decoder" neural network, $f_\theta$. 

You may also use library provided layers such as `Dense` [as described in the documentation](https://fluxml.ai/Flux.jl/stable/models/basics/#Stacking-It-Up-1). 

Note that, like many neural network libraries, Flux avoids explicitly providing parameters as function arguments, i.e. `neural_net(z)` instead of `neural_net(z, params)`.

You can access the model parameters `params(neural_net)` for taking gradients `gradient(()->loss(data), params(neural_net))` and updating the parameters with an [Optimiser](https://fluxml.ai/Flux.jl/stable/training/optimisers/).

However, if all this is too fancy feel free to continue using your implementations of simple neural networks and gradient descent from previous assignments.

"""

# ╔═╡ 45bc7e00-90ac-11eb-2d62-092a13dd1360
md"""
### Numerical Stability

The Bernoulli distribution $\text{Ber}(x \mid \mu)$ where $\mu \in [0,1]$ is difficult to optimize for a few reasons.

We prefer unconstrained parameters for gradient optimization. This suggests we might want to transform our parameters into an unconstrained domain, e.g. by parameterizing the `log` parameter.

We also should consider the behaviour of the gradients with respect to our parameter, even under the transformation to unconstrained domain. For instance a poor transformation might encourage optimization into regions where gradient magnitude vanishes. This is often called "saturation".

For this reasons we should use a numerically stable transformation of the Bernoulli parameters. 
One solution is to parameterize the "logit-means": $y = \log(\frac{\mu}{1-\mu})$.

We can exploit further numerical stability, e.g. in computing $\log(1 + exp(x))$, using library provided functions `log1pexp`
"""


# ╔═╡ e12a5b5e-90ad-11eb-25a8-43c9aff1e0db
# Numerically stable bernoulli density, why do we do this?
function bernoulli_log_density(x, logit_means)
  	"""Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
	"""	
		in: x, logit_means: dim × batch_size
		out: dim × batch_size 
	"""
	# Naive
	# return x .* logit_means - log1pexp.(logit_means)
	# Stable
	b = x .* 2 .- 1 # [0,1] -> [-1,1]
  	return - log1pexp.(-b .* logit_means)
end

# ╔═╡ 3b07d20a-8e88-11eb-1956-ddbaaf178cb3
md"""
## Model Implementation

- `log_prior` that computes the log-density of a latent representation under the prior distribution.
- `decoder` that takes a latent representation $z$ and produces a 784-dimensional vector $y$. This will be a simple neural network with the following architecture: a fully connected layer with 500 hidden units and `tanh` non-linearity, a fully connected output layer with 784-dimensions. The output will be unconstrained, no activation function.
- `log_likelihood` that given an array binary pixels $x$ and the output from the decoder, $y$ corresponding to "logit-means" of the pixel Bernoullis $y = log(\frac{\mu}{1-\mu})$ compute the **log-**likelihood under our model. 
- `joint_log_density` that uses the `log_prior` and `log_likelihood` and gives the log-density of their joint distribution under our model $\log p_\theta(x,z)$.

Note that these functions should accept a batch of digits and representations, an array with elements concatenated along the last dimension.
"""

# ╔═╡ ec53e34c-9139-11eb-2cef-eb640a77b9b4
function fully_factorizerd_gaussian(z, μ, logσ)
	σ = exp.(logσ)
	return sum(log.(1 ./ (2 * π * σ .^ 2) .^ 0.5) .+ (-0.5 * ((z .- μ) ./ σ) .^ 2),
dims=1) # 1 × BS
end

# ╔═╡ ce50c994-90af-11eb-3fc1-a3eea9cda1a2
log_prior(z) =  fully_factorizerd_gaussian(z, 0, 0)

# ╔═╡ 3b386e56-90ac-11eb-31c2-29ba365a6967
Dz, Dh, Ddata = 2, 500, 28^2

# ╔═╡ d7415d20-90af-11eb-266b-b3ea86750c98
# You can use Flux's Chain and Dense here
decoder = Chain(Dense(Dz, Dh, tanh), Dense(Dh, Ddata))

# ╔═╡ 5b8721dc-8ea3-11eb-3ace-0b13c00ce256
function log_likelihood(x,z)
	""" Compute log likelihood log_p(x|z)"""
	# use numerically stable bernoulli
	logit_means = decoder(z)
	return sum(bernoulli_log_density(x, logit_means), dims=1) # 1 × BS
end

# ╔═╡ 0afbe054-90b0-11eb-0233-6faede537bc4
joint_log_density(x,z) = log_likelihood(x,z) + log_prior(z) # 1 × BS

# ╔═╡ b8a20c8c-8ea4-11eb-0d48-a37047ab70c5
md"""
## Amortized Approximate Inference with Learned Variational Distribution

Now that we have set up a model, we would like to learn the model parameters $\theta$.
Notice that the only indication for *how* our model should represent digits in $z \in \mathbb{R}^2$ is that they should look like our prior $\mathcal{N}(0,1)$.

How should our model learn to represent digits by 2D latent codes? 
We want to maximize the likelihood of the data under our model $p_\theta(x) = \int p_\theta(x,z) dz = \int p_\theta(x \mid z)p(z) dz$.

We have learned a few techniques to approximate these integrals, such as sampling via MCMC. 
Also, 2D is a low enough latent dimension, we could numerically integrate, e.g. with a quadrature.

Instead, we will use variational inference and find an approximation $q_\phi(z) \approx p_\theta(z \mid x)$. This approximation will allow us to efficiently estimate our objective, the data likelihood under our model. Further, we will be able to use this estimate to update our model parameters via gradient optimization.

Following the motivating paper, we will define our variational distribution as $q_\phi$ also using a neural network. The variational parameters, $\phi$ are the weights and biases of this "encoder" network.

This encoder network $q_\phi$ will take an element of the data $x$ and give a variational distribution over latent representations. In our case we will assume this output variational distribution is a fully-factorized Gaussian.
So our network should output the $(\mu, \log \sigma)$.

To train our model parameters $\theta$ we will need also train variational parameters $\phi$.
We can do both of these optimization tasks at once, propagating gradients of the loss to update both sets of parameters.

The loss, in this case, no longer being the data likelihood, but the Evidence Lower BOund (ELBO).

1. Implement `log_q` that accepts a representation $z$ and parameters $\mu, \log \sigma$ and computes the logdensity under our variational family of fully factorized guassians.
1. Implement `encoder` that accepts input in data domain $x$ and outputs parameters to a fully-factorized guassian $\mu, \log \sigma$. This will be a neural network with fully-connected architecture, a single hidden layer with 500 units and `tanh` nonlinearity and fully-connected output layer to the parameter space.
2. Implement `elbo` which computes an unbiased estimate of the Evidence Lower BOund (using simple monte carlo and the variational distribution). This function should take the model $p_\theta$, the variational model $q_\phi$, and a batch of inputs $x$ and return a single scalar averaging the ELBO estimates over the entire batch.
4. Implement simple loss function `loss` that we can use to optimize the parameters $\theta$ and $\phi$ with `gradient`. We want to maximize the lower bound, with gradient descent. (This is already implemented)

"""

# ╔═╡ 615e59c6-90b6-11eb-2598-d32538e14e8f
log_q(z, q_μ, q_logσ) = fully_factorizerd_gaussian(z, q_μ, q_logσ) # 1 × BS

# ╔═╡ 1859c670-9133-11eb-275b-cfdb9b3affae
encoder = Chain(Dense(Ddata, Dh, tanh), Dense(Dh, 2*Dz))

# ╔═╡ 3eda2d1a-9202-11eb-2909-23a577d74f6f
function wrap_to_gaussian_par(out)
	return out[1:Dz, :], out[Dz+1:2*Dz, :]
end

# ╔═╡ ccf226b8-90b6-11eb-15a2-d30c9e27aebb
function elbo(x)
  	#TODO variational parameters from data
	q_μ, q_logσ = wrap_to_gaussian_par(encoder(x))
  	#TODO: sample from variational distribution
	z = q_μ .+ exp.(q_logσ) .* randn(size(q_μ))
	#TODO: joint likelihood of z and x under model
  	joint_ll = joint_log_density(x, z)
	#TODO: likelihood of z under variational distribution
  	log_q_z = log_q(z, q_μ, q_logσ)
	#TODO: Scalar value, mean variational evidence lower bound over batch
  	elbo_estimate = sum(joint_ll - log_q_z) / size(x)[2] # scalar
  	return elbo_estimate
end

# ╔═╡ f00b5444-90b6-11eb-1e0d-5d034735ec0e
function loss(x)
  	return -elbo(x)
end

# ╔═╡ 70ccd9a4-90b7-11eb-1fb2-3f7aff4073a0
md"""
## Optimize the model and amortized variational parameters

If the above are implemented correctly, stable numerically, and differentiable automatically then we can train both the `encoder` and `decoder` networks with graident optimzation.

We can compute `gradient`s of our `loss` with respect to the `encoder` and `decoder` parameters `theta` and `phi`.

We can use a `Flux.Optimise` provided optimizer such as `ADAM` or our own implementation of gradient descent to `update!` the model and variational parameters.

Use the training data to learn the model and variational networks.
"""

# ╔═╡ 5efb0baa-90b8-11eb-304f-7dbb8d5c0ba6
function train!(enc, dec, data; nepochs=100, force_retrain=false)
	params = Flux.params(enc, dec)
	opt = ADAM()
	losses_train = []
	
	if isfile("params.bson") & (~force_retrain)
		@info "Loading pre-trained weights from params.bson\n"
		Core.eval(Main, :(import Zygote))
		@load "params.bson" params
		Flux.loadparams!((encoder, decoder), params)
	else
		@info "Training model from scratch\n"
		steps = length(data)
		tic_base = time()
		for epoch in 1:nepochs
			# CHANGE FOR LOGGING!
			# for batch in data
			for (step, batch) in enumerate(data)
				tic = time()

				loss_train = loss(batch)
				push!(losses_train, loss_train)
				# compute gradient wrt loss
				dparams = Flux.gradient(() -> loss(batch), params)
				# update parameters
				Flux.Optimise.update!(opt, params, dparams)

				toc = time()
				eta = convert(Int, round((toc - tic_base) / ((epoch - 1) * steps +
	step) * ((nepochs - epoch) * steps + steps - step), digits=0))
				@info "iter: $epoch / $nepochs\tstep: $step / $steps\tloss:$loss_train\ttime: $(round(toc - tic, digits=4)) s ETA: $eta s\n"

			end
			# Optional: log loss using @info "Epoch $epoch: loss:..."
			# Optional: visualize training progress with plot of loss
			# Optional: save trained parameters to avoid retraining later
		end
		toc_base = time()
		@info "Total training time: $(round(toc_base - tic_base, digits=4)) s"
		params = Flux.params(enc, dec)
		@save "params.bson" params
		# return nothing, this mutates the parameters of enc and dec!
	end
	
end

# ╔═╡ c86a877c-90b9-11eb-31d8-bbcb71e4fa66
train!(encoder, decoder, batches, nepochs=20, force_retrain=false)

# ╔═╡ 17c5ddda-90ba-11eb-1fce-93b8306264fb
md"""
## Visualizing the Model Learned Representation

We will use the model and variational networks to visualize the latent representations of our data learned by the model.

We will use a variatety of qualitative techniques to get a sense for our model by generating distributions over our data, sampling from them, and interpolating in the latent space.
"""

# ╔═╡ 1201bfee-90bb-11eb-23e5-af9a61f64679
md"""
### 1. Latent Distribution of Batch

1. Use `encoder` to produce a batch of latent parameters $\mu, \log \sigma$
2. Take the 2D mean vector $\mu$ for each latent parameter in the batch.
3. Plot these mene vectors in the 2D latent space with a scatterplot
4. Colour each point according to its "digit class label" 0 to 9.
5. Display a single colourful scatterplot
"""

# ╔═╡ d9b3c078-90bb-11eb-1c41-c784851a9148
begin
	μ, logσ = wrap_to_gaussian_par(encoder(first(batches)))
	label = Flux.Data.MNIST.labels(:train)[1:BS]
	plot()
	for i in 0:9
		plot!(μ[1,label.==i],μ[2,label.==i],seriestype = :scatter,label=i)
	end
	title!("2D means of VAE")
	xlabel!("VAE1")
	ylabel!("VAE2")
end

# ╔═╡ dcedbba4-90bb-11eb-2652-bf6448095107
md"""
### 2. Visualizing Generative Model of Data

1. Sample 10 $z$ from the prior $p(z)$.
2. Use the model to decode each $z$ to the distribution logit-means over $x$.
3. Transform the logit-means to the Bernoulli means $\mu$. (this does not need to be efficient)
4. For each $z$, visualize the $\mu$ as a $28 \times 28$ greyscale images.
5. For each $z$, sample 3 examples from the Bernoulli likelihood $x \sim \text{Bern}(x \mid \mu(z))$.
6. Display all plots in a single 10 x 4 grid. Each row corresponding to a sample $z$. Do not include any axis labels.
"""

# ╔═╡ ab1d0b80-9358-11eb-256c-6fe6bba786dc
manual_bernoulli(μ) = convert(Int64, rand() < μ)

# ╔═╡ 79804c6c-929b-11eb-1ae5-3dbcf295195b
wrap_to_image(array) = Gray.(reshape(array, (28, 28, size(array)[2])))

# ╔═╡ 632b10ae-929c-11eb-1de0-e1e6330c5277
wrap_to_μ(logit_means) = 2 .- exp.(log1pexp.(-log1pexp.(logit_means)))

# ╔═╡ 805a265e-90be-11eb-2c34-1dd0cd1a968c
begin
	z = reshape(randn(20),(2,10))
	log_mean = decoder(z)
	μs = wrap_to_μ(log_mean)
	sample1 = wrap_to_image(manual_bernoulli.(μs))
	sample2 = wrap_to_image(manual_bernoulli.(μs))
	sample3 = wrap_to_image(manual_bernoulli.(μs))
	imgs = wrap_to_image(μs)
	imshow = []
	for i in 1:10
		push!(imshow, plot(imgs[:,:,i], axis=nothing))
		push!(imshow, plot(sample1[:,:,i], axis=nothing))
		push!(imshow, plot(sample2[:,:,i], axis=nothing))
		push!(imshow, plot(sample3[:,:,i], axis=nothing))
	end
	display(plot(imshow..., layout=((10,4)), size=(700,700)))
	current();
end

# ╔═╡ 82b0a368-90be-11eb-0ddb-310f332a83f0
md"""
### 3. Visualizing Regenerative Model and Reconstruction

1. Sample 4 digits from the data $x \sim \mathcal{D}$
2. Encode each digit to a latent distribution $q_\phi(z)$
3. For each latent distribution, sample 2 representations $z \sim q_\phi$
4. Decode each $z$ and transform to the Bernoulli means $\mu$
5. For each $\mu$, sample 1 "reconstruction" $\hat x \sim \text{Bern}(x \mid \mu)$
6. For each digit $x$ display (28x28) greyscale images of $x, \mu, \hat x$
"""

# ╔═╡ f27bdffa-90c0-11eb-0f71-6d572f799290
begin
	sample_digits = binarized_MNIST[:,rand(1:size(binarized_MNIST)[2], 4)]
	μ3, logσ3 = wrap_to_gaussian_par(encoder(sample_digits))
	σ3 = exp.(logσ3)
	samples_latent = μ3 .+ σ3 .* randn(2, 4)
	# 1st: (1,5) 2nd: (2,6) 3rd: (3,7) 4th: (4,8)
	samples_latent = hcat(samples_latent, μ3 .+ σ3 .* randn(2, 4))
	log_mean3 = decoder(samples_latent)
	μs3 = wrap_to_μ(log_mean3)
	recon = manual_bernoulli.(μs3)
	img_x = wrap_to_image(sample_digits)
	img_μs3 = wrap_to_image(μs3)
	img_recon = wrap_to_image(recon)
	imshow3 = []
	for i in 1:4
		push!(imshow3, plot(img_x[:,:,i],title="$i-th digit", axis=nothing))
		push!(imshow3, plot(img_μs3[:,:,i], title="μ 1", axis=nothing))
		push!(imshow3, plot(img_recon[:,:,i], title="recon 1", axis=nothing))
		push!(imshow3, plot(img_μs3[:,:,i + 4], title="μ 2", axis=nothing))
		push!(imshow3, plot(img_recon[:,:,i + 4], title="recon 2", axis=nothing))
	end
	display(plot(imshow3..., layout=((4,5))))
	current();
end

# ╔═╡ 02181adc-90c1-11eb-29d7-736dce72a0ac
md"""
### 4. Latent Interpolation Along Lattice

1. Produce a $50 \times 50$ "lattice" or collection of cartesian coordinates $z = (z_x, z_y) \in \mathbb{R}^2$.
2. For each $z$, decode and transform to a 28x28 greyscale image of the Bernoulli means $\mu$
3. Each point in the `50x50` latent lattice corresponds now to a `28x28` greyscale image. Concatenate all these images appropriately.
4. Display a single `1400x1400` pixel greyscale image corresponding to the learned latent space.
"""

# ╔═╡ 1dee7a66-92c0-11eb-1222-0fa933c03bc1
function manifold_cat(μz_img)
	img_whole = nothing
	for row in 1:50
		row_cur = nothing
		for column in 1:50
			img_cur = μz_img[:,:,(row-1) * 50 + column]
			if row_cur == nothing
				row_cur = img_cur
			else
				row_cur = hcat(row_cur, img_cur)
			end
		end
		if img_whole == nothing
			img_whole = row_cur
		else
			img_whole = vcat(img_whole, row_cur)
		end
	end
	return img_whole
end

# ╔═╡ 3a0e1d5a-90c2-11eb-16a7-8f9de1ea09e4
begin
	vmax = 2
	vmin = -vmax
	res = 2 * vmax / (50 - 1)
	
	z_lattice = reshape([[i; j] for i in vmin:res:vmax, j in vmax:-res:vmin], 2500)
	z_input = reshape([(z_lattice...)...], (2,2500)) # 2 × 2500
	log_μz = decoder(z_input)
	μz = wrap_to_μ(log_μz)
	img_μz = wrap_to_image(μz)
	img_manifold = manifold_cat(img_μz)
	plot(img_manifold, title="manifold", axis=nothing, size=(700,700))
end

# ╔═╡ 95263d58-92c0-11eb-37d8-11a9fb0be272
md"""
The manifold is displayed in a standard Cartesian coordinates, i.e., the top-left corner corresponds to (-2, 2), the bottom-right corner corrsponds to (2, -2).
"""

# ╔═╡ Cell order:
# ╟─9e368304-8c16-11eb-0417-c3792a4cd8ce
# ╠═d402633e-8c18-11eb-119d-017ad87927b0
# ╠═54749c92-8c1d-11eb-2a54-a1ae0b1dc587
# ╠═176f0938-8c1e-11eb-1135-a5db6781404d
# ╠═c6fa2a9c-8c1e-11eb-3e3c-9f8f5c218dec
# ╠═9e7e46b0-8e84-11eb-1648-0f033e4e6068
# ╠═743d473c-8c1f-11eb-396d-c92cacb0235b
# ╠═db655546-8e84-11eb-21df-25f7c8e82362
# ╟─2093080c-8e85-11eb-1cdb-b35eb40e3949
# ╟─45bc7e00-90ac-11eb-2d62-092a13dd1360
# ╠═c70eaa72-90ad-11eb-3600-016807d53697
# ╠═e12a5b5e-90ad-11eb-25a8-43c9aff1e0db
# ╟─3b07d20a-8e88-11eb-1956-ddbaaf178cb3
# ╠═ec53e34c-9139-11eb-2cef-eb640a77b9b4
# ╠═ce50c994-90af-11eb-3fc1-a3eea9cda1a2
# ╠═3b386e56-90ac-11eb-31c2-29ba365a6967
# ╠═d7415d20-90af-11eb-266b-b3ea86750c98
# ╠═5b8721dc-8ea3-11eb-3ace-0b13c00ce256
# ╠═0afbe054-90b0-11eb-0233-6faede537bc4
# ╟─b8a20c8c-8ea4-11eb-0d48-a37047ab70c5
# ╠═615e59c6-90b6-11eb-2598-d32538e14e8f
# ╠═1859c670-9133-11eb-275b-cfdb9b3affae
# ╠═3eda2d1a-9202-11eb-2909-23a577d74f6f
# ╠═ccf226b8-90b6-11eb-15a2-d30c9e27aebb
# ╠═f00b5444-90b6-11eb-1e0d-5d034735ec0e
# ╟─70ccd9a4-90b7-11eb-1fb2-3f7aff4073a0
# ╠═76f893fe-928b-11eb-0034-8586ba4e1918
# ╠═5efb0baa-90b8-11eb-304f-7dbb8d5c0ba6
# ╠═c86a877c-90b9-11eb-31d8-bbcb71e4fa66
# ╟─17c5ddda-90ba-11eb-1fce-93b8306264fb
# ╠═0a761dc4-90bb-11eb-1f6c-fba559ed5f66
# ╟─1201bfee-90bb-11eb-23e5-af9a61f64679
# ╠═d9b3c078-90bb-11eb-1c41-c784851a9148
# ╟─dcedbba4-90bb-11eb-2652-bf6448095107
# ╠═ab1d0b80-9358-11eb-256c-6fe6bba786dc
# ╠═79804c6c-929b-11eb-1ae5-3dbcf295195b
# ╠═632b10ae-929c-11eb-1de0-e1e6330c5277
# ╠═805a265e-90be-11eb-2c34-1dd0cd1a968c
# ╟─82b0a368-90be-11eb-0ddb-310f332a83f0
# ╠═f27bdffa-90c0-11eb-0f71-6d572f799290
# ╟─02181adc-90c1-11eb-29d7-736dce72a0ac
# ╠═1dee7a66-92c0-11eb-1222-0fa933c03bc1
# ╠═3a0e1d5a-90c2-11eb-16a7-8f9de1ea09e4
# ╟─95263d58-92c0-11eb-37d8-11a9fb0be272
