module SuperHawkes

# ===
# MODULES
# ===

using LinearAlgebra: Matrix
using Distributions: length, get_evalsamples
using Random
using Distributions
using LinearAlgebra
using StatsFuns
using StatsBase
using Plots

include("SuperNodes.jl")
using .SuperNodes: SuperArray, SuperMatrix, flatten_dims, expand_dims, to_supernode, from_supernode

using Base: @kwdef


# ===
# EXPORTS
# ===

# Generative model
export SuperHawkesProcess, Spikes
export sample, resample, plot_spikes, partially_observed_spikes, fully_observed_spikes
export length, get_parents, get_sequenceIDs, evaluate_pdf
export logjoint, loglike_data, logprob_prior, exact_loglike

# Inference
export get_lookback_spikes, sample_parents!
export forward_pass_tree, backward_sample_tree!

# Conjugate posterior updates
export update_posteriors!

# Utility functions
export make_α0_prior, make_θ0_prior, make_αW_prior, make_θW_prior, make_αR_prior, make_θR_prior
export make_α0_prior_flat, make_θ0_prior_flat, make_αW_prior_flat, make_θW_prior_flat, make_αR_prior_flat, make_θR_prior_flat
export accuracy, posterior_accuracy

# ===
# INCLUDES
# ===

include("model.jl")
include("parents.jl")  # Parent assignment inference
include("sequences.jl")  # Sequence assignment inference
include("posteriors.jl") # Posterior updates
include("utils.jl")

end


