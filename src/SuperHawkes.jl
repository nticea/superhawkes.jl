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
export sample, resample, plot_spikes, partially_observed_spikes, fully_observed_spikes, length, get_parents, get_sequenceIDs

# Inference
export get_lookback_spikes, sample_parents!
export forward_pass_tree, backward_sample_tree!

# Conjugate posterior updates
export update_posteriors!


# ===
# INCLUDES
# ===

include("model.jl")
include("parents.jl")  # Parent assignment inference
include("sequences.jl")  # Sequence assignment inference
include("posteriors.jl") # Posterior updates

end


