#=##############################################################################
# DESCRIPTION
    Wake trajectory analysis utilities for FLOWUnsteady VPM simulations
    
# AUTHORSHIP
  * Created   : 2025-01-09
  * License   : MIT
=###############################################################################

import FLOWUnsteady as uns
import FLOWVPM as vpm
import PyPlot as plt
using LinearAlgebra
using Statistics

"""
    WakeTrajectoryCollector

Data structure to collect and store particle trajectories during VPM simulation.
"""
mutable struct WakeTrajectoryCollector
    trajectories::Dict{Int, Vector{Vector{Float64}}}  # particle_id => [position_history]
    particle_birth_times::Dict{Int, Float64}          # particle_id => birth_time
    particle_strengths::Dict{Int, Vector{Float64}}    # particle_id => strength_history
    time_history::Vector{Float64}                     # Time stamps
    max_age::Float64                                  # Maximum particle age to track
    min_strength::Float64                             # Minimum strength to track
    sample_frequency::Int                             # Sample every N time steps
    current_step::Int                                 # Current time step counter
    
    function WakeTrajectoryCollector(; max_age=Inf, min_strength=0.0, sample_frequency=1)
        new(Dict{Int, Vector{Vector{Float64}}}(),
            Dict{Int, Float64}(),
            Dict{Int, Vector{Float64}}(),
            Float64[],
            max_age,
            min_strength,
            sample_frequency,
            0)
    end
end

"""
    collect_wake_trajectories!(collector, pfield, t; 
                               filter_region=nothing, max_particles=1000)

Runtime function to collect particle trajectories during simulation.
This function should be included in the `extra_runtime_function` of run_simulation.

# Arguments
- `collector`: WakeTrajectoryCollector instance
- `pfield`: VPM particle field
- `t`: Current simulation time
- `filter_region`: Optional function to filter particles by region (x,y,z) -> Bool
- `max_particles`: Maximum number of particles to track simultaneously
"""
function collect_wake_trajectories!(collector::WakeTrajectoryCollector, 
                                   pfield, t; 
                                   filter_region=nothing, 
                                   max_particles=1000)
    
    collector.current_step += 1
    
    # Sample at specified frequency
    if collector.current_step % collector.sample_frequency != 0
        return false
    end
    
    push!(collector.time_history, t)
    current_time = t
    
    # Track existing particles and add new ones
    particles_to_track = Set{Int}()
    
    for i in 1:vpm.get_np(pfield)
        particle = vpm.get_particle(pfield, i)
        particle_id = i  # Using index as ID (could be improved)
        
        # Skip if particle doesn't meet criteria
        gamma_mag = norm(particle.Gamma)
        if gamma_mag < collector.min_strength
            continue
        end
        
        # Apply spatial filter if provided
        if filter_region !== nothing && !filter_region(particle.X...)
            continue
        end
        
        # Initialize new particle tracking
        if !haskey(collector.trajectories, particle_id)
            if length(collector.trajectories) >= max_particles
                continue  # Skip new particles if at max capacity
            end
            
            collector.trajectories[particle_id] = [copy(particle.X)]
            collector.particle_birth_times[particle_id] = current_time
            collector.particle_strengths[particle_id] = [gamma_mag]
        else
            # Update existing particle
            push!(collector.trajectories[particle_id], copy(particle.X))
            push!(collector.particle_strengths[particle_id], gamma_mag)
        end
        
        push!(particles_to_track, particle_id)
    end
    
    # Remove particles that are too old or no longer exist
    particles_to_remove = Int[]
    for (particle_id, birth_time) in collector.particle_birth_times
        particle_age = current_time - birth_time
        
        if particle_age > collector.max_age || !(particle_id in particles_to_track)
            push!(particles_to_remove, particle_id)
        end
    end
    
    # Clean up old particles
    for particle_id in particles_to_remove
        delete!(collector.trajectories, particle_id)
        delete!(collector.particle_birth_times, particle_id)
        delete!(collector.particle_strengths, particle_id)
    end
    
    return false  # Continue simulation
end

"""
    create_wake_monitor(; max_age=5.0, min_strength=1e-6, sample_frequency=2)

Creates a wake trajectory monitoring function for use in VPM simulations.
Returns both the collector and the monitoring function.

# Example usage:
```julia
collector, wake_monitor = create_wake_monitor(max_age=3.0, sample_frequency=1)

# In run_simulation call:
runtime_function = uns.concatenate(other_monitors, wake_monitor)
```
"""
function create_wake_monitor(; max_age=5.0, min_strength=1e-6, sample_frequency=2,
                            filter_region=nothing, max_particles=1000)
    
    collector = WakeTrajectoryCollector(max_age=max_age, 
                                       min_strength=min_strength,
                                       sample_frequency=sample_frequency)
    
    function wake_monitor(sim, pfield, t, dt; optargs...)
        collect_wake_trajectories!(collector, pfield, t; 
                                  filter_region=filter_region,
                                  max_particles=max_particles)
        return false
    end
    
    return collector, wake_monitor
end

"""
    plot_wake_trajectories_3d(collector; 
                              color_by=:time, alpha=0.7, linewidth=1.0,
                              rotor_centers=nothing, rotor_radius=nothing,
                              xlim=nothing, ylim=nothing, zlim=nothing,
                              title="Wake Trajectories")

Plot 3D wake trajectories from collected data.

# Arguments
- `collector`: WakeTrajectoryCollector with trajectory data
- `color_by`: Color trajectories by :time, :strength, or :age
- `alpha`: Line transparency
- `linewidth`: Line width
- `rotor_centers`: Array of rotor center positions for reference
- `rotor_radius`: Rotor radius for reference circles
- `xlim`, `ylim`, `zlim`: Plot limits
- `title`: Plot title
"""
function plot_wake_trajectories_3d(collector::WakeTrajectoryCollector;
                                  color_by::Symbol=:time,
                                  alpha::Float64=0.7,
                                  linewidth::Float64=1.0,
                                  rotor_centers=nothing,
                                  rotor_radius=nothing,
                                  xlim=nothing, ylim=nothing, zlim=nothing,
                                  title="Wake Trajectories",
                                  figsize=(12, 8))
    
    if isempty(collector.trajectories)
        @warn "No trajectory data to plot"
        return nothing
    end
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot trajectories
    for (particle_id, trajectory) in collector.trajectories
        if length(trajectory) < 2
            continue  # Skip single-point trajectories
        end
        
        # Extract coordinates
        xs = [pos[1] for pos in trajectory]
        ys = [pos[2] for pos in trajectory]
        zs = [pos[3] for pos in trajectory]
        
        # Determine color based on coloring scheme
        if color_by == :time
            birth_time = collector.particle_birth_times[particle_id]
            color_val = birth_time / maximum(values(collector.particle_birth_times))
            color = plt.cm.plasma(color_val)
        elseif color_by == :strength
            avg_strength = mean(collector.particle_strengths[particle_id])
            max_strength = maximum([mean(s) for s in values(collector.particle_strengths)])
            color_val = avg_strength / max_strength
            color = plt.cm.viridis(color_val)
        elseif color_by == :age
            birth_time = collector.particle_birth_times[particle_id]
            age = collector.time_history[end] - birth_time
            max_age = maximum([collector.time_history[end] - bt for bt in values(collector.particle_birth_times)])
            color_val = age / max_age
            color = plt.cm.cool(color_val)
        else
            color = "blue"
        end
        
        ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=linewidth)
    end
    
    # Add rotor reference if provided
    if rotor_centers !== nothing && rotor_radius !== nothing
        theta = range(0, 2π, length=50)
        for center in rotor_centers
            # Draw rotor disc in YZ plane (assuming X is axial direction)
            y_circle = center[2] .+ rotor_radius * cos.(theta)
            z_circle = center[3] .+ rotor_radius * sin.(theta)
            x_circle = fill(center[1], length(theta))
            
            ax.plot(x_circle, y_circle, z_circle, "k--", linewidth=2, alpha=0.8)
        end
    end
    
    # Set labels and limits
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    
    if xlim !== nothing; ax.set_xlim(xlim); end
    if ylim !== nothing; ax.set_ylim(ylim); end
    if zlim !== nothing; ax.set_zlim(zlim); end
    
    # Add colorbar
    if color_by in [:time, :strength, :age]
        mappable = plt.cm.ScalarMappable(cmap=color_by == :time ? plt.cm.plasma : 
                                            color_by == :strength ? plt.cm.viridis : plt.cm.cool)
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5)
        cbar.set_label(string(color_by))
    end
    
    plt.tight_layout()
    return fig, ax
end

"""
    save_trajectories_csv(collector, filename)

Save trajectory data to CSV files for external analysis.
"""
function save_trajectories_csv(collector::WakeTrajectoryCollector, filename::String)
    if isempty(collector.trajectories)
        @warn "No trajectory data to save"
        return
    end
    
    # Create output directory if needed
    dir = dirname(filename)
    if !isdir(dir) && dir != ""
        mkpath(dir)
    end
    
    open(filename, "w") do file
        println(file, "particle_id,time_step,x,y,z,strength,age")
        
        for (particle_id, trajectory) in collector.trajectories
            birth_time = collector.particle_birth_times[particle_id]
            strengths = collector.particle_strengths[particle_id]
            
            for (i, pos) in enumerate(trajectory)
                if i <= length(strengths) && i <= length(collector.time_history)
                    current_time = collector.time_history[i]
                    age = current_time - birth_time
                    strength = strengths[i]
                    
                    println(file, "$particle_id,$i,$(pos[1]),$(pos[2]),$(pos[3]),$strength,$age")
                end
            end
        end
    end
    
    println("Trajectory data saved to $filename")
end

"""
    filter_trajectories_by_region(collector, region_func)

Filter trajectories to keep only particles that pass through a specified region.
`region_func` should accept (x, y, z) coordinates and return a boolean.
"""
function filter_trajectories_by_region(collector::WakeTrajectoryCollector, region_func)
    filtered_trajectories = Dict{Int, Vector{Vector{Float64}}}()
    filtered_birth_times = Dict{Int, Float64}()
    filtered_strengths = Dict{Int, Vector{Float64}}()
    
    for (particle_id, trajectory) in collector.trajectories
        # Check if any point in trajectory passes through region
        if any(region_func(pos...) for pos in trajectory)
            filtered_trajectories[particle_id] = trajectory
            filtered_birth_times[particle_id] = collector.particle_birth_times[particle_id]
            filtered_strengths[particle_id] = collector.particle_strengths[particle_id]
        end
    end
    
    # Create new collector with filtered data
    filtered_collector = WakeTrajectoryCollector(
        max_age=collector.max_age,
        min_strength=collector.min_strength,
        sample_frequency=collector.sample_frequency
    )
    
    filtered_collector.trajectories = filtered_trajectories
    filtered_collector.particle_birth_times = filtered_birth_times
    filtered_collector.particle_strengths = filtered_strengths
    filtered_collector.time_history = copy(collector.time_history)
    
    return filtered_collector
end