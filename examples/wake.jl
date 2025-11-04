#=##############################################################################
# DESCRIPTION
    Load and visualize wake trajectories from saved VPM particle field data
    
    Use this script to post-process existing simulation results and extract
    wake trajectories from saved particle field files.
    
# AUTHORSHIP
  * Created   : 2025-01-09
  * License   : MIT
=###############################################################################

import FLOWUnsteady as uns
import FLOWVPM as vpm
import PyPlot as plt
using HDF5
using LinearAlgebra

# Load wake trajectory utilities
include("wake_trajectory_utils.jl")

"""
    load_wake_trajectories_from_hdf5(data_path, run_name, start_step, end_step;
                                     step_increment=1, min_strength=1e-6)

Load particle trajectories from saved HDF5 particle field files using spatial tracking.
This method tracks particles based on spatial proximity rather than index matching.

# Arguments
- `data_path`: Directory containing the simulation results
- `run_name`: Name of the simulation run
- `start_step`: First time step to load
- `end_step`: Last time step to load
- `step_increment`: Load every Nth time step
- `min_strength`: Minimum vortex strength to include
"""
function load_wake_trajectories_from_hdf5(data_path::String, run_name::String, 
                                          start_step::Int, end_step::Int;
                                          step_increment::Int=1, 
                                          min_strength::Real=1e-6)
    
    println("Loading wake trajectories from saved data using spatial tracking...")
    println("  Path: $data_path")
    println("  Run: $run_name")
    println("  Steps: $start_step to $end_step (every $step_increment)")
    
    trajectories = Vector{Vector{Vector{Float64}}}()  # Array of trajectories, each trajectory is array of positions
    trajectory_strengths = Vector{Vector{Float64}}()  # Strength history for each trajectory
    trajectory_birth_times = Vector{Float64}()        # Birth time for each trajectory
    time_history = Float64[]
    
    steps_to_load = start_step:step_increment:end_step
    max_connection_distance = 0.02  # Maximum distance to connect particles between time steps (2cm)
    
    for (time_idx, step) in enumerate(steps_to_load)
        
        # Construct filename for this time step
        pfield_file = joinpath(data_path, "$(run_name)_pfield.$(step).h5")
        
        if !isfile(pfield_file)
            @warn "File not found: $pfield_file"
            continue
        end
        
        current_positions = Vector{Vector{Float64}}()
        current_strengths = Vector{Float64}()
        current_time = Float64(step)
        
        try
            h5open(pfield_file, "r") do file
                # Read particle positions and circulation
                if haskey(file, "X") && haskey(file, "Gamma")
                    positions = read(file, "X")  # Shape: [3, np]
                    circulations = read(file, "Gamma")  # Shape: [3, np]
                    
                    # Read time if available
                    current_time = haskey(file, "t") ? read(file, "t") : Float64(step)
                    
                    np = size(positions, 2)
                    
                    for i in 1:np
                        # Extract position and circulation
                        pos = [positions[1,i], positions[2,i], positions[3,i]]
                        gamma = [circulations[1,i], circulations[2,i], circulations[3,i]]
                        gamma_magnitude = norm(gamma)
                        
                        # Skip weak particles
                        if gamma_magnitude >= min_strength
                            push!(current_positions, pos)
                            push!(current_strengths, gamma_magnitude)
                        end
                    end
                end
            end
            
            push!(time_history, current_time)
            
            if time_idx == 1
                # First time step: create new trajectories for all particles
                for (i, pos) in enumerate(current_positions)
                    push!(trajectories, [pos])
                    push!(trajectory_strengths, [current_strengths[i]])
                    push!(trajectory_birth_times, current_time)
                end
                
            else
                # Subsequent time steps: connect particles to existing trajectories
                trajectory_updated = fill(false, length(trajectories))
                
                for (pos_idx, pos) in enumerate(current_positions)
                    min_distance = Inf
                    best_traj_idx = 0
                    
                    # Find the closest trajectory endpoint
                    for (traj_idx, traj) in enumerate(trajectories)
                        if !trajectory_updated[traj_idx] && !isempty(traj)
                            last_pos = traj[end]
                            distance = norm(pos - last_pos)
                            
                            if distance < min_distance && distance < max_connection_distance
                                min_distance = distance
                                best_traj_idx = traj_idx
                            end
                        end
                    end
                    
                    if best_traj_idx > 0
                        # Connect to existing trajectory
                        push!(trajectories[best_traj_idx], pos)
                        push!(trajectory_strengths[best_traj_idx], current_strengths[pos_idx])
                        trajectory_updated[best_traj_idx] = true
                    else
                        # Create new trajectory
                        push!(trajectories, [pos])
                        push!(trajectory_strengths, [current_strengths[pos_idx]])
                        push!(trajectory_birth_times, current_time)
                    end
                end
            end
            
            if time_idx % 10 == 0
                active_trajectories = sum(length(traj) > 1 for traj in trajectories)
                println("  Loaded step $step ($active_trajectories active trajectories)")
            end
            
        catch e
            @warn "Error reading file $pfield_file: $e"
        end
    end
    
    # Filter out trajectories that are too short
    min_trajectory_length = 3
    valid_indices = [i for (i, traj) in enumerate(trajectories) if length(traj) >= min_trajectory_length]
    
    trajectories = trajectories[valid_indices]
    trajectory_strengths = trajectory_strengths[valid_indices]
    trajectory_birth_times = trajectory_birth_times[valid_indices]
    
    # Convert to WakeTrajectoryCollector format
    collector_trajectories = Dict{Int, Vector{Vector{Float64}}}()
    collector_strengths = Dict{Int, Vector{Float64}}()
    collector_birth_times = Dict{Int, Float64}()
    
    for (i, traj) in enumerate(trajectories)
        collector_trajectories[i] = traj
        collector_strengths[i] = trajectory_strengths[i]
        collector_birth_times[i] = trajectory_birth_times[i]
    end
    
    # Create collector object with loaded data
    collector = WakeTrajectoryCollector(min_strength=min_strength)
    collector.trajectories = collector_trajectories
    collector.particle_birth_times = collector_birth_times
    collector.particle_strengths = collector_strengths
    collector.time_history = time_history
    
    println("Loaded $(length(trajectories)) valid particle trajectories using spatial tracking")
    if !isempty(time_history)
        println("Time range: $(round(minimum(time_history), digits=3)) - $(round(maximum(time_history), digits=3)) s")
    end
    
    return collector
end

"""
    analyze_tip_vortex_decay(collector, rotor_centers, rotor_radius;
                             downstream_range=(0.1, 5.0), num_bins=50)

Analyze tip vortex decay by tracking peak vorticity vs downstream distance (Z/R).
Returns downstream distances and peak vorticity values for each rotor.
"""
function analyze_tip_vortex_decay(collector, rotor_centers, rotor_radius;
                                 downstream_range=(0.1, 5.0), num_bins=50,
                                 tip_region_fraction=0.8)

    println("Analyzing tip vortex decay for $(length(rotor_centers)) rotors...")

    # Create downstream distance bins
    z_r_min, z_r_max = downstream_range
    z_r_bins = range(z_r_min, z_r_max, length=num_bins+1)
    z_r_centers = [(z_r_bins[i] + z_r_bins[i+1])/2 for i in 1:num_bins]

    # Results for each rotor
    rotor_results = Dict()

    for (rotor_idx, rotor_center) in enumerate(rotor_centers)
        peak_vorticities = Float64[]
        valid_z_r = Float64[]

        println("Processing rotor $rotor_idx at position $(rotor_center)...")

        for bin_idx in 1:num_bins
            z_r_low = z_r_bins[bin_idx]
            z_r_high = z_r_bins[bin_idx+1]
            z_r_center = z_r_centers[bin_idx]

            # Find particles in this downstream distance range for this rotor
            bin_particles = []

            for (particle_id, trajectory) in collector.trajectories
                strengths = collector.particle_strengths[particle_id]

                for (pos_idx, pos) in enumerate(trajectory)
                    # Calculate downstream distance from rotor center
                    dx = pos[1] - rotor_center[1]  # Assuming downstream is +X direction
                    dy = pos[2] - rotor_center[2]
                    dz = pos[3] - rotor_center[3]

                    # Calculate radial distance from rotor axis (in YZ plane)
                    radial_distance = sqrt(dy^2 + dz^2)

                    # Only consider tip region particles
                    if radial_distance >= tip_region_fraction * rotor_radius
                        # Calculate non-dimensional downstream distance Z/R
                        downstream_distance = dx / rotor_radius

                        if z_r_low <= downstream_distance <= z_r_high
                            if pos_idx <= length(strengths)
                                push!(bin_particles, strengths[pos_idx])
                            end
                        end
                    end
                end
            end

            # Find peak vorticity in this bin
            if !isempty(bin_particles)
                peak_vorticity = maximum(bin_particles)
                push!(peak_vorticities, peak_vorticity)
                push!(valid_z_r, z_r_center)
            end
        end

        rotor_results[rotor_idx] = Dict(
            "z_r" => valid_z_r,
            "peak_vorticity" => peak_vorticities,
            "center" => rotor_center
        )

        println("  Found $(length(valid_z_r)) valid downstream bins for rotor $rotor_idx")
    end

    return rotor_results
end

"""
    track_tip_vortex_trajectories(collector, rotor_centers, rotor_radius;
                                 tip_region_min=0.7, tip_region_max=1.2)

Track tip vortex trajectories and analyze their r/R vs z/R patterns.
"""
function track_tip_vortex_trajectories(collector, rotor_centers, rotor_radius;
                                      tip_region_min=0.7, tip_region_max=1.2,
                                      min_strength=1e-5)

    println("Tracking tip vortex trajectories for $(length(rotor_centers)) rotors...")

    rotor_tip_data = Dict()

    for (rotor_idx, rotor_center) in enumerate(rotor_centers)
        println("Processing rotor $rotor_idx at position $(rotor_center)...")

        tip_trajectories = []

        for (particle_id, trajectory) in collector.trajectories
            strengths = collector.particle_strengths[particle_id]

            # Filter trajectory points that are in tip region
            tip_points = []
            tip_strengths_filtered = []

            for (pos_idx, pos) in enumerate(trajectory)
                if pos_idx <= length(strengths)
                    # Calculate position relative to rotor center
                    dx = pos[1] - rotor_center[1]  # Downstream distance
                    dy = pos[2] - rotor_center[2]
                    dz = pos[3] - rotor_center[3]

                    # Calculate radial distance from rotor axis
                    radial_distance = sqrt(dy^2 + dz^2)
                    r_over_R = radial_distance / rotor_radius
                    z_over_R = dx / rotor_radius

                    # Check if in tip region and has sufficient strength
                    strength = strengths[pos_idx]
                    if (tip_region_min <= r_over_R <= tip_region_max &&
                        strength >= min_strength && z_over_R >= 0)

                        push!(tip_points, [r_over_R, z_over_R, strength])
                        push!(tip_strengths_filtered, strength)
                    end
                end
            end

            # Only keep trajectories with sufficient tip points
            if length(tip_points) >= 3
                push!(tip_trajectories, tip_points)
            end
        end

        rotor_tip_data[rotor_idx] = Dict(
            "trajectories" => tip_trajectories,
            "center" => rotor_center,
            "total_trajectories" => length(tip_trajectories)
        )

        println("  Found $(length(tip_trajectories)) tip vortex trajectories for rotor $rotor_idx")
    end

    return rotor_tip_data
end

"""
    plot_tip_vortex_rz_comparison(tip_data, rotor_radius; save_path=nothing)

Plot r/R vs z/R trajectories for tip vortices from multiple rotors.
"""
function plot_tip_vortex_rz_comparison(tip_data, rotor_radius;
                                      save_path=nothing,
                                      rotor_labels=nothing,
                                      max_trajectories_per_rotor=50)

    println("Creating r/R vs z/R tip vortex trajectory plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ["blue", "red", "green", "orange", "purple"]
    alphas = [0.6, 0.6, 0.6, 0.6, 0.6]

    if rotor_labels === nothing
        rotor_labels = ["Rotor $i" for i in keys(tip_data)]
    end

    # Plot trajectories for each rotor
    for (rotor_idx, data) in tip_data
        trajectories = data["trajectories"]

        if !isempty(trajectories)
            color_idx = ((rotor_idx - 1) % length(colors)) + 1
            color = colors[color_idx]
            alpha = alphas[color_idx]

            # Limit number of trajectories to plot for clarity
            n_plot = min(length(trajectories), max_trajectories_per_rotor)

            for (traj_idx, trajectory) in enumerate(trajectories[1:n_plot])
                r_over_R = [point[1] for point in trajectory]
                z_over_R = [point[2] for point in trajectory]
                strengths = [point[3] for point in trajectory]

                # Plot trajectory colored by vorticity strength
                if traj_idx == 1
                    # Add label only for first trajectory of each rotor
                    ax.plot(z_over_R, r_over_R, color=color, alpha=alpha,
                           linewidth=1.5, label=rotor_labels[rotor_idx])
                else
                    ax.plot(z_over_R, r_over_R, color=color, alpha=alpha,
                           linewidth=1.0)
                end
            end

            println("Plotted $(n_plot) trajectories for $(rotor_labels[rotor_idx])")
        end
    end

    # Add reference lines
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, linewidth=1,
               label="Rotor Tip (r/R = 1.0)")
    ax.axhline(y=0.7, color="gray", linestyle=":", alpha=0.3, linewidth=1)
    ax.axhline(y=1.2, color="gray", linestyle=":", alpha=0.3, linewidth=1)

    ax.set_xlabel("Downstream Distance (z/R)", fontsize=14)
    ax.set_ylabel("Radial Distance (r/R)", fontsize=14)
    ax.set_title("Tip Vortex Trajectories: Radial vs Downstream Distance", fontsize=16, fontweight="bold")
    ax.grid(true, alpha=0.3)
    ax.legend(fontsize=12, loc="upper right")

    # Set reasonable axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 1.5)

    plt.tight_layout()

    if save_path !== nothing
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        println("Saved tip vortex r/R vs z/R plot: $save_path")
    end

    return fig, ax
end

"""
    plot_vortex_decay_comparison(rotor_results, rotor_radius; save_path=nothing)

Plot tip vortex decay curves for multiple rotors in comparison.
"""
function plot_vortex_decay_comparison(rotor_results, rotor_radius;
                                    save_path=nothing,
                                    rotor_labels=nothing)

    println("Creating vortex decay comparison plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["blue", "red", "green", "orange", "purple"]
    markers = ["o", "s", "^", "D", "v"]

    if rotor_labels === nothing
        rotor_labels = ["Rotor $i" for i in keys(rotor_results)]
    end

    # Plot decay curves for each rotor
    for (rotor_idx, results) in rotor_results
        z_r = results["z_r"]
        peak_vorticity = results["peak_vorticity"]

        if !isempty(z_r)
            color_idx = ((rotor_idx - 1) % length(colors)) + 1
            marker_idx = ((rotor_idx - 1) % length(markers)) + 1

            ax.semilogy(z_r, peak_vorticity,
                       color=colors[color_idx],
                       marker=markers[marker_idx],
                       linewidth=2, markersize=8, alpha=0.8,
                       label=rotor_labels[rotor_idx])

            # Calculate decay rate (slope in log space)
            if length(z_r) >= 3
                # Fit exponential decay: γ = γ₀ * exp(-α * Z/R)
                # ln(γ) = ln(γ₀) - α * (Z/R)
                log_vorticity = log.(peak_vorticity)

                # Simple linear regression for decay rate
                n = length(z_r)
                sum_x = sum(z_r)
                sum_y = sum(log_vorticity)
                sum_xy = sum(z_r .* log_vorticity)
                sum_x2 = sum(z_r.^2)

                decay_rate = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)

                println("Rotor $rotor_idx decay rate: α = $(round(-decay_rate, digits=3)) (1/R)")
            end
        end
    end

    ax.set_xlabel("Downstream Distance (Z/R)", fontsize=14)
    ax.set_ylabel("Peak Vorticity (1/s)", fontsize=14)
    ax.set_title("Tip Vortex Decay Comparison", fontsize=16, fontweight="bold")
    ax.grid(true, alpha=0.3)
    ax.legend(fontsize=12)

    # Set reasonable axis limits
    ax.set_xlim(0, maximum([maximum(results["z_r"]) for results in values(rotor_results) if !isempty(results["z_r"])]))

    plt.tight_layout()

    if save_path !== nothing
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        println("Saved vortex decay plot: $save_path")
    end

    return fig, ax
end

"""
    plot_wake_from_simulation_data(data_path, run_name;
                                  start_step=1, end_step=100,
                                  rotor_radius=0.14, rotor_centers=nothing)

Complete function to load and plot wake trajectories from saved simulation data.
"""
function plot_wake_from_simulation_data(data_path::String, run_name::String;
                                       start_step::Int=1,
                                       end_step::Int=100,
                                       step_increment::Int=2,
                                       rotor_radius::Real=0.14,
                                       rotor_centers=nothing,
                                       min_strength::Real=1e-6,
                                       save_plots::Bool=true)
    
    # Load trajectory data
    collector = load_wake_trajectories_from_hdf5(data_path, run_name, 
                                                start_step, end_step;
                                                step_increment=step_increment,
                                                min_strength=min_strength)
    
    if isempty(collector.trajectories)
        @error "No trajectory data loaded. Check file paths and parameters."
        return nothing
    end
    
    # Default rotor centers if not provided
    if rotor_centers === nothing
        rotor_centers = [[0.0, 0.0, 0.0], [-0.1343, 0.0, 0.0]]
    end
    
    # Skip 3D visualization - Focus on tip vortex analysis
    println("Skipping 3D visualizations, focusing on tip vortex tracking...")
    
    # Save trajectory data
    if save_plots
        csv_file = joinpath(data_path, "$(run_name)_wake_trajectories_postprocessed.csv")
        save_trajectories_csv(collector, csv_file)
    end

    # === NEW: Tip Vortex Trajectory Analysis ===
    println("\nPerforming tip vortex trajectory analysis...")

    # Track tip vortex trajectories
    tip_data = track_tip_vortex_trajectories(collector, rotor_centers, rotor_radius;
                                           tip_region_min=0.7,
                                           tip_region_max=1.2,
                                           min_strength=1e-5)

    # Create r/R vs z/R trajectory plot
    rotor_labels = ["Upper Rotor", "Lower Rotor"]  # Customize as needed
    if length(rotor_centers) > 2
        rotor_labels = ["Rotor $i" for i in 1:length(rotor_centers)]
    end

    trajectory_plot_file = joinpath(data_path, "$(run_name)_tip_vortex_rz_trajectories.png")
    fig1, ax1 = plot_tip_vortex_rz_comparison(tip_data, rotor_radius;
                                            save_path=save_plots ? trajectory_plot_file : nothing,
                                            rotor_labels=rotor_labels,
                                            max_trajectories_per_rotor=30)

    # === Vortex Decay Analysis ===
    println("\nPerforming vortex strength decay analysis...")

    # Analyze tip vortex decay for each rotor
    decay_results = analyze_tip_vortex_decay(collector, rotor_centers, rotor_radius;
                                           downstream_range=(0.5, 8.0),
                                           num_bins=30,
                                           tip_region_fraction=0.7)

    # Create vortex decay comparison plot
    decay_plot_file = joinpath(data_path, "$(run_name)_vortex_decay_comparison.png")
    fig2, ax2 = plot_vortex_decay_comparison(decay_results, rotor_radius;
                                           save_path=save_plots ? decay_plot_file : nothing,
                                           rotor_labels=rotor_labels)
    
    # Display summary
    println("\nWake Trajectory Post-Processing Summary:")
    println("="^50)
    println("Simulation: $run_name")
    println("Data path: $data_path") 
    println("Time steps processed: $start_step to $end_step (every $step_increment)")
    println("Total trajectories: $(length(collector.trajectories))")
    
    trajectory_lengths = [length(traj) for traj in values(collector.trajectories)]
    println("Average trajectory length: $(round(mean(trajectory_lengths), digits=1)) points")
    println("Max trajectory length: $(maximum(trajectory_lengths)) points")
    println("Min trajectory length: $(minimum(trajectory_lengths)) points")
    
    if !isempty(collector.time_history)
        println("Time range: $(round(minimum(collector.time_history), digits=3)) - $(round(maximum(collector.time_history), digits=3)) s")
    end

    # Print tip vortex trajectory analysis summary
    println("\nTip Vortex Trajectory Analysis Summary:")
    println("="^40)
    for (rotor_idx, data) in tip_data
        trajectories = data["trajectories"]
        if !isempty(trajectories)
            # Calculate statistics for r/R and z/R ranges
            all_r_over_R = []
            all_z_over_R = []
            for traj in trajectories
                append!(all_r_over_R, [point[1] for point in traj])
                append!(all_z_over_R, [point[2] for point in traj])
            end

            println("$(rotor_labels[rotor_idx]):")
            println("  Total tip trajectories tracked: $(length(trajectories))")
            println("  r/R range: $(round(minimum(all_r_over_R), digits=2)) - $(round(maximum(all_r_over_R), digits=2))")
            println("  z/R range: $(round(minimum(all_z_over_R), digits=1)) - $(round(maximum(all_z_over_R), digits=1))")
        end
    end

    # Print vortex decay analysis summary
    println("\nVortex Decay Analysis Summary:")
    println("="^30)
    for (rotor_idx, results) in decay_results
        if !isempty(results["z_r"])
            initial_vorticity = results["peak_vorticity"][1]
            final_vorticity = results["peak_vorticity"][end]
            decay_ratio = final_vorticity / initial_vorticity

            println("$(rotor_labels[rotor_idx]):")
            println("  Downstream range analyzed: $(round(results["z_r"][1], digits=1)) - $(round(results["z_r"][end], digits=1)) Z/R")
            println("  Initial peak vorticity: $(round(initial_vorticity, digits=4)) (1/s)")
            println("  Final peak vorticity: $(round(final_vorticity, digits=4)) (1/s)")
            println("  Vorticity decay ratio: $(round(decay_ratio, digits=3))")
        end
    end

    return collector, (fig1, fig2), tip_data, decay_results
end

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

# Remove this condition to allow execution via include()
# if abspath(PROGRAM_FILE) == @__FILE__
if true
    
    println("Wake Trajectory Post-Processing Example")
    println("="^50)
    
    # Modify these parameters for your simulation
    data_path = "/home/kweon/V_stacked_90_mid_high_monitor2"  # Path to your simulation results
    run_name = "V_stacked_90_mid_high_monitor2"              # Name of your simulation
    
    # Check if data exists
    if !isdir(data_path)
        @error "Data path not found: $data_path"
        @error "Please modify the data_path variable to point to your simulation results"
        exit(1)
    end
    
    # Parameters for loading data
    start_step = 720    # Start after some initial transient
    end_step = 2880     # Or use the last available step
    step_increment = 8 # Load every 5th step for faster processing
    
    # Rotor parameters (adjust to match your simulation)
    R = 0.14  # Rotor radius in meters
    rotor_centers = [[0.0, 0.0, 0.0], [-0.005, 0.0, 0.0]]  # Upper and lower rotor positions
    
    # Load and plot trajectories
    try
        collector, figs = plot_wake_from_simulation_data(
            data_path, run_name;
            start_step=start_step,
            end_step=end_step,
            step_increment=step_increment,
            rotor_radius=R,
            rotor_centers=rotor_centers,
            min_strength=1e-6,
            save_plots=true
        )
        
        println("\nPost-processing completed successfully!")
        println("Check the output directory for visualization plots and CSV data.")
        
    catch e
        @error "Error during post-processing: $e"
        @error "Make sure the simulation data files exist and paths are correct"
    end
    
end