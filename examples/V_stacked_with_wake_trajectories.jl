#=##############################################################################
# DESCRIPTION
    Modified V_stacked.jl example with wake trajectory collection and visualization
    
    This example shows how to extract and plot wake trajectories from your
    VPM simulation results.
    
# AUTHORSHIP
  * Based on V_stacked.jl example
  * Wake trajectory additions : 2025-01-09
  * License   : MIT
=###############################################################################

import FLOWUnsteady as uns
import FLOWVLM as vlm
import FLOWVPM as vpm
using DelimitedFiles

# Load wake trajectory utilities
include("wake_trajectory_utils.jl")

run_name        = "sunwoong_test_wake_trajectories"
save_path       = run_name
paraview        = true

# ----------------- GEOMETRY PARAMETERS ----------------------------------------
rotor_file      = "3blade.csv"
data_path       = uns.def_data_path
pitchs          = [6.25, 8.25]
x_arr           = [0.0, -0.1343]
y_arr           = [0.0, 0.0]
z_arr           = [0.0, 0.0]
nrotors         = length(pitchs)
CWs             = [false, true]
xfoil           = false
read_polar      = vlm.ap.read_polar2

# Discretization
n               = 20
r               = 1/10

# Read radius of this rotor and number of blades
R, B            = uns.read_rotor(rotor_file; data_path=data_path)[[1,3]]

# ----------------- SIMULATION PARAMETERS --------------------------------------
RPM1            = 1500
RPM2            = 1500
RPM             = RPM1 = RPM2
J               = 0.0001
AOA             = 0

rho             = 1.225
mu              = 1.81e-5
speedofsound    = 343

magVinf         = J*RPM/60*(2*R)
Vinf(X, t)      = magVinf*[cos(AOA*pi/180), sin(AOA*pi/180), 0]

ReD             = 2*pi*RPM/60*R * rho/mu * 2*R
Matip           = 2*pi*RPM/60 * R / speedofsound

println("""
    RPM:    $(RPM)
    Vinf:   $(Vinf(zeros(3), 0)) m/s
    Matip:  $(round(Matip, digits=3))
    ReD:    $(round(ReD, digits=0))
""")

# ----------------- SOLVER PARAMETERS ------------------------------------------
VehicleType     = uns.UVLMVehicle
const_solution  = VehicleType==uns.QVLMVehicle

# Time parameters - reduced for demonstration
nrevs           = 10                        # Reduced from 20 for faster demo
nsteps_per_rev  = 36
nsteps          = const_solution ? 50 : nrevs*nsteps_per_rev
ttot            = nsteps/nsteps_per_rev / (RPM/60)
dt              = ttot/nsteps

# VPM particle shedding
p_per_step      = 4
shed_starting   = false
shed_unsteady   = true
unsteady_shedcrit = 0.001
max_particles   = ((2*n+1)*B)*nsteps*p_per_step + 1

# Regularization
sigma_rotor_surf= R/50
sigma_rotor_self= R/3
lambda_vpm      = 2.125
sigma_vpm_overwrite = lambda_vpm * 2*pi*R/(nsteps_per_rev*p_per_step)
sigmafactor_vpmonvlm= 1

# Rotor solvers
vlm_rlx         = 0.5
hubtiploss_correction = ((0.6, 5, 0.1, 0.05),(2, 1, 1, 0.05))

# VPM solver
vpm_integration = vpm.euler
vpm_viscous     = vpm.Inviscid()
vpm_SFS         = vpm.SFS_none

if VehicleType == uns.QVLMVehicle
    uns.vlm.VLMSolver._mute_warning(true)
end

# ----------------- WAKE TREATMENT ---------------------------------------------
suppress_fountain   = true
no_shedding_Rthreshold = suppress_fountain ? 0.35 : 0.0
no_shedding_nstepsthreshold = 3*nsteps_per_rev
omit_shedding = []

function wake_treatment_supress(sim, args...; optargs...)
    if sim.nt == 1
        for rotor in [rotor1, rotor2]
            for i in 1:vlm.get_m(rotor)
                HS = vlm.getHorseshoe(rotor, i)
                CP = HS[5]
                if uns.vlm.norm(CP - vlm._get_O(rotor)) <= no_shedding_Rthreshold*R
                    push!(omit_shedding, i)
                end
            end
        end
    end

    if sim.nt == no_shedding_nstepsthreshold
        omit_shedding .= -1
    end

    return false
end

# Wake treatment by strength and size
rmv_strngth = 2*2/p_per_step * dt/(30/(RPM*2))
minmaxGamma = rmv_strngth*[0.0001, 0.05]
wake_treatment_strength = uns.remove_particles_strength( minmaxGamma[1]^2, minmaxGamma[2]^2; every_nsteps=1)

minmaxsigma = sigma_vpm_overwrite*[0.1, 5]
wake_treatment_sigma = uns.remove_particles_sigma( minmaxsigma[1], minmaxsigma[2]; every_nsteps=1)

# Wake treatment by distance
Pstart = [0.2, 0.0, 0.0]
Pend   = [-5R, 0.0, 0.0]
Rcyl   = 1.8R
step   = 1
wake_treatment_cyl = uns.remove_particles_cylinder(Pstart, Pend, Rcyl, step)

# Concatenate all wake treatments
wake_treatment = uns.concatenate(wake_treatment_cyl, wake_treatment_strength, wake_treatment_sigma)

# ----------------- WAKE TRAJECTORY SETUP -------------------------------------
println("Setting up wake trajectory collection...")

# Create wake trajectory collector
# Parameters:
# - max_age: Track particles for up to 3 rotor revolutions
# - min_strength: Minimum vortex strength to track 
# - sample_frequency: Collect data every 2 time steps (for performance)
max_trajectory_age = 3.0 * (60.0/RPM)  # 3 revolutions in seconds
min_vortex_strength = rmv_strngth * 0.01  # 1% of reference strength

# Define region filter to focus on near-wake region
wake_region_filter(x, y, z) = (
    x >= -3*R &&           # Don't track too far downstream
    x <= 2*R &&            # Don't track too far upstream  
    sqrt(y^2 + z^2) <= 2*R # Focus on particles within 2R of centerline
)

collector, wake_monitor = create_wake_monitor(
    max_age=max_trajectory_age,
    min_strength=min_vortex_strength,
    sample_frequency=2,
    filter_region=wake_region_filter,
    max_particles=500  # Limit to prevent memory issues
)

println("Wake trajectory collector configured:")
println("  Max age: $(max_trajectory_age) seconds")
println("  Min strength: $(min_vortex_strength)")
println("  Sample frequency: every 2 steps")
println("  Max particles: 500")

# ----------------- 1) VEHICLE DEFINITION --------------------------------------
println("Generating geometry...")

# Rotor1
ri = 1
rotor1 = uns.generate_rotor(rotor_file; pitch=pitchs[ri],
                        n=n, CW=CWs[ri], blade_r=r,
                        altReD=[RPM, J, mu/rho],
                        xfoil=xfoil,
                        read_polar=read_polar,
                        data_path=data_path,
                        verbose=true,
                        verbose_xfoil=false,
                        plot_disc=true
                        );
y = y_arr[ri]
x = x_arr[ri]
z = z_arr[ri]
O = [x, y, z]
Oaxis = uns.gt.rotation_matrix2(0, 0, 0)
vlm.setcoordsystem(rotor1, O, Oaxis)

# Rotor2
ri = 2
rotor2 = uns.generate_rotor(rotor_file; pitch=pitchs[ri],
                        n=n, CW=CWs[ri], blade_r=r,
                        altReD=[RPM, J, mu/rho],
                        xfoil=xfoil,
                        read_polar=read_polar,
                        data_path=data_path,
                        verbose=true,
                        verbose_xfoil=false,
                        plot_disc=true
                        );
y = y_arr[ri]
x = x_arr[ri]
z = z_arr[ri]
O = [x, y, z]
Oaxis = uns.gt.rotation_matrix2(0, 0, 0)
vlm.setcoordsystem(rotor2, O, Oaxis)

# Store rotor centers for visualization
rotor_centers = [x_arr[i], y_arr[i], z_arr[i]] for i in 1:nrotors]

rotors1 = vlm.Rotor[]
rotors2 = vlm.Rotor[]
push!(rotors1, rotor1)
push!(rotors2, rotor2)

println("Generating vehicle...")

# Generate vehicle
system = vlm.WingSystem()
vlm.addwing(system, "rotor1", rotor1)
vlm.addwing(system, "rotor2", rotor2)

rotor_systems = ([rotor1], [rotor2])

wake_system = vlm.WingSystem()
if VehicleType != uns.QVLMVehicle
    vlm.addwing(wake_system, "rotor1", rotor1)
    vlm.addwing(wake_system, "rotor2", rotor2)
end

vehicle = VehicleType(   system;
                            rotor_systems=rotor_systems,
                            wake_system=wake_system
                         );

# ------------- 2) MANEUVER DEFINITION -----------------------------------------
Vvehicle(t) = zeros(3)
anglevehicle(t) = zeros(3)
RPMcontrol1(t) = 1.0
RPMcontrol2(t) = 1.0

angles = ()
RPMs = (RPMcontrol1, RPMcontrol2)

maneuver = uns.KinematicManeuver(angles, RPMs, Vvehicle, anglevehicle)

# ------------- 3) SIMULATION DEFINITION ---------------------------------------
Vref = 0.0
RPMref = RPM
Vinit = Vref*Vvehicle(0)
Winit = pi/180*(anglevehicle(1e-6) - anglevehicle(0))/(1e-6*ttot)

simulation = uns.Simulation(vehicle, maneuver, Vref, RPMref, ttot;
                                                    Vinit=Vinit, Winit=Winit);

restart_file = nothing

# ------------- 4) MONITORS DEFINITIONS ----------------------------------------
figs, figaxs = [], []

rotors = vcat(rotor_systems...)

# Generate rotor monitor
monitor_rotor = uns.generate_monitor_rotors(rotors, J, rho, RPM, nsteps;
                                            t_scale=RPM/60,
                                            t_lbl="Revolutions",
                                            save_path=save_path,
                                            run_name=run_name,
                                            figname="rotor monitor",
                                            )

# Generate monitor of flow enstrophy
monitor_enstrophy = uns.generate_monitor_enstrophy(;
                                            save_path=save_path,
                                            run_name=run_name,
                                            figname="enstrophy monitor"
                                            )

# Generate monitor of SFS model coefficient Cd
monitor_Cd = uns.generate_monitor_Cd(;
                                            save_path=save_path,
                                            run_name=run_name,
                                            figname="Cd monitor"
                                            )

# Concatenate all monitors including wake trajectory collector
monitors = uns.concatenate(monitor_rotor, monitor_enstrophy, monitor_Cd, wake_monitor)

# ------------- 5) RUN SIMULATION ----------------------------------------------
println("Running simulation with wake trajectory collection...")

# Concatenate monitors and wake treatment procedure into one runtime function
runtime_function = uns.concatenate(monitors, wake_treatment_supress, wake_treatment)

# Run simulation
uns.run_simulation(simulation, nsteps;
                    # ----- SIMULATION OPTIONS -------------
                    Vinf=Vinf,
                    rho=rho, mu=mu, sound_spd=speedofsound,
                    # ----- SOLVERS OPTIONS ----------------
                    p_per_step=p_per_step,
                    max_particles=max_particles,
                    vpm_integration=vpm_integration,
                    vpm_viscous=vpm_viscous,
                    vpm_SFS=vpm_SFS,
                    sigma_vlm_surf=sigma_rotor_surf,
                    sigma_rotor_surf=sigma_rotor_surf,
                    sigma_rotor_self=sigma_rotor_self,
                    sigma_vpm_overwrite=sigma_vpm_overwrite,
                    sigmafactor_vpmonvlm=sigmafactor_vpmonvlm,
                    vlm_rlx=vlm_rlx,
                    hubtiploss_correction=hubtiploss_correction,
                    shed_starting=shed_starting,
                    shed_unsteady=shed_unsteady,
                    unsteady_shedcrit=unsteady_shedcrit,
                    omit_shedding=omit_shedding,
                    extra_runtime_function=runtime_function,
                    # ----- RESTART OPTIONS -----------------
                    restart_vpmfile=restart_file,
                    # ----- OUTPUT OPTIONS ------------------
                    save_path=save_path,
                    run_name=run_name,
                    save_wopwopin=false,
                    save_code=@__FILE__
                    );

# ------------- 6) WAKE TRAJECTORY VISUALIZATION ------------------------------
println("\nProcessing wake trajectory data...")
println("Total trajectories collected: $(length(collector.trajectories))")
println("Time steps sampled: $(length(collector.time_history))")

if !isempty(collector.trajectories)
    
    # Save trajectory data
    trajectory_csv_file = joinpath(save_path, run_name*"_wake_trajectories.csv")
    save_trajectories_csv(collector, trajectory_csv_file)
    
    # Create 3D visualization colored by birth time
    println("Creating wake trajectory plots...")
    
    fig1, ax1 = plot_wake_trajectories_3d(collector;
                                         color_by=:time,
                                         alpha=0.8,
                                         linewidth=1.5,
                                         rotor_centers=rotor_centers,
                                         rotor_radius=R,
                                         title="Wake Trajectories - Colored by Birth Time",
                                         xlim=(-3*R, 2*R),
                                         ylim=(-2*R, 2*R),
                                         zlim=(-2*R, 2*R))
    
    fig1.savefig(joinpath(save_path, run_name*"_wake_trajectories_time.png"), 
                dpi=300, bbox_inches="tight")
    
    # Create visualization colored by vortex strength
    fig2, ax2 = plot_wake_trajectories_3d(collector;
                                         color_by=:strength,
                                         alpha=0.8,
                                         linewidth=1.5,
                                         rotor_centers=rotor_centers,
                                         rotor_radius=R,
                                         title="Wake Trajectories - Colored by Vortex Strength",
                                         xlim=(-3*R, 2*R),
                                         ylim=(-2*R, 2*R),
                                         zlim=(-2*R, 2*R))
    
    fig2.savefig(joinpath(save_path, run_name*"_wake_trajectories_strength.png"), 
                dpi=300, bbox_inches="tight")
    
    # Filter trajectories that pass close to rotor plane for tip vortex analysis
    tip_vortex_region(x, y, z) = (
        abs(x - x_arr[1]) < 0.1*R &&  # Near upper rotor plane
        sqrt(y^2 + z^2) > 0.8*R &&    # Near tip region
        sqrt(y^2 + z^2) < 1.2*R
    )
    
    tip_collector = filter_trajectories_by_region(collector, tip_vortex_region)
    
    if !isempty(tip_collector.trajectories)
        fig3, ax3 = plot_wake_trajectories_3d(tip_collector;
                                             color_by=:age,
                                             alpha=0.9,
                                             linewidth=2.0,
                                             rotor_centers=rotor_centers,
                                             rotor_radius=R,
                                             title="Tip Vortex Trajectories - Colored by Age",
                                             xlim=(-2*R, R),
                                             ylim=(-1.5*R, 1.5*R),
                                             zlim=(-1.5*R, 1.5*R))
        
        fig3.savefig(joinpath(save_path, run_name*"_tip_vortex_trajectories.png"), 
                    dpi=300, bbox_inches="tight")
        
        println("Tip vortex trajectories: $(length(tip_collector.trajectories))")
    end
    
    # Print summary statistics
    println("\nWake Trajectory Analysis Summary:")
    println("="^50)
    println("Total particles tracked: $(length(collector.trajectories))")
    println("Average trajectory length: $(round(mean([length(traj) for traj in values(collector.trajectories)]), digits=1)) points")
    println("Simulation time range: $(round(collector.time_history[1], digits=3)) - $(round(collector.time_history[end], digits=3)) s")
    println("Data saved to: $trajectory_csv_file")
    println("Visualization plots saved in: $save_path")
    
else
    @warn "No wake trajectories were collected. Check simulation parameters."
end

# ----------------- 7) PARAVIEW VISUALIZATION ----------------------------------
if paraview
    println("Calling Paraview...")
    
    files = joinpath(save_path, run_name*"_pfield...xmf;")
    for bi in 1:B
        global files
        files *= run_name*"_Rotor_Blade$(bi)_loft...vtk;"
        files *= run_name*"_Rotor_Blade$(bi)_vlm...vtk;"
    end
    
    run(`paraview --data=$(files)`)
end

println("\nSimulation completed!")
println("Wake trajectory data and plots are available in: $save_path")