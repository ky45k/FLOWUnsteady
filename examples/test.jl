#=##############################################################################
# DESCRIPTION
    Simulation of a DJI 9443 rotor in hover (two-bladed rotor, 9.4 inches
    diameter).

    This example replicates the experiment described in Zawodny & Boyd (2016),
    "Acoustic Characterization and Prediction of Representative,
    Small-scale Rotary-wing Unmanned Aircraft System Components."

# AUTHORSHIP
  * Author          : Eduardo J. Alvarez (edoalvarez.com)
  * Email           : Edo.AlvarezR@gmail.com
  * Created         : Mar 2023
  * Last updated    : Mar 2023
  * License         : MIT
=###############################################################################

#=
Use the following parameters to obtain the desired fidelity

---- MID-LOW FIDELITY ---
n               = 20                        # Number of blade elements per blade
nsteps_per_rev  = 36                        # Time steps per revolution
p_per_step      = 4                         # Sheds per time step
sigma_rotor_surf= R/10                      # Rotor-on-VPM smoothing radius
vpm_integration = vpm.euler                 # VPM time integration scheme
vpm_SFS         = vpm.SFS_none              # VPM LES subfilter-scale model
shed_starting   = false                     # Whether to shed starting vortex
suppress_fountain    = true                 # Suppress hub fountain effect
sigmafactor_vpmonvlm = 1.0                  # Shrink particles by this factor when
                                            #  calculating VPM-on-VLM/Rotor induced velocities

---- MID-HIGH FIDELITY ---
n               = 50
nsteps_per_rev  = 72
p_per_step      = 2
sigma_rotor_surf= R/10
sigmafactor_vpmonvlm = 1.0
shed_starting   = false
suppress_fountain    = true
vpm_integration = vpm.rungekutta3
vpm_SFS         = vpm.SFS_none

---- HIGH FIDELITY -----
n               = 50
nsteps_per_rev  = 360
p_per_step      = 2
sigma_rotor_surf= R/80
sigmafactor_vpmonvlm = 5.5
shed_starting   = true
suppress_fountain    = false
vpm_integration = vpm.rungekutta3
vpm_SFS         = vpm.DynamicSFS(vpm.Estr_fmm, vpm.pseudo3level_positive;
                                    alpha=0.999, maxC=1.0,
                                    clippings=[vpm.clipping_backscatter])
=#

import FLOWUnsteady as uns
import FLOWVLM as vlm
import FLOWVPM as vpm
using CSV, Interpolations,LinearAlgebra

run_name        = "sunwoong_test_1"      # Name of this simulation
save_path       = run_name                  # Where to save this simulation
paraview        = true                      # Whether to visualize with Paraview

# Uncomment this to have the folder named after this file instead
# save_path     = String(split(@__FILE__, ".")[1])
# run_name      = "singlerotor"
# paraview      = false
# ----------------- GEOMETRY PARAMETERS ----------------------------------------

# Rotor geometry
rotor_file      = "3blade.csv"             # Rotor geometry
data_path       = uns.def_data_path         # Path to rotor database
pitchs          = [6.25, 8.25]                       # (deg) collective pitch of blades
x_arr           = [0.0, -0.1343]
y_arr           = [0.0, 0.0]
z_arr           = [0.0, 0.0]
nrotors         = length(pitchs)            # Number of rotors
CWs             = [false, true]                     # Clock-wise rotation
xfoil           = false                     # Whether to run XFOIL
read_polar      = vlm.ap.read_polar2        # What polar reader to use

# NOTE: If `xfoil=true`, XFOIL will be run to generate the airfoil polars used
#       by blade elements before starting the simulation. XFOIL is run
#       on the airfoil contours found in `rotor_file` at the corresponding
#       local Reynolds and Mach numbers along the blade.
#       Alternatively, the user can provide pre-computer airfoil polars using
#       `xfoil=false` and providing the polar files through `rotor_file`.
#       `read_polar` is the function that will be used to parse polar files. Use
#       `vlm.ap.read_polar` for files that are direct outputs of XFOIL (e.g., as
#       downloaded from www.airfoiltools.com). Use `vlm.ap.read_polar2` for CSV
#       files.

# Discretization
n               = 20                        # Number of blade elements per blade
r               = 1/10                      # Geometric expansion of elements

# NOTE: Here a geometric expansion of 1/10 means that the spacing between the
#       tip elements is 1/10 of the spacing between the hub elements. Refine the
#       discretization towards the blade tip like this in order to better
#       resolve the tip vortex.

# Read radius of this rotor and number of blades
R, B            = uns.read_rotor(rotor_file; data_path=data_path)[[1,3]]

# ----------------- SIMULATION PARAMETERS --------------------------------------

# Operating conditions
RPM1            = 1500                      #Upper rotor
RPM2            = 1500                      #Lower rotor
RPM             = RPM1 = RPM2               # Total RPM
J               = 0.0001                    # Advance ratio Vinf/(nD)
AOA             = 0                         # (deg) Angle of attack (incidence angle)

rho             = 1.225                  # (kg/m^3) air density
mu              = 1.81e-5                # (kg/ms) air dynamic viscosity
speedofsound    = 343                    # (m/s) speed of sound

# NOTE: For cases with zero freestream velocity, it is recommended that a
#       negligible small velocity is used instead of zero in order to avoid
#       potential numerical instabilities (hence, J here is negligible small
#       instead of zero)

magVinf         = J*RPM/60*(2*R)
Vinf(X, t)      = magVinf*[cos(AOA*pi/180), sin(AOA*pi/180), 0]  # (m/s) freestream velocity vector

ReD             = 2*pi*RPM/60*R * rho/mu * 2*R      # Diameter-based Reynolds number
Matip           = 2*pi*RPM/60 * R / speedofsound    # Tip Mach number

println("""
    RPM:    $(RPM)
    Vinf:   $(Vinf(zeros(3), 0)) m/s
    Matip:  $(round(Matip, digits=3))
    ReD:    $(round(ReD, digits=0))
""")

# ----------------- SOLVER PARAMETERS ------------------------------------------

# Aerodynamic solver
VehicleType     = uns.UVLMVehicle           # Unsteady solver
# VehicleType     = uns.QVLMVehicle         # Quasi-steady solver
const_solution  = VehicleType==uns.QVLMVehicle  # Whether to assume that the
                                                # solution is constant or not
# Time parameters
nrevs           = 20                        # Number of revolutions in simulation
nsteps_per_rev  = 36                        # Time steps per revolution
nsteps          = const_solution ? 50 : nrevs*nsteps_per_rev # Number of time steps
ttot            = nsteps/nsteps_per_rev / (RPM/60)       # (s) total simulation time
dt              = ttot/nsteps               # (s) time step

# VPM particle shedding
p_per_step      = 4                         # Sheds per time step
shed_starting   = false                     # Whether to shed starting vortex
shed_unsteady   = true                      # Whether to shed vorticity from unsteady loading
unsteady_shedcrit = 0.001                   # Shed unsteady loading whenever circulation
                                            #  fluctuates by more than this ratio
# max_particles   = 50000 # Maximum number of particles
max_particles   = ((2*n+1)*B)*nsteps*p_per_step + 1 # Maximum number of particles

# Regularization
sigma_rotor_surf= R/50                      # Rotor-on-VPM smoothing radius
sigma_rotor_self= R/3                       # Rotor-on-Rotor smoothing radius
lambda_vpm      = 2.125                     # VPM core overlap
                                            # VPM smoothing radius
sigma_vpm_overwrite = lambda_vpm * 2*pi*R/(nsteps_per_rev*p_per_step)
sigmafactor_vpmonvlm= 1                     # Shrink particles by this factor when
                                              #  calculating VPM-on-VLM/Rotor induced velocities

# Rotor solvers
vlm_rlx         = 0.5                       # VLM relaxation <-- this also applied to rotors
hubtiploss_correction = ((0.6, 5, 0.1, 0.05),(2, 1, 1, 0.05))

# VPM solver
vpm_integration = vpm.euler                 # VPM temporal integration scheme
# vpm_integration = vpm.rungekutta3

vpm_viscous     = vpm.Inviscid()            # VPM viscous diffusion scheme
# vpm_viscous   = vpm.CoreSpreading(-1, -1, vpm.zeta_fmm; beta=100.0, itmax=20, tol=1e-1)

vpm_SFS         = vpm.SFS_none    
# vpm_SFS       = vpm.SFS_Cd_twolevel_nobackscatter
# vpm_SFS       = vpm.SFS_Cd_threelevel_nobackscatter
# vpm_SFS       = vpm.DynamicSFS(vpm.Estr_fmm, vpm.pseudo3level_positive;
#                                   alpha=0.999, maxC=1.0,
#                                   clippings=[vpm.clipping_backscatter])
# vpm_SFS       = vpm.DynamicSFS(vpm.Estr_fmm, vpm.pseudo3level_positive;
#                                   alpha=0.999, rlxf=0.005, minC=0, maxC=1
#                                   clippings=[vpm.clipping_backscatter],
#                                   controls=[vpm.control_sigmasensor],
#                                   )

# NOTE: In most practical situations, open rotors operate at a Reynolds number
#       high enough that viscous diffusion in the wake is actually negligible.
#       Hence, it does not make much of a difference whether we run the
#       simulation with viscous diffusion enabled or not. On the other hand,
#       such high Reynolds numbers mean that the wake quickly becomes turbulent
#       and it is crucial to use a subfilter-scale (SFS) model to accurately
#       capture the turbulent decay of the wake (turbulent diffusion).

if VehicleType == uns.QVLMVehicle
    # Mute warnings regarding potential colinear vortex filaments. This is
    # needed since the quasi-steady solver will probe induced velocities at the
    # lifting line of the blade
    uns.vlm.VLMSolver._mute_warning(true)
end





# ----------------- WAKE TREATMENT ---------------------------------------------
# NOTE: It is known in the CFD community that rotor simulations with an
#       impulsive RPM start (*i.e.*, 0 to RPM in the first time step, as opposed
#       to gradually ramping up the RPM) leads to the hub "fountain effect",
#       with the root wake reversing the flow near the hub.
#       The fountain eventually goes away as the wake develops, but this happens
#       very slowly, which delays the convergence of the simulation to a steady
#       state. To accelerate convergence, here we define a wake treatment
#       procedure that suppresses the hub wake for the first three revolutions,
#       avoiding the fountain effect altogether.
#       This is especially helpful in low and mid-fidelity simulations.

suppress_fountain   = false                  # Toggle

# Supress wake shedding on blade elements inboard of this r/R radial station
no_shedding_Rthreshold = suppress_fountain ? 0.0 : 0.0

# Supress wake shedding for this many time steps
no_shedding_nstepsthreshold = 3*nsteps_per_rev

omit_shedding = []          # Index of blade elements to supress wake shedding

# Function to suppress or activate wake shedding
function wake_treatment_supress(sim, args...; optargs...)

    # Case: start of simulation -> suppress shedding
    if sim.nt == 1

        # Identify blade elements on which to suppress shedding
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

    # Case: sufficient time steps -> enable shedding
    if sim.nt == no_shedding_nstepsthreshold

        # Flag to stop suppressing
        omit_shedding .= -1

    end

    return false
end

# ------------- Wake Treatment 추가 ------------------
# Remove by particle strength
# (remove particles neglibly faint, remove blown up)
rmv_strngth = 2*2/p_per_step * dt/(30/(RPM*2))         # Reference strength
minmaxGamma = rmv_strngth*[0.0001, 0.05]                # Strength bounds (removes particles outside of these bounds)
wake_treatment_strength = uns.remove_particles_strength( minmaxGamma[1]^2, minmaxGamma[2]^2; every_nsteps=1)

# Remove by particle size
# (remove particle nearly singular, remove negligibly smeared)
minmaxsigma = sigma_vpm_overwrite*[0.1, 5]              # Size bounds (removes particles outside of these bounds)
wake_treatment_sigma = uns.remove_particles_sigma( minmaxsigma[1], minmaxsigma[2]; every_nsteps=1)

# Remove by distance
Pstart = [0.0, 0.0, 0.0]
Pend   = [5R, 0.0, 0.0]
Rcyl   = 1.8R
step   = 1

wake_treatment_cyl = uns.remove_particles_cylinder(Pstart, Pend, Rcyl, step)

# Concatenate all wake treatments
wake_treatment = uns.concatenate(wake_treatment_cyl, wake_treatment_strength, wake_treatment_sigma)

# # ----------------- Airfoil interpolation --------------------------------------
# const POLAR_DATA_CACHE = Dict{String, Dict{String, Any}}()
# const POLAR_DATA_LOADED = Ref(false)

# function resample_airfoil(x, y, n_points)
#     """
#     에어포일 좌표를 주어진 점 개수로 리샘플링
    
#     Parameters:
#     - x, y: 원본 에어포일 좌표 벡터
#     - n_points: 리샘플링할 점의 개수
    
#     Returns:
#     - x_new, y_new: 리샘플링된 에어포일 좌표
#     """
    
#     if length(x) == n_points
#         return x, y
#     end
    
#     # 에어포일 둘레를 따라 매개변수화
#     # 각 점 사이의 거리 계산
#     distances = [0.0]
#     for i in 2:length(x)
#         dist = sqrt((x[i] - x[i-1])^2 + (y[i] - y[i-1])^2)
#         push!(distances, distances[end] + dist)
#     end
    
#     # 전체 둘레로 정규화
#     total_length = distances[end]
#     t_original = distances ./ total_length
    
#     # 새로운 매개변수 값들 (균등 분포)
#     t_new = range(0, 1, length=n_points)
    
#     # 선형 보간을 사용해서 새로운 좌표 계산
#     x_new = zeros(n_points)
#     y_new = zeros(n_points)
    
#     for (i, t) in enumerate(t_new)
#         if t <= t_original[1]
#             x_new[i] = x[1]
#             y_new[i] = y[1]
#         elseif t >= t_original[end]
#             x_new[i] = x[end]
#             y_new[i] = y[end]
#         else
#             # 선형 보간
#             for j in 1:length(t_original)-1
#                 if t_original[j] <= t <= t_original[j+1]
#                     # 보간 가중치
#                     w = (t - t_original[j]) / (t_original[j+1] - t_original[j])
#                     x_new[i] = x[j] * (1 - w) + x[j+1] * w
#                     y_new[i] = y[j] * (1 - w) + y[j+1] * w
#                     break
#                 end
#             end
#         end
#     end
    
#     return x_new, y_new
# end

# function linear_interp(x_data, y_data, x_query)
#     """
#     1D 선형 보간 함수
#     """
#     if x_query <= x_data[1]
#         return y_data[1]
#     elseif x_query >= x_data[end]
#         return y_data[end]
#     else
#         for i in 1:length(x_data)-1
#             if x_data[i] <= x_query <= x_data[i+1]
#                 t = (x_query - x_data[i]) / (x_data[i+1] - x_data[i])
#                 return y_data[i] * (1-t) + y_data[i+1] * t
#             end
#         end
#     end
#     return y_data[end]  # 안전장치
# end

# function safe_dataframe_to_matrix(df, start_col=2)
#     """
#     DataFrame을 Matrix로 안전하게 변환 (missing values 및 문자열 처리)
#     """
#     try
#         # 데이터 부분만 추출
#         data_df = df[:, start_col:end]
        
#         # 각 컬럼을 개별적으로 처리
#         processed_cols = []
#         for col_name in names(data_df)
#             col_data = data_df[!, col_name]
            
#             # 문자열이나 missing values를 숫자로 변환
#             cleaned_col = []
#             for val in col_data
#                 if ismissing(val) || val == "" || (isa(val, String) && strip(val) == "")
#                     push!(cleaned_col, NaN)
#                 elseif isa(val, String)
#                     # 문자열을 숫자로 변환 시도
#                     cleaned_str = strip(val)
#                     if cleaned_str == ""
#                         push!(cleaned_col, NaN)
#                     else
#                         try
#                             push!(cleaned_col, parse(Float64, cleaned_str))
#                         catch
#                             println("Warning: Cannot convert '$val' to Float64, using NaN")
#                             push!(cleaned_col, NaN)
#                         end
#                     end
#                 else
#                     # 이미 숫자인 경우
#                     push!(cleaned_col, Float64(val))
#                 end
#             end
#             push!(processed_cols, cleaned_col)
#         end
        
#         # 매트릭스로 변환
#         n_rows = length(processed_cols[1])
#         n_cols = length(processed_cols)
#         result_matrix = zeros(Float64, n_rows, n_cols)
        
#         for (j, col) in enumerate(processed_cols)
#             for (i, val) in enumerate(col)
#                 result_matrix[i, j] = val
#             end
#         end
        
#         return result_matrix
        
#     catch e
#         println("Error converting DataFrame to Matrix: $e")
#         return nothing
#     end
# end

# function load_polar_data_once()
#     if POLAR_DATA_LOADED[]
#         return POLAR_DATA_CACHE
#     end
    
#     datapath = joinpath(uns.def_data_path, "interpolate")
#     successful_loads = 0
    
#     # NACA0012 에어포일 폴라 데이터 로드
#     try
#         println("Attempting to load NACA0012 data...")
#         df_CL_NACA0012 = uns.CSV.read(joinpath(datapath, "NACA0012_CLmap.csv"), uns.DataFrames.DataFrame)
#         df_CD_NACA0012 = uns.CSV.read(joinpath(datapath, "NACA0012_CDmap.csv"), uns.DataFrames.DataFrame)
#         df_Cm_NACA0012 = uns.CSV.read(joinpath(datapath, "NACA0012_CMmap.csv"), uns.DataFrames.DataFrame)
        
#         println("NACA0012 files loaded successfully")
        
#         # 실제 데이터 크기 확인
#         n_re_cols = size(df_CL_NACA0012, 2) - 1  # 첫 번째 열은 AOA
#         println("NACA0012 data structure: $(size(df_CL_NACA0012)) with $n_re_cols Re columns")
        
#         # 데이터 타입 확인
#         println("Column types:")
#         for (i, col_name) in enumerate(names(df_CL_NACA0012))
#             col_type = eltype(df_CL_NACA0012[!, col_name])
#             println("  Column $i ($col_name): $col_type")
            
#             # 문제가 될 수 있는 값들 확인
#             if i > 1  # AOA 컬럼 제외
#                 problematic_vals = []
#                 for (row_idx, val) in enumerate(df_CL_NACA0012[!, col_name])
#                     if ismissing(val) || (isa(val, String) && (strip(val) == "" || val == " "))
#                         push!(problematic_vals, (row_idx, val))
#                     end
#                 end
#                 if !isempty(problematic_vals)
#                     println("    Problematic values: $problematic_vals")
#                 end
#             end
#         end
        
#         # 안전한 매트릭스 변환
#         println("Converting to matrices...")
#         data_CL = safe_dataframe_to_matrix(df_CL_NACA0012)
#         data_CD = safe_dataframe_to_matrix(df_CD_NACA0012)
#         data_Cm = safe_dataframe_to_matrix(df_Cm_NACA0012)
        
#         if data_CL !== nothing && data_CD !== nothing && data_Cm !== nothing
#             # 실제 NACA0012 데이터의 Reynolds 수
#             Re_vec_NACA0012 = [1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000]
            
#             # 데이터 컬럼 수와 Re 벡터 크기 일치 확인
#             if length(Re_vec_NACA0012) != n_re_cols
#                 println("Warning: Re vector length ($(length(Re_vec_NACA0012))) != data columns ($n_re_cols)")
#                 # 데이터에 맞게 Re 벡터 조정
#                 if n_re_cols > 0
#                     Re_vec_NACA0012 = [10^(4 + i * 2/(n_re_cols-1)) for i in 0:(n_re_cols-1)]
#                     Re_vec_NACA0012 = round.(Re_vec_NACA0012)
#                 end
#             end
            
#             POLAR_DATA_CACHE["NACA0012"] = Dict(
#                 "aoa_vec" => Float64.(df_CL_NACA0012[:, 1]),
#                 "data_CL" => data_CL,
#                 "data_CD" => data_CD,
#                 "data_Cm" => data_Cm,
#                 "Re_vec" => Re_vec_NACA0012
#             )
#             println("✓ Loaded NACA0012 polar data")
#             successful_loads += 1
#         else
#             println("✗ Failed to convert NACA0012 data matrices")
#         end
#     catch e
#         println("✗ Could not load NACA0012 data: $e")
#     end
    
#     # NACA0015 에어포일 폴라 데이터 로드
#     try
#         df_CL_2416 = uns.CSV.read(joinpath(datapath, "NACA0015_CLmap.csv"), uns.DataFrames.DataFrame)
#         df_CD_2416 = uns.CSV.read(joinpath(datapath, "NACA0015_CDmap.csv"), uns.DataFrames.DataFrame)
#         df_Cm_2416 = uns.CSV.read(joinpath(datapath, "NACA0015_CMmap.csv"), uns.DataFrames.DataFrame)
        
#         # 실제 데이터 크기 확인
#         n_re_cols = size(df_CL_2416, 2) - 1
#         println("NACA0015 data structure: $(size(df_CL_2416)) with $n_re_cols Re columns")
        
#         # 안전한 매트릭스 변환
#         data_CL = safe_dataframe_to_matrix(df_CL_2416)
#         data_CD = safe_dataframe_to_matrix(df_CD_2416)
#         data_Cm = safe_dataframe_to_matrix(df_Cm_2416)
        
#         if data_CL !== nothing && data_CD !== nothing && data_Cm !== nothing
#             # NACA0015도 NACA0012과 동일한 확장된 레이놀즈 수 범위 사용
#             if n_re_cols == 8
#                 # 8개 컬럼인 경우: 확장된 범위 사용
#                 Re_vec_2416 = [1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000]
#             elseif n_re_cols == 6
#                 # 6개 컬럼인 경우: 기존 범위 사용
#                 Re_vec_2416 = [10000, 50000, 100000, 200000, 500000, 1000000]
#             else
#                 # 기타 경우: 로그 스케일로 자동 생성
#                 println("Warning: Unexpected number of Re columns ($n_re_cols), generating automatic range")
#                 Re_vec_2416 = [10^(3 + i * 3/(n_re_cols-1)) for i in 0:(n_re_cols-1)]
#                 Re_vec_2416 = round.(Re_vec_2416)
#             end
            
#             # 데이터 컬럼 수와 Re 벡터 크기 일치 확인
#             if length(Re_vec_2416) != n_re_cols
#                 println("Error: Re vector length ($(length(Re_vec_2416))) != data columns ($n_re_cols)")
#                 println("Please check your NACA0015 data files have the correct number of columns")
#             else
#                 POLAR_DATA_CACHE["NACA0015"] = Dict(
#                     "aoa_vec" => Float64.(df_CL_2416[:, 1]),
#                     "data_CL" => data_CL,
#                     "data_CD" => data_CD,
#                     "data_Cm" => data_Cm,
#                     "Re_vec" => Re_vec_2416
#                 )
#                 println("✓ Loaded NACA0015 polar data with $(n_re_cols) Re points: $(Re_vec_2416[1]) - $(Re_vec_2416[end])")
#                 successful_loads += 1
#             end
#         else
#             println("✗ Failed to convert NACA0015 data matrices")
#         end
#     catch e
#         println("✗ Could not load NACA0015 data: $e")
#     end
    
#     # 로딩 실패 시 에러 발생 (기본 데이터 사용하지 않음)
#     if successful_loads == 0
#         error("Critical Error: Could not load any airfoil polar data. Please check data files and formats.")
#     end
    
#     println("Successfully loaded $successful_loads airfoil(s) polar data")
#     POLAR_DATA_LOADED[] = true
#     return POLAR_DATA_CACHE
# end

# function polaRe(sim, pfield, T, DT; vprintln=uns._dummy)
#     rotor = sim.vehicle.rotor_systems[1][1]

#     # 폴라 데이터 로드 (한 번만 실행됨)
#     polar_data_map = load_polar_data_once()
    
#     # 사용 가능한 에어포일 확인 (첫 번째 타임스텝에서만 출력)
#     available_airfoils = collect(keys(polar_data_map))
#     if T == 0.0
#         println("Available airfoils: $(available_airfoils)")
#     end

#     # spline chord
#     spline_k = 5
#     spline_s = 0.001
#     splines_s = nothing
#     spline_bc = "extrapolate"
#     _spl_chord = uns.Dierckx.Spline1D(rotor.r, rotor.chord;
#         k=spline_k, s=splines_s != nothing ? splines_s[1] : spline_s, bc=spline_bc)
#     spl_chord(x) = uns.Dierckx.evaluate(_spl_chord, x)

#     alpha_vec = collect(-20:1:20)

#     # 개선된 에어포일 선택 함수
#     function select_airfoil(roR::Float64)
#         if haskey(polar_data_map, "NACA0015") && roR <= 0.15
#             return "NACA0015"  # 허브 근처
#         elseif haskey(polar_data_map, "NACA0012") && roR > 0.15
#             return "NACA0012"  # 팁 쪽
#         else
#             # 사용 가능한 첫 번째 에어포일 사용
#             return available_airfoils[1]
#         end
#     end

#     airfoils = []
    
#     # 현재 시간에서의 블레이드 방위각 계산
#     omega = uns.get_RPM(sim.maneuver, 1, T) * sim.RPMref * 2π / 60
#     current_psi = mod(omega * T, 2π)
    
#     # Freestream 벡터
#     V_inf_vec = Vinf(zeros(3), T)
    
#     processed_sections = 0
    
#     for i in 1:size(rotor.airfoils, 1)
#         pos = rotor.airfoils[i][1]
#         airfoil_geom = rotor.airfoils[i][2]
#         x_airfoil, y_airfoil = airfoil_geom.x, airfoil_geom.y
#         x_airfoil, y_airfoil = resample_airfoil(x_airfoil, y_airfoil, 101)

#         roR = (rotor.hubR + pos*(rotor.rotorR - rotor.hubR)) / rotor.rotorR
#         r_abs = roR * rotor.rotorR
#         chord = spl_chord(r_abs)

#         # 현재 방위각에서의 회전 속도 벡터 계산
#         if CW
#             V_rot_vec = omega * r_abs * [-sin(current_psi), cos(current_psi), 0.0]
#         else  # CCW
#             V_rot_vec = omega * r_abs * [sin(current_psi), -cos(current_psi), 0.0]
#         end

#         # 상대 속도 벡터 (회전속도 + freestream)
#         V_rel_vec = V_rot_vec + V_inf_vec
#         V_rel = norm(V_rel_vec)
        
#         # r/R에 따른 에어포일 선택
#         airfoil_key = select_airfoil(roR)
        
#         # 선택된 에어포일이 실제로 로딩되었는지 확인
#         if !haskey(polar_data_map, airfoil_key)
#             vprintln("Error: Selected airfoil '$airfoil_key' not found in loaded data")
#             continue  # 이 섹션은 건너뜀
#         end
        
#         polar_data = polar_data_map[airfoil_key]
#         aoa_vec = polar_data["aoa_vec"]
#         data_CL = polar_data["data_CL"]
#         data_CD = polar_data["data_CD"]
#         data_Cm = polar_data["data_Cm"]
#         Re_vec = polar_data["Re_vec"]

#         # Reynolds 수 계산
#         nu = mu / rho
#         Re_c = V_rel * chord / nu
        
#         # Reynolds 수 범위 확인 및 경고 (더 관대한 범위)
#         Re_min = Re_vec[1] * 0.1  # 하한을 10배 더 관대하게
#         Re_max = Re_vec[end] * 10  # 상한을 10배 더 관대하게
        
#         if Re_c < Re_min
#             vprintln("Warning: Calculated Re ($Re_c) significantly below data range ($(Re_vec[1])) for r/R=$roR")
#         elseif Re_c > Re_max
#             vprintln("Warning: Calculated Re ($Re_c) significantly above data range ($(Re_vec[end])) for r/R=$roR")
#         end
        
#         # 범위 내로 클램핑 (원래 데이터 범위 사용)
#         Re_c = max(Re_vec[1], min(Re_vec[end], Re_c))

#         # 보간 수행 전 크기 확인
#         println("Checking data dimensions for airfoil: $airfoil_key")
#         println("  AOA vec length: $(length(aoa_vec)), Re vec length: $(length(Re_vec))")
#         println("  Data CL size: $(size(data_CL))")
        
#         if size(data_CL) != (length(aoa_vec), length(Re_vec))
#             println("Error: Data size mismatch for $airfoil_key")
#             println("  Expected: ($(length(aoa_vec)), $(length(Re_vec)))")
#             println("  Got: $(size(data_CL))")
#             continue  # 데이터 크기가 맞지 않으면 건너뛰기
#         end

#         # 보간을 위한 interpolation 객체 생성
#         try
#             interp_CL = Interpolations.interpolate((aoa_vec, Re_vec), data_CL, Interpolations.Gridded(Linear()))
#             interp_CD = Interpolations.interpolate((aoa_vec, Re_vec), data_CD, Interpolations.Gridded(Linear()))
#             interp_Cm = Interpolations.interpolate((aoa_vec, Re_vec), data_Cm, Interpolations.Gridded(Linear()))

#             # 현재 Reynolds 수에서의 극선 데이터 보간
#             CL_i = [interp_CL(alpha, Re_c) for alpha in alpha_vec]
#             CD_i = [interp_CD(alpha, Re_c) for alpha in alpha_vec]  
#             Cm_i = [interp_Cm(alpha, Re_c) for alpha in alpha_vec]

#             # 새로운 극선 생성
#             polar = vlm.ap.Polar(-1, alpha_vec, CL_i, CD_i, Cm_i; x=x_airfoil, y=y_airfoil)
#             push!(airfoils, (pos, polar))
#             processed_sections += 1
            
#         catch e
#             vprintln("Error: Interpolation failed for $airfoil_key at section $i: $e")
#             continue
#         end
        
#         # 디버깅용 출력 (처음 몇 개 섹션만, 빈도 줄임)
#         if i <= 3 && mod(round(Int, T*10), 10) == 0  # 0.1초마다만 출력
#             vprintln("T=$(round(T, digits=2))s, Section $i: r/R=$(round(roR, digits=3)), Airfoil=$airfoil_key")
#             vprintln("  Re_range: $(Re_vec[1]) - $(Re_vec[end]), Current Re: $(round(Re_c, digits=0))")
#             vprintln("  V_rel: $(round(V_rel, digits=1)) m/s, chord: $(round(chord, digits=3)) m")
#         end
#     end

#     # 처리 결과 요약 (첫 번째 타임스텝에서만 출력)
#     if T == 0.0
#         vprintln("Processed $processed_sections sections out of $(size(rotor.airfoils, 1))")
#     end
    
#     # 처리된 섹션이 없으면 에러
#     if processed_sections == 0
#         error("Critical Error: No airfoil sections could be processed")
#     end

#     # 로터 업데이트
#     rotor.airfoils = airfoils
#     vlm._calc_airfoils(rotor, rotor.m, 1/5, false, []; rediscretize=false,
#                        rfl_n_lower=15, rfl_n_upper=15,
#                        rfl_r=14.0, rfl_central=true)

#     # 시뮬레이션 객체 업데이트
#     sim.vehicle.rotor_systems[1][1] = rotor
#     sim.vehicle.system.wings[1] = rotor
#     sim.vehicle.wake_system.wings[1] = rotor

#     return false
# end



# ----------------- 1) VEHICLE DEFINITION --------------------------------------
println("Generating geometry...")
#Rotor1
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
Oaxis = uns.gt.rotation_matrix2(0, 0, 0)          # New orientation
vlm.setcoordsystem(rotor1, O, Oaxis)

#Rotor2
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
Oaxis = uns.gt.rotation_matrix2(0, 0, 0)          # New orientation
vlm.setcoordsystem(rotor2, O, Oaxis)

rotors1 = vlm.Rotor[]
rotors2 = vlm.Rotor[]

push!(rotors1, rotor1)
push!(rotors2, rotor2)

println("Generating vehicle...")

"""
# Generate rotor
rotor = uns.generate_rotor(rotor_file; pitch=pitch,
                                        n=n, CW=CW, blade_r=r,
                                        altReD=[RPM, J, mu/rho],
                                        xfoil=xfoil,
                                        read_polar=read_polar,
                                        data_path=data_path,
                                        verbose=true,
                                        plot_disc=true
                                        );

println("Generating vehicle...")
"""

# Generate vehicle
system = vlm.WingSystem()                   # System of all FLOWVLM objects
vlm.addwing(system, "rotor1", rotor1)
vlm.addwing(system, "rotor2", rotor2)

# rotors = [rotor];                           # Defining this rotor as its own system
rotor_systems = ([rotor1], [rotor2]);                 # All systems of rotors

wake_system = vlm.WingSystem()              # System that will shed a VPM wake
                                            # NOTE: Do NOT include rotor when using the quasi-steady solver
if VehicleType != uns.QVLMVehicle
    vlm.addwing(wake_system, "rotor1", rotor1)
    vlm.addwing(wake_system, "rotor2", rotor2)
end

vehicle = VehicleType(   system;
                            rotor_systems=rotor_systems,
                            wake_system=wake_system
                         );


# ------------- 2) MANEUVER DEFINITION -----------------------------------------
# Non-dimensional translational velocity of vehicle over time
Vvehicle(t) = zeros(3)

# Angle of the vehicle over time
anglevehicle(t) = zeros(3)

# RPM control input over time (RPM over `RPMref`)
RPMcontrol1(t) = 1.0
RPMcontrol2(t) = 1.0

angles = ()                                 # Angle of each tilting system (none)
RPMs = (RPMcontrol1, RPMcontrol2)                       # RPM of each rotor system

maneuver = uns.KinematicManeuver(angles, RPMs, Vvehicle, anglevehicle)


# ------------- 3) SIMULATION DEFINITION ---------------------------------------

Vref = 0.0                                  # Reference velocity to scale maneuver by
RPMref = RPM                                # Reference RPM to scale maneuver by
Vinit = Vref*Vvehicle(0)                    # Initial vehicle velocity
Winit = pi/180*(anglevehicle(1e-6) - anglevehicle(0))/(1e-6*ttot)  # Initial angular velocity

simulation = uns.Simulation(vehicle, maneuver, Vref, RPMref, ttot;
                                                    Vinit=Vinit, Winit=Winit);

# Restart simulation
restart_file = nothing

# NOTE: Uncomment the following line to restart a previous simulation.
#       Point it to a particle field file (with its full path) at a specific
#       time step, and `run_simulation` will start this simulation with the
#       particle field found in the restart simulation.

# restart_file = "/path/to/a/previous/simulation/rotorhover-example_pfield.360"


# ------------- 4) MONITORS DEFINITIONS ----------------------------------------

figs, figaxs = [], []                       # Figures generated by monitor

rotors = vcat(rotor_systems...)  # rotor1과 rotor2를 한 벡터로 합침

# Generate rotor monitor
monitor_rotor = uns.generate_monitor_rotors(rotors, J, rho, RPM, nsteps;
                                            t_scale=RPM/60,        # Scaling factor for time in plots
                                            t_lbl="Revolutions",   # Label for time axis
                                            save_path=save_path,
                                            run_name=run_name,
                                            figname="rotor monitor",
                                            )

# Generate monitor of flow enstrophy (numerical stability)
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
# Concatenate monitors
monitors = uns.concatenate(monitor_rotor, monitor_enstrophy, monitor_Cd)


# ------------- 5) RUN SIMULATION ----------------------------------------------
println("Running simulation...")

# Concatenate monitors and wake treatment procedure into one runtime function
runtime_function = uns.concatenate(monitors ,wake_treatment_supress, wake_treatment)
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
                    save_wopwopin=false,  # <--- Generates input files for PSU-WOPWOP noise analysis
                    save_code=@__FILE__
                    );




# ----------------- 6) VISUALIZATION -------------------------------------------
if paraview
    println("Calling Paraview...")

    # Files to open in Paraview
    files = joinpath(save_path, run_name*"_pfield...xmf;")
    for bi in 1:B
        global files
        files *= run_name*"_Rotor_Blade$(bi)_loft...vtk;"
        files *= run_name*"_Rotor_Blade$(bi)_vlm...vtk;"
    end

    # Call Paraview
    run(`paraview --data=$(files)`)

end


# ------------- 6) POSTPROCESSING ----------------------------------------------

# Post-process monitor plots
# include(joinpath(uns.examples_path, "postprocessing.jl"))
