# Wake Trajectory Analysis for FLOWUnsteady VPM Simulations

이 가이드는 FLOWUnsteady VPM 시뮬레이션에서 wake trajectory를 추출하고 시각화하는 방법을 설명합니다.

## 개요

VPM (Vortex Particle Method) 시뮬레이션에서 wake trajectory를 그리기 위해서는 다음과 같은 데이터가 필요합니다:

1. **Particle positions** (`particle.X`): 각 vortex particle의 3D 위치 [x, y, z]
2. **Particle circulation** (`particle.Gamma`): 각 particle의 vortex 강도
3. **Time history**: 시간에 따른 particle 위치 변화

## 제공된 파일들

### 1. `wake_trajectory_utils.jl`
Wake trajectory 수집 및 시각화를 위한 핵심 유틸리티 함수들:

- `WakeTrajectoryCollector`: Particle trajectory 데이터를 저장하는 구조체
- `collect_wake_trajectories!()`: 시뮬레이션 중 실시간으로 trajectory 수집
- `create_wake_monitor()`: 시뮬레이션에 추가할 monitoring 함수 생성
- `plot_wake_trajectories_3d()`: 3D trajectory 시각화
- `save_trajectories_csv()`: CSV 형태로 데이터 저장
- `filter_trajectories_by_region()`: 특정 영역의 trajectory만 필터링

### 2. `V_stacked_with_wake_trajectories.jl`
기존 V_stacked.jl을 수정하여 wake trajectory 수집 기능을 추가한 완전한 예제:

- 실시간 trajectory 수집
- 시뮬레이션 완료 후 자동 시각화
- 여러 종류의 plot 생성 (시간별, 강도별, tip vortex)

### 3. `plot_wake_from_saved_data.jl`
기존 시뮬레이션 결과에서 wake trajectory를 추출하는 후처리 스크립트:

- 저장된 HDF5/particle field 파일에서 데이터 로드
- 후처리를 통한 trajectory 재구성
- 다양한 시각화 옵션

## 사용 방법

### 방법 1: 실시간 Trajectory 수집 (권장)

1. `V_stacked_with_wake_trajectories.jl` 파일을 사용
2. 필요에 따라 시뮬레이션 매개변수 조정:
```julia
# Wake trajectory 설정
max_trajectory_age = 3.0 * (60.0/RPM)  # 3회전 동안 추적
min_vortex_strength = 1e-6              # 최소 vortex 강도
sample_frequency = 2                    # 2 step마다 샘플링
max_particles = 500                     # 최대 추적 particle 수
```

3. 시뮬레이션 실행:
```bash
julia V_stacked_with_wake_trajectories.jl
```

4. 결과 확인:
   - `{run_name}_wake_trajectories_time.png`: 시간별 색깔구분
   - `{run_name}_wake_trajectories_strength.png`: 강도별 색깔구분
   - `{run_name}_tip_vortex_trajectories.png`: Tip vortex만 필터링
   - `{run_name}_wake_trajectories.csv`: 원본 데이터

### 방법 2: 기존 데이터에서 후처리

1. 기존 시뮬레이션 결과가 있는 경우 `plot_wake_from_saved_data.jl` 사용
2. 스크립트에서 경로와 매개변수 수정:
```julia
data_path = "/path/to/your/simulation"  # 시뮬레이션 결과 경로
run_name = "your_simulation_name"       # 시뮬레이션 이름
start_step = 50                         # 시작 step
end_step = 200                          # 끝 step
```

3. 실행:
```bash
julia plot_wake_from_saved_data.jl
```

### 방법 3: 사용자 정의 구현

기존 시뮬레이션에 wake trajectory 수집 기능을 추가하려면:

```julia
# wake_trajectory_utils.jl 로드
include("wake_trajectory_utils.jl")

# Collector와 monitor 생성
collector, wake_monitor = create_wake_monitor(
    max_age=5.0,
    min_strength=1e-6,
    sample_frequency=2
)

# 기존 runtime function에 추가
runtime_function = uns.concatenate(existing_monitors, wake_monitor)

# 시뮬레이션 실행
uns.run_simulation(simulation, nsteps;
    extra_runtime_function=runtime_function,
    # ... 기타 옵션들
)

# 결과 시각화
fig, ax = plot_wake_trajectories_3d(collector;
    color_by=:time,
    rotor_centers=[[0.0, 0.0, 0.0]],
    rotor_radius=0.14
)
```

## 시각화 옵션

### 색깔 구분 방식
- `color_by=:time`: Particle 생성 시간별
- `color_by=:strength`: Vortex 강도별  
- `color_by=:age`: Particle 나이별

### 필터링 옵션
```julia
# 특정 영역만 추적 (근거리 wake)
wake_region_filter(x, y, z) = (
    x >= -3*R && x <= 2*R &&           # 축방향 범위
    sqrt(y^2 + z^2) <= 2*R             # 반지름 범위
)

# Tip vortex 영역만 추적
tip_vortex_region(x, y, z) = (
    abs(x) < 0.1*R &&                  # Rotor 평면 근처
    sqrt(y^2 + z^2) > 0.8*R &&         # Tip 영역
    sqrt(y^2 + z^2) < 1.2*R
)
```

## 매개변수 최적화

### 성능 최적화
- `sample_frequency`: 높일수록 성능 향상, 해상도 감소
- `max_particles`: 메모리 사용량 제한
- `max_age`: 오래된 particle 제거로 메모리 절약

### 품질 최적화
- `min_strength`: 낮출수록 더 많은 weak particle 추적
- `sample_frequency=1`: 모든 step에서 데이터 수집
- 적절한 spatial filter 사용으로 관심 영역만 추적

## 출력 데이터 형식

### CSV 파일 구조
```
particle_id,time_step,x,y,z,strength,age
1,1,0.1,0.2,0.3,0.001,0.0
1,2,0.11,0.21,0.31,0.0009,0.01
...
```

### 활용 방법
- MATLAB, Python 등에서 추가 분석 가능
- 다른 시각화 도구에서 사용 가능
- 통계 분석 및 정량적 평가

## 문제해결

### 일반적인 문제들

1. **Trajectory가 수집되지 않음**
   - `min_strength` 값이 너무 높은지 확인
   - `sample_frequency`가 너무 높은지 확인
   - Spatial filter가 너무 제한적인지 확인

2. **메모리 부족**
   - `max_particles` 값 감소
   - `sample_frequency` 증가
   - `max_age` 감소

3. **시각화가 복잡함**
   - Spatial filter 사용으로 관심 영역만 표시
   - `alpha` 값 조정으로 투명도 변경
   - 시간 범위 제한

### 성능 팁

1. **실시간 수집 vs 후처리**
   - 실시간: 더 정확한 trajectory 추적
   - 후처리: 기존 데이터 활용, 더 빠른 테스트

2. **효율적인 설정**
   - Development: `sample_frequency=5`, `max_particles=100`
   - Production: `sample_frequency=1`, `max_particles=1000`

3. **대용량 시뮬레이션**
   - 여러 번에 나누어 처리
   - 관심 시간 구간만 분석
   - 적절한 spatial filtering 사용

## 예제 결과

성공적으로 실행되면 다음과 같은 결과를 얻을 수 있습니다:

1. **3D Wake Trajectory Plots**: Particle 경로의 3차원 시각화
2. **Top View Plots**: 위에서 본 wake 패턴  
3. **Tip Vortex Trajectories**: Blade tip에서 발생하는 vortex 추적
4. **CSV Data**: 정량적 분석을 위한 원본 데이터

이 도구들을 사용하여 VPM 시뮬레이션의 wake 구조를 자세히 분석하고 로터 성능을 이해할 수 있습니다.