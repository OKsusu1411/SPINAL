# G1 spinal terrain curriculum consultation

상태: **solution.md R2 설계 검토 완료 — 소스 변경은 사용자 최종 승인 대기.** 운영 모드: **TEAMWORK**.

## Conclusion

`src/spinal_policy/envs/G1/G1_spinal/G1_spinal_env_cfg.py`를 지형 정의의 단일 원천으로 사용한다. 이 파일에서 Floor/Wall/Wall+Box의 변형 목록, 벽·박스 위치와 크기, 각 표면의 마찰·반발계수, 난수 seed 및 curriculum 전환 시점을 모두 정한다.

지형 설정을 준비하는 일과 curriculum을 진행하는 일은 분리한다. 설정 파일에서 모든 타일의 값을 미리 확정하고, scene 초기화 시 해당 값으로 정적 충돌 지형 전체를 생성한다. 이후 curriculum은 env step에 따라 사용할 지형 집합만 변경한다. reset마다 형상이나 물성을 다시 생성하지 않는다.

커리큘럼은 **0=Floor → 1=Wall → 2=Wall+Box → 3=전체 지형 무작위 선택**으로 제안한다. 마지막 3은 새로운 장애물 종류가 아니라, 이전 세 종류를 계속 경험하게 하는 혼합 단계다. 따라서 Wall+Box 전용 학습 단계도 유지된다. 혼합 학습은 catastrophic forgetting을 완화하기 위한 장치이며, 실제 완화 효과는 학습 평가로 확인해야 한다.

이전 문서의 “형상만 무작위화” 가정은 폐기한다. **위치·크기뿐 아니라 friction과 restitution도 이번 요구사항에 포함한다.** 사용자에게 보여지는 이 문서는 한국어로 작성하고, 이후 코드·주석·solution.md·검토 피드백·에이전트 통신 및 응답은 영어로 작성한다.

## Objectives and requirements

1. 지형 종류는 정확히 Floor, Wall, Wall+Box 세 종류다. 모두 바닥을 포함한다.
2. 각 종류에 같은 수의 변형 타일을 미리 준비한다. 기본 변형 수는 3이고, 변형 번호가 커진다고 난이도가 증가하지 않는다.
3. `G1_spinal_env_cfg.py`만 수정하여 박스 위치·크기, 벽 위치·크기, 바닥·벽·박스의 static friction, dynamic friction, restitution과 combine mode를 정할 수 있어야 한다.
4. 설정에서 결정된 실제 타일 값과 생성된 USD collision prim의 속성이 일치해야 한다. 설정에만 속성을 추가하고 물리 시뮬레이션에는 반영하지 않는 구현은 허용하지 않는다.
5. curriculum은 AISL-RL의 env-step 기반 관리자를 재사용한다. 물리 substep이나 환경 개수를 곱한 transition 수를 별도 기준으로 도입하지 않는다.
6. 마지막 단계에서는 모든 종류·모든 변형을 계속 선택할 수 있어야 한다. 각 reset 환경은 독립적으로 타일을 선택한다.
7. 단계 전환은 진행 중인 로봇을 즉시 이동시키지 않고 각 환경의 다음 reset에 적용한다.
8. 정적 지형을 공유하는 환경 사이의 로봇 충돌은 격리하고 로봇-지형 접촉은 유지한다. observation/action/reward와 로봇 자산은 변경하지 않는다.

## Context and confirmed findings

참조 행 번호는 분석 시점 체크아웃 기준이다.

| 위치 | 확인한 현재 문제 |
| --- | --- |
| `G1_spinal_env_cfg.py:139–155` | stage가 `0..8`로 정의되어 지형 종류와 변형 번호가 섞여 있다. 마지막에도 Wall+Box만 선택한다. |
| `G1_spinal_env.py:442–453` | `row=stage % num_rows`, `col=stage // num_rows`이므로 stage 1·2도 Floor다. 모든 reset 환경이 동일 타일을 선택한다. |
| `G1_spinal_env.py:176–180` | 초기 환경도 전부 `[0,0]`으로 배치한다. |
| `AISL-RL/src/lib/terrain/wall_box_terrain.py:56–57` | 없는 `cfg.box_size`를 참조하고, 이동 벡터를 `(1,3)`으로 만든다. 각각 AttributeError와 trimesh ValueError를 분리 실행으로 재현했다. |
| 같은 파일 `:25–61` | 생성자 내부 cfg와 callback 입력 cfg가 별개이며, 실제 위치·크기 표본 추출은 없다. |
| `G1_spinal_env_cfg.py:141` | `terrain_stage`는 지역 변수이므로 curriculum을 끈 직접 play 설정에서는 속성이 보장되지 않는다. |
| `AISL-RL/src/lib/utils/terrain_utils.py:23–32` | Floor wrapper가 전달 cfg 대신 내부 cfg를 사용하여 타일 크기 변경이 반영되지 않을 수 있다. |

### 물성을 반영하기 위해 필요한 spawn 경로 변경

로컬 Isaac Lab의 `TerrainGenerator`는 모든 타일을 하나의 `terrain_mesh`로 합친다. `TerrainImporter`는 이를 하나의 collision mesh로 import하고 하나의 `physics_material`을 전달한다. 따라서 이 경로를 그대로 두고 변형마다 friction/restitution 필드만 추가하면 요구사항을 충족하지 못한다.

로컬 `isaaclab.terrains.utils.create_prim_from_mesh()`는 입력한 메시마다 collider를 만들고 해당 prim 아래의 별도 physics material을 바인딩한다. 이를 이용해 **각 타일의 floor/wall/box를 별도 collision prim으로 생성**하는 경로를 제안한다. 이미 지원되는 Isaac Lab 기능을 재사용하며 설치된 라이브러리 자체를 수정하지 않는다.

## Logic and proposed architecture

### 1. 설정 파일에서 전체 타일 목록 확정

`G1_spinal_env_cfg.py`에 작은 타일 설정 구조와 목록 생성 메서드를 둔다. 데이터 구조는 다음 정보만 가진다.

- 지형 종류 인덱스 `0/1/2`, 변형 인덱스, 타일 크기.
- 벽 유무, 벽의 최종 위치·크기.
- 박스 유무, 박스의 최종 위치·크기.
- floor/wall/box 각각의 `RigidBodyMaterialCfg`.

같은 파일에서 설정한 범위를 로컬 NumPy RNG로 표본 추출하여 **최종 값이 들어 있는 `terrain_tiles` 목록**을 만든다. 사용자는 이 파일에서 고정값 또는 범위를 정한다. 고정값은 범위 양끝을 같게 지정하는 것으로 표현할 수 있으므로 별도의 복잡한 randomization 프레임워크를 만들지 않는다.

목록 준비는 최종 seed와 설정 override가 반영된 후 scene 생성 전에 한 번 수행한다. 예를 들어 환경 `_setup_scene()` 시작 시 cfg에 정의된 준비 메서드를 호출한다. 범위·분포·기본값·표본 추출 로직은 cfg 파일에 있고, env는 호출만 한다. 모듈 import 시 전역 RNG를 실행하거나 USD prim을 생성하지 않는다. 이렇게 해야 CLI/실험 설정 변경 이후에도 오래된 목록을 사용하지 않는다.

명시적 terrain seed가 있으면 우선하고, 없으면 환경에서 해석된 seed를 사용한다. 동일 seed+설정은 동일한 전체 타일 목록을 만든다. 물성 객체와 설정 리스트는 타일별로 분리하여 한 타일의 값이 다른 타일에 덮어써지지 않게 한다.

### 2. 그리드 배치와 물리 생성

논리 배치는 기존 방향을 유지한다: **행=변형, 열=지형 종류**.

| 변형 행 | column 0 | column 1 | column 2 |
| --- | --- | --- | --- |
| 0 | Floor A | Wall A | Wall+Box A |
| 1 | Floor B | Wall B | Wall+Box B |
| 2 | Floor C | Wall C | Wall+Box C |

단계 3을 위해 네 번째 열을 만들지 않는다. 혼합 단계도 이 그리드 전체를 재사용한다. `num_variants=N`, 타일 폭·길이 `(sx, sy)`라면 원점은 다음처럼 중앙 정렬한다.

```python
# Center the terrain grid in world coordinates.
origin = ((variant + 0.5 - N / 2) * sx,
          (terrain_type + 0.5 - 3 / 2) * sy,
          0.0)
```

타일별 경로 예시는 `/World/ground/tile_<variant>_<type>/floor`, `/wall`, `/box`다. floor/wall/box 각각에 cfg에서 준비한 material을 바인딩한다. 렌더링 색상을 바꾸는 것으로 물성을 적용했다고 판단하지 않는다.

메시 생성에는 기존 AISL-RL Wall/Wall+Box helper와 Isaac Lab `flat_terrain()`을 재사용한다. helper는 전달된 **최종 위치·크기**만 사용하고 추가로 난수를 뽑지 않는다. 반환된 `[floor, wall, box]` 메시를 합치지 않고 각각 생성하여 표면별 물성을 보존한다. 바닥 callback의 `(sx/2, sy/2,0)` 로컬 origin을 제거한 뒤 위의 world origin을 적용하여 중심 이동을 이중으로 하지 않는다.

단일 메시 import와 연속 난이도 생성이 필요하지 않으므로 이 task의 기존 `TerrainGenerator`/`TerrainImporter` 호출을 이 명시적인 타일 spawn 경로로 대체한다. 별도의 범용 importer, material router, terrain framework는 만들지 않는다. 작은 task 전용 helper는 `G1_spinal_terrain.py`에 두고, 값의 결정은 cfg 파일에 유지한다.

기존 importer가 보유하던 `terrain_origins[N,3,3]`, 환경별 종류·변형 인덱스와 origin은 task의 명시적인 tensor로 관리한다. `scene.env_origins`를 reset 이벤트의 위치 기준으로 유지한다. 현재 검색 결과 기반 환경은 해당 importer 상태를 사용하지 않지만, 구현 리뷰에서 다른 소비자가 없는지 다시 확인한다. importer 객체의 공개 사용처가 발견되면 호환 방안을 `solution.md`에 명시한다.

지형 생성 loop는 scene 초기화 때만 실행한다. reset에서는 PyTorch 배치 인덱싱만 수행한다. 기존 기본 border/시각화 등 importer에 암묵적으로 의존하던 동작은 제거 전에 확인하고, 필요하면 cfg에 명시한 고정 border로 재현한다. 타일 외부에 겹치는 무한 바닥을 추가해 표면별 물성을 무효화하지 않는다.

### 3. 형상·물성의 제안 기본값

아래는 승인용 시작값이며 학습 효과가 검증된 값은 아니다. 모든 값과 범위는 `G1_spinal_env_cfg.py`에 둔다.

| 항목 | 제안 값 또는 표본 분포 |
| --- | --- |
| 타일 크기 / 종류별 변형 수 | `(10.0, 10.0)` m / 3 |
| 벽 크기 | `(0.1, 3.0, 2.0)` m |
| 벽 x / y / z | uniform `[2.0,3.0]` / `0` / 벽 높이의 절반 |
| 박스 크기 | 한 변을 uniform `[0.5,0.8]` m에서 추출한 정육면체 |
| 박스 중심 x / y | uniform `[1.0,1.4]` / `[-0.8,0.8]` m |
| 박스 중심 z | 바닥 높이 + 박스 높이의 절반 |
| static friction | uniform `[0.8,1.2]` |
| dynamic friction | uniform `[0.6,0.8]` |
| restitution | uniform `[0.0,0.3]` |
| friction / restitution combine mode | 기존 `multiply` / `average` 유지 |

floor/wall/box의 물성 범위는 각각 설정할 수 있게 하고 기본 범위는 위 값을 공유한다. 각 실제 표면은 독립적으로 표본 추출한 material을 사용한다. Floor도 마찰·반발계수가 다른 변형을 가진다.

위 형상 범위에서는 박스 x 최소 경계가 원점에서 0.6 m, 벽-박스 최소 x 간격이 0.15 m다. 이 계산만으로 실제 로봇의 모든 reset yaw에서 충돌이 없다고 단정하지 않는다. 설정의 finite 값, 양수 크기, 범위 순서, 타일 내부 배치, `0 <= dynamic_friction <= static_friction`, `0 <= restitution <= 1`, 초기 로봇 공간 및 장애물 비중첩을 검증한다.

설정한 마찰·반발계수는 지형 표면의 값이다. 실제 접촉 응답에는 로봇 측 재질과 combine mode도 참여하므로, 동역학 검증에서는 양쪽 재질을 확인한다. 벽과 박스는 기존처럼 정적 장애물이며 질량·동적 박스는 이번 요구에 포함하지 않는다.

### 4. Env step 기반 curriculum

기존 AISL-RL `CurriculumManager`는 `common_step_counter / cfg.total_timesteps`를 schedule에 전달한다. `common_step_counter`는 vector environment의 `step()` 호출당 한 번 증가한다. 이를 그대로 사용한다. cfg의 총 step 수는 양수여야 하고 같은 env-step 단위여야 한다.

`G1_spinal_env_cfg.py`에 실제 `terrain_stage: float = 0.0` 속성을 정의하고 다음 설정을 제안한다.

```python
# Advance terrain sampling stages using environment-step progress.
CurriculumParamCfg(
    name="terrain_stage",
    attr_path="cfg/terrain_stage",
    start_value=0.0,
    end_value=3.0,
    schedule="step",
    schedule_kwargs={"steps": [0.3, 0.6, 0.8]},
)
```

| Env-step 진행률 | curriculum stage | reset 시 선택 대상 |
| --- | --- | --- |
| `[0%,30%)` | 0 | Floor의 모든 변형 |
| `[30%,60%)` | 1 | Wall의 모든 변형 |
| `[60%,80%)` | 2 | Wall+Box의 모든 변형 |
| `[80%,100%]` 및 그 이후 | 3 | 세 종류와 모든 변형 |

30%/60%는 기존 종류 전환 시점을 보존한 값이고, 마지막 20%를 혼합 단계로 사용하는 것은 이번 설계의 제안이다. 사용자는 같은 cfg 파일에서 전환 시점을 바꿀 수 있다. 변형 개수를 바꿔도 이 시점은 달라지지 않는다.

실제 기존 `step()` 함수에 `[0.3,0.6,0.8]`을 적용하여 경계 직전·경계·100% 이후에 `0→1→2→3`이 나오는 것을 확인했다. 이 검사는 schedule 계산만 확인했으며 시뮬레이터를 실행한 것은 아니다.

### 5. Reset에서 선택하고 이동

초기 배치와 reset에 같은 선택 함수를 사용한다. 단계 0/1/2에서는 종류를 고정하고 변형만 uniform하게 선택한다. 단계 3에서는 **종류를 uniform하게 선택한 뒤 해당 종류의 변형을 uniform하게 선택**한다. 현재처럼 종류별 변형 수가 같으면 전체 타일에 대한 uniform 선택과 같다.

```python
# Sample a prepared tile for each resetting environment.
stage = min(max(round(self.cfg.terrain_stage), 0), 3)
variants = torch.randint(num_variants, (env_ids.numel(),), device=self.device)
if stage == 3:
    terrain_types = torch.randint(3, (env_ids.numel(),), device=self.device)
else:
    terrain_types = torch.full_like(variants, stage)
origins = self.terrain_origins[variants, terrain_types]
self.terrain_variants[env_ids] = variants
self.terrain_types[env_ids] = terrain_types
self.scene.env_origins[env_ids] = origins
```

stage는 전역 curriculum 상태이고 환경별 `terrain_types`는 실제 선택한 종류다. 혼합 단계에서도 terrain type은 0/1/2이며 3을 지형 열 인덱스로 사용하지 않는다. finite stage만 허용한다.

origin 갱신은 기존처럼 `super()._reset_idx(env_ids)`보다 앞에 두어 root reset 이벤트가 새 위치를 사용하게 한다. 빈 집합·부분집합·전체 reset을 지원하고 reset되지 않은 환경의 배치는 유지한다. 준비된 메시·material은 reset 때 변경하지 않는다. 동일 타일이 연속 선택되는 것은 정상이며, 각 reset에서 이전과 다른 타일을 강제하지 않는다.

## Code: proposed changes

| 파일 | 변경 내용 |
| --- | --- |
| `src/spinal_policy/envs/G1/G1_spinal/G1_spinal_env_cfg.py:139–201` | 전체 타일 설정 구조·목록 준비 메서드·최종 위치·크기·표면별 material을 정의. 기존 단일-material generator 설정을 명시적인 타일 설정으로 교체. 실제 stage 속성과 4단계 curriculum 정의. |
| `src/spinal_policy/envs/G1/G1_spinal/G1_spinal_terrain.py` (신규, 작은 task 전용 helper) | cfg의 완성된 타일 목록을 받아 기존 메시 helper와 `create_prim_from_mesh()`로 충돌 prim을 생성하고 원점 그리드를 반환. 분포·수치·단계 정책은 두지 않음. |
| `src/spinal_policy/envs/G1/G1_spinal/G1_spinal_env.py:169–180,442–460` | cfg 목록 준비 후 scene에 한 번 생성. 필요한 terrain tensor를 초기화하고 단계별 타일 선택 및 reset origin을 갱신. 전역 지형 충돌과 환경 간 충돌 격리를 설정. |
| `AISL-RL/src/lib/terrain/wall_box_terrain.py:12–23,35–61` | cfg에서 전달한 확정 `box_size`를 지원하고 `(3,)` translation 및 지면 정렬을 수정. 생성자 내부 설정을 우회하지 않고 전달 cfg를 사용. task의 물성 범위나 난수 생성은 넣지 않음. |

AISL-RL은 별도 submodule이다. 기존 callback 이름·호출 규약은 유지하고 기존 사용처의 설정 호환성을 검토한다. task에서 더 이상 사용하지 않는 import와 설정만 제거하며, 다른 terrain utility·Isaac Lab 설치본·정책·보상·entry point는 리팩터링하지 않는다. 사용하지 않는 `WallBoxTerrain.generate()`의 별도 중복-self 문제는 필요한 경우가 아니면 보고만 한다.

## Verification and acceptance criteria

### 완료한 조사

- 전역/프로젝트 지침과 실제 source 확인. `src/` 및 `AISL-RL/` 아래 추가 AGENTS.md는 검색되지 않았다.
- AISL-RL: `cba01c0d262b28ea3127ecde4dd6975f42fe20e0`, 분석 시 source 작업 트리는 깨끗했다.
- 로컬 환경: Python 3.11.14, Isaac Sim 5.1.0.0, isaaclab 패키지 0.53.1, PyTorch 2.7.0+cu128, trimesh 4.5.1. Isaac Lab checkout은 `training-checkpoints-develop-16-gb4c32102479`다.
- 실행 환경은 `/home/oksusu/Apps/miniconda3/envs/isaaclab/bin/python`이며 실제 import 경로를 확인했다. simulation 의존 모듈 전체 import를 검증한 것은 아니다.
- 기존 박스 생성 오류와 잘못된 stage 매핑을 분리 재현했다. 이번 수정에서는 로컬 mesh별 material 바인딩 구현과 기존 curriculum의 4단계 계산을 추가 확인했다.

### 이번 요청의 검증 범위

사용자는 이번 요청에서 엄격한 디버깅과 실행 검증을 다음 요청으로 미루도록 명시했다. 따라서 최종 solution.md 승인 후 구현할 때는 **Conda 환경 이름 `isaaclab`에서 변경 Python 파일의 문법 검사만 수행**한다. 예: `conda run -n isaaclab python`으로 `ast.parse()` 또는 `compile(..., 'exec')`를 실행한다. simulation 모듈을 import하거나 Isaac Sim을 시작하지 않는다.

아래의 통합·물성·접촉·통계·실행 검증 목록은 다음 요청을 위한 기준으로 보존하며, 이번 완료 조건으로 요구하지 않는다. 문법 검사 통과를 실제 실행 성공으로 표현하지 않는다. 설계 검토에서는 명백한 요구사항 누락과 코드 구조의 문제만 바로잡고, 대규모 테스트나 무관한 수정은 추가하지 않는다.

### 다음 요청으로 이관한 실행 검증

1. **설정 재현성:** 같은 seed+설정으로 같은 타일 위치·크기·material 값이 나와야 한다. seed 변경 시 변형이 달라야 하며, 특정 표면 설정 변경이 다른 타일의 객체를 덮어쓰지 않아야 한다. 범위·seed·최종 목록은 cfg에서 확인할 수 있어야 한다.
2. **실제 scene 생성:** AppLauncher 이후 3×3 타일 전체를 생성하고 종류별로 정확히 floor / floor+wall / floor+wall+box가 존재하는지 확인한다. 각 박스는 지면 위에 놓이고 각 타일 원점·바닥 크기·로봇 초기 공간이 일치해야 한다.
3. **물성:** 각 collision prim의 physics material binding과 static/dynamic friction, restitution, combine mode를 USD/PhysX에서 읽어 cfg와 대조한다. 두 표면에 구별되는 고정값을 설정하여 공유 material에 의해 덮어쓰이지 않는지 확인한다. 제어된 접촉 실험으로 서로 다른 마찰·반발계수가 실제 물리에 반영되는지도 확인한다.
4. **단계 전환:** env-step 진행률 0, 0.3 직전/경계, 0.6 직전/경계, 0.8 직전/경계, 1 및 1 이후를 검사한다. 진행 중인 환경은 이동하지 않고 reset된 환경부터 새 선택 집합을 사용해야 한다.
5. **혼합 단계:** 충분한 반복 reset에서 Floor/Wall/Wall+Box와 모든 변형이 선택되어야 한다. 종류별 빈도가 uniform 확률과 일치하는지 통계적으로 확인한다. 작은 배치에서 세 종류가 반드시 한 번씩 나와야 한다는 잘못된 조건은 두지 않는다.
6. **배치 reset:** 부분집합·전체·빈 집합 reset, `num_variants=1,3,5`를 확인한다. 비선택 환경은 불변이고 origin/종류/변형 인덱스가 일치해야 한다. index는 long, origin은 기존 dtype/device를 유지한다. curriculum 비활성화 시 명시 stage 0/1/2/3을 지원한다.
7. **충돌·짧은 물리 실행:** 환경 4–8개로 로봇 간 충돌 격리, floor/wall/box 접촉, 초기화와 reset 후 상태의 finite 여부를 확인한다. GPU 복제 기본 경로를 우선하고 CPU/비복제 경로의 검증 여부를 별도 보고한다.
8. **학습 목표:** 혼합 단계의 선택 분포 검증은 학습 망각 해소를 증명하지 않는다. catastrophic forgetting 완화는 향후 각 지형별 성능 유지 비교로 평가한다. 이번 변경 검증을 이유로 기본 4096환경 훈련이나 장시간 학습을 시작하지 않는다.
9. **결과 보고:** 실제 명령과 통과/실패/미검증 항목을 영어로 기록한다. 문법 검사만으로 시뮬레이션 성공을 선언하지 않는다.

### 범위 밖의 기존 통합 문제

- spinal package의 자동 task 등록이 비어 있으므로 검증 harness는 AppLauncher 이후 해당 task 모듈을 명시적으로 import한다. `G1-spinal-play`는 현재 play 전용 cfg 대신 훈련 cfg를 참조한다. 등록 체계 수정은 이번 범위가 아니다.
- 분석 시 `AISL-RL/src/lib/assets/robots/G1/usd/G1_standard.usd`는 없었다. 설정이 URDF converter를 실행하므로 실제 검증 전에 원본 URDF/mesh 의존성과 변환 결과를 확인한다.
- 지형과 무관한 기존 초기화 오류가 통합 검증을 막으면 별도 보고하고 승인 범위 밖 소스를 묵시적으로 수정하지 않는다.

## Approval gate

사용자는 이 consult.md의 방향을 승인하고 TEAMWORK 설계 작성을 시작하도록 요청했다. 추가 조건은 **검증에 Conda 환경 `isaaclab` 사용**, **이번 구현 검증은 Python 문법까지만**, **실제 실행·엄격한 디버깅은 다음 요청으로 이관**이다. 이 승인은 최종 소스 코드 변경 승인이 아니다.

AGENTS.md의 **“Approval of consult.md permits design only.”**에 따라 Claude에 `DRAFT R1`을 요청하고 최대 3회 검토한다. 검토된 solution.md를 사용자에게 제시한 뒤 **최종 코드 변경 승인을 기다린다**. solution.md와 피드백은 영어로 작성한다. TEAMWORK는 유지하고 Git stage/commit/push는 수행하지 않는다.

## 후속 협의 — G1-spinal 실행 및 스폰 복구

### 결론과 범위

현재 작업 트리의 `G1-spinal`을 Isaac Sim에서 생성하고 `reset()` 및 `step()`을 호출할 수 있도록 복구한다. 4개 환경의 headless GPU 실행에서 재현한 순환 import와 접촉 센서 초기화 오류를 수정한다. 지형 생성 설정, G1 자산과 URDF 변환 옵션, 관측·행동 차원 및 종료 임계값을 유지한다.

### 재현 결과와 수정 근거

- `G1_spinal_env_cfg.py:14`와 `G1_spinal_terrain.py:13`의 상호 import 때문에 시뮬레이터에서 환경을 생성하기 전에 `ImportError`가 난다. terrain 모듈의 `G1SpinalEnvCfg` 참조는 타입 힌트용이고 파일에는 이미 `from __future__ import annotations`가 있다.
- 해당 import만 테스트 프로세스에서 우회하고 MAPPO 설정과 같이 `total_timesteps=25600`을 넘겼을 때, G1 USD prim 4개와 지형 타일 9개에 있는 활성 충돌 메시 18개를 확인했다. `G1_spinal_env.py:27`의 접촉 센서 조회에서 `head_link`, `left_rubber_hand`, `right_rubber_hand`, `pelvis_contour_link`를 찾지 못해 생성자가 중단됐다.
- 현재 G1 표준 URDF에서 위 네 링크는 각각 `torso_link`, `left_wrist_yaw_link`, `right_wrist_yaw_link`, `pelvis`에 고정되어 있다. 현재 변환기는 고정 조인트를 병합하고, 실행 로그에서 네 부모 링크를 모두 접촉 센서가 노출하는 것을 확인했다. 부모 링크의 접촉 힘에는 해당 강체에 병합된 충돌 형상의 힘이 함께 포함된다. 머리와 몸통, 손과 손목을 개별적으로 구별하려면 별도 자산·센서 설계가 필요하다.

### 제안 코드

| 파일 | 정확한 변경 |
|---|---|
| `src/spinal_policy/envs/G1/G1_spinal/G1_spinal_terrain.py:13` | `from typing import TYPE_CHECKING`을 추가하고 `G1SpinalEnvCfg` import를 `if TYPE_CHECKING:` 안으로 옮긴다. cfg에서 terrain 클래스 및 설정 타입을 가져오는 런타임 경로는 유지한다. |
| `src/spinal_policy/envs/G1/G1_spinal/G1_spinal_env_cfg.py:82` | `forbidden_collision_bodies`에서 `head_link`와 `pelvis_contour_link`를 제거한다. 해당 부모인 `torso_link`와 `pelvis`는 이미 목록에 있다. `left_rubber_hand`와 `right_rubber_hand`를 각각 `left_wrist_yaw_link`와 `right_wrist_yaw_link`로 교체한다. 변경 이유를 짧은 영어 주석으로 남긴다. |
| `src/spinal_policy/envs/G1/G1_spinal/G1_spinal_env.py:27` | `self.forbidden_collision_link_ids, _ = self.contact_sensors.find_bodies(self.cfg.forbidden_collision_bodies)`로 센서 인덱스만 저장한다. 이 값은 팔·다리 하위 집합 계산, 텐서 할당과 접촉 힘 텐서 인덱싱에 사용된다. |

`total_timesteps`는 기존 훈련 진입점에서 가하는 설정이므로 스모크 테스트에 MAPPO 설정값을 넣는다. 테스트에서 추가 오류가 재현되면 실패 위치와 수정 범위를 검토 문서에 명시한다.

### 검증과 완료 기준

1. `isaaclab`의 새 Python 프로세스에서 `AppLauncher`를 먼저 시작하고 cfg와 terrain 모듈을 두 순서로 import한다. 두 경우 모두 원본 모듈을 정상적으로 import해야 한다.
2. 등록된 `G1-spinal`을 headless `cuda:0`, `num_envs=4`, `seed=42`, `total_timesteps=25600`으로 생성한다. G1 로봇 4대의 관절 텐서가 `(4, 29)`이고, terrain origins가 `(3, 3, 3)`이며 타일 9개에 지면·벽·박스 충돌 메시 18개가 활성화되어 있어야 한다.
3. 목록 전체가 센서에서 중복 없는 정수 인덱스로 해석되는지 확인한다. 팔 하위 집합에는 두 손목과 몸통이, 다리 하위 집합에는 골반이 포함되어야 한다. 해당 인덱스에 대한 접촉 힘 텐서의 차원을 확인한다.
4. `reset()`을 두 번 호출하고 매번 arm 17차원, leg 12차원의 zero action으로 4스텝씩 실행한다. 로봇 상태·관측·보상이 유한하고 생성·리셋·스텝·종료가 예외 없이 끝나야 한다.

이 후속 범위는 별도 `consult.md` 승인 사항이다. 승인 후 Claude에 `DRAFT R1`을 전달하고, 최종 `solution.md`의 승인을 받은 뒤 `APPLY`를 요청한다.
