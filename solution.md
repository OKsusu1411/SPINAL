# G1GaitEnv Reward shaping

## Requirement
일반적으로 humanoid gait RL에서 발생하는 문제 중 하나는 hopping입니다.
Velocity tracking reward만으로 학습하면 로봇이 양발로 동시에 점프하면서 이동하는 동작을 발견할 수 있습니다.
이를 방지하기 위해 논문에서는 single-foot contact를 유도합니다.
$$[ r_{\mathrm{contact}}= \begin{cases} 1,&\text{Standing command}\\ 1,&\exists t'\in[t-0.2,t]:n_c(t')=1\\ 0,&\text{Otherwise} \end{cases} \
$$
여기서 $n_c$는 지면에 접촉한 발의 개수입니다.
즉 walking 중에는 최근 0.2초 안에 한쪽 발만 접촉한 순간이 있으면 reward를 제공합니다.
흥미로운 점은 standing 상태에서 양발 접촉을 강제하지 않는다는 것입니다.
Standing command에서는 reward를 상수 1로 제공합니다.
그 이유는 외부 disturbance 발생 시 recovery stepping을 허용하기 위해서입니다.
저자들은 양발 접촉을 강하게 reward하면 넘어지지 않기 위해 발을 들어야 하는 상황에서도 foot contact를 유지하려는 부작용이 나타날 수 있다고 설명합니다.


논문의 air-time reward는 다음과 같습니다.
$$
 r_{\mathrm{air}}= \begin{cases} 1,&\text{Standing}\\ \displaystyle\sum_{f\in\{L,R\}} (T_{\mathrm{air},f}-0.4) \mathbb 1_{\mathrm{touchdown},f}, &\text{Walking} \end{cases}
$$
Weight는
$w_{\mathrm{air}}=1.0$
입니다.
여기서 $T_{\mathrm{air},f}$는 해당 발의 touchdown 직전 swing duration입니다.
이 reward는
$T_{\mathrm{air},f}-0.4$
를 사용하므로, air time이 0.4초보다 짧으면 음수이고 길면 양수입니다.
따라서 짧은 주기로 발을 반복적으로 디디는 행동을 억제하고, 적절한 swing duration을 갖도록 유도합니다.

---

## 구현 방안

### 0. 사전 확인 사항 (현재 코드 기준)
- 발 contact 정보: `self.ankle_contact_roll_link_ids` (`[Left, Right]`), contact sensor는 `track_air_time=True`, `update_period = sim_dt` 로 설정되어 있음 → Isaac Lab `ContactSensor`의 `current_contact_time`, `last_air_time`, `compute_first_contact(dt)`를 그대로 사용 가능.
- Policy step은 `step_dt = sim_dt * decimation = 0.02s`. Touchdown 판정은 반드시 `step_dt`로 해야 함 (`sim_dt`로 하면 decimation 중간에 발생한 touchdown을 놓침).
- Standing command 판별: `UniformVelocityCommand.is_standing_env`가 있으나, resampling 외 경로로 0 command가 들어오는 경우까지 포괄하도록 command norm 기반으로 판별하는 것을 권장.
  ```python
  is_standing = torch.norm(self.command_inputs_b[:, :2], dim=-1) < self.cfg.standing_cmd_threshold
  ```
  (현재 `ang_vel_z` range가 0이므로 xy만 사용. yaw command를 쓰게 되면 `command_inputs_b[:, 2]`도 포함.)

### 1. Config 추가 (`G1_gait_env_cfg.py`)
```python
    ## ==================== Reward Shaping ==================== ##
    w_single_contact:     float = 1.0
    w_feet_air_time:      float = 1.0

    # ===== Gait guidance ===== #
    single_contact_window: float = 0.2    # [s] look-back window for single-foot contact
    air_time_threshold:    float = 0.4    # [s] target swing duration
    standing_cmd_threshold: float = 0.1   # [m/s] command norm below this is treated as standing
```

### 2. 버퍼 추가 (`__init__`)
`[t-0.2, t]` 구간 안에 `n_c = 1`인 순간이 있었는지 알기 위해 히스토리 전체를 저장할 필요 없이, **"마지막으로 single contact였던 시점으로부터 경과한 시간"** 하나만 들고 있으면 됨.
```python
        # Gait guidance (Single foot contact)
        self.time_since_single_contact = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
```

### 3. 중간값 갱신 (`_compute_intermediate_values`)
`self.in_contact` 계산 직후에 추가:
```python
        # Single foot contact timer: reset to zero when exactly one foot is in contact
        num_contacts = torch.sum(self.in_contact[i], dim=-1)    # [E]
        single_contact = num_contacts == 1
        if env_ids is not None:
            # Grace period after reset (robot spawns in double support)
            self.time_since_single_contact[i] = 0.0
        else:
            self.time_since_single_contact[i] = torch.where(single_contact,
                                                            torch.zeros_like(self.time_since_single_contact[i]),
                                                            self.time_since_single_contact[i] + self.step_dt)
```
- reset 시 0으로 두어 초기 양발 지지 상태에서 첫 0.2s 동안은 reward를 주도록 함 (원치 않으면 `single_contact_window` 보다 큰 값으로 초기화).
- `_get_dones()` 에서 매 step 1회 `_compute_intermediate_values()`가 호출되므로 timer는 step 당 정확히 `step_dt`씩 증가.

### 4. Reward 계산 (`_get_rewards`)
기존 placeholder 위치(`# Single foot contact`, `# Feet airtime`)에 작성:
```python
        # Standing mask
        is_standing = torch.norm(self.command_inputs_b[:, :2], dim=-1) < self.cfg.standing_cmd_threshold

        # Single foot contact
        single_contact = self.time_since_single_contact <= self.cfg.single_contact_window
        single_contact_rewards = torch.where(is_standing,
                                             torch.ones_like(single_contact, dtype=torch.float),
                                             single_contact.float())

        # Feet airtime
        first_contact = self.contact_sensors.compute_first_contact(self.step_dt)[:, self.ankle_contact_roll_link_ids]  # [E, 2]
        last_air_time = self.contact_sensors.data.last_air_time[:, self.ankle_contact_roll_link_ids]                 # [E, 2]
        air_time = torch.sum((last_air_time - self.cfg.air_time_threshold) * first_contact.float(), dim=-1)
        air_time_rewards = torch.where(is_standing, torch.ones_like(air_time), air_time)
```
- `compute_first_contact(step_dt)`: 이번 policy step 안에서 contact가 새로 시작된 발 = $\mathbb 1_{\mathrm{touchdown},f}$.
- `last_air_time`: touchdown 시점에 직전 swing duration으로 갱신됨 = $T_{\mathrm{air},f}$.
- 따라서 reward는 touchdown이 일어난 step에만 non-zero (sparse) 값을 가짐.

### 5. Reward 합산 및 로깅
보행 패턴(발 접촉)과 관련된 항이므로 `leg_specific_rewards`에 포함:
```python
        leg_specific_rewards = self.cfg.w_track_lin_vel       * lin_vel_rewards                 + \
                               self.cfg.w_single_contact      * single_contact_rewards          + \
                               self.cfg.w_feet_air_time       * air_time_rewards                + \
                               ...
```
`self.extras["reward"]`에 추가:
```python
            "Task Reward / Leg_Single_Contact"     : single_contact_rewards,
            "Task Reward / Leg_Feet_Air_Time"      : air_time_rewards,
```
(air time 항은 음수도 가지므로 Reward/Penalty 어느 쪽에 두든 상관없으나, standing 시 +1이므로 Reward 쪽에 둠.)

### 6. 주의 사항
1. **Gait scheduler와 threshold 충돌**: 현재 `time_period = 0.35`로 swing foot가 0.35s 마다 교대됨 → 실제 swing(air) 시간은 최대 ~0.35s 미만. `air_time_threshold = 0.4`를 그대로 쓰면 $T_{\mathrm{air}} - 0.4 < 0$ 이 항상 성립하여 **사실상 매 touchdown마다 penalty만 주는 항**이 됨. 논문 값(0.4)을 유지하려면 `time_period`를 늘리거나, threshold를 `time_period` 이하(예: 0.3)로 맞추는 것을 권장. threshold를 cfg로 뺀 이유.
2. **Standing reward 상수 1**: 두 항 모두 standing 시 1을 주므로 standing env 비율(`prob_standing_envs`)이 올라가면 walking env 대비 reward scale 차이가 생김. 현재 `prob_standing_envs=0.0`이라 영향 없음.
3. **Reset 직후 touchdown**: 스폰 높이가 지면보다 약간 높으면 첫 touchdown에서 짧은 air time으로 작은 penalty가 들어갈 수 있음 (무시 가능한 수준).
4. **Contact 판정 기준**: `self.in_contact`(contact time 기반)를 사용. Force threshold 기반의 `self.is_contacts`(history max)는 3 substep history를 보므로 판정이 더 관대함 → single contact 판정이 흐려질 수 있어 `in_contact` 권장.
