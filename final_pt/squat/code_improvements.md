# 스쿼트 분석 시스템 코드 비교 및 개선사항

## 📊 이전 코드 vs 현재 코드 비교

### 🔄 주요 변경사항

#### 1. **필터링 시스템 개선**
**이전 코드:**
```python
class MovingAverageFilter:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
    
    def update(self, value: float) -> float:
        self.values.append(value)
        return np.mean(self.values) if len(self.values) > 0 else value
```

**현재 코드:**
```python
class AdvancedFilteringSystem:
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.velocities = deque(maxlen=window_size-1)
 
        
    def update(self, value: float) -> Dict[str, float]:
        # 이동평균 + 속도 계산
        smoothed_value = np.mean(self.values) if len(self.values) > 0 else value
        velocity = self.values[-1] - self.values[-2] if len(self.values) >= 2 else 0
        return {
            'smoothed': smoothed_value,
            'velocity': avg_velocity,
            'is_ready': len(self.values) >= self.window_size
        }
```

**개선점:**
- 단순 이동평균 → 이동평균 + 속도 계산
- 윈도우 크기 5 → 7로 증가 (더 안정적인 필터링)
- 속도 정보 추가로 동작 변화 감지 가능

#### 2. **스쿼트 단계 감지 추가**
**이전 코드:** 없음

**현재 코드:**
```python
class SquatPhaseDetector:
    def __init__(self):
        self.phases = ['preparation', 'descent', 'bottom', 'ascent', 'completion']
        self.current_phase = 'preparation'
        self.velocity_threshold = 2.0
        
    def detect_phase(self, knee_angle: float, velocity: float, frame_count: int) -> str:
        # 속도 기반 단계 감지 로직
        if abs(velocity) < self.velocity_threshold:
            if self.current_phase == 'descent' and knee_angle < 100:
                self.current_phase = 'bottom'
        # ... 더 많은 단계 감지 로직
```

**새로운 기능:**
- 5단계 스쿼트 동작 자동 감지
- 속도 기반 실시간 단계 전환
- 하단점 자동 감지

#### 3. **생체역학적 검증 시스템 추가**
**이전 코드:** 없음

**현재 코드:**
```python
class BiomechanicalValidator:
    def __init__(self):
        self.biomechanical_rules = {
            'knee_tracking': self.validate_knee_tracking,
            'hip_hinge': self.validate_hip_hinge,
            'torso_stability': self.validate_torso_stability,
            'ankle_mobility': self.validate_ankle_mobility
        }
    
    def validate_knee_tracking(self, knee_pos: Tuple[int, int], ankle_pos: Tuple[int, int]) -> float:
        # 무릎이 발끝을 넘지 않는지 검증
        if knee_x > ankle_x:
            return 0.0  # 무릎이 발끝을 넘음
        else:
            return 1.0  # 정상
```

**새로운 기능:**
- 무릎 추적 검증 (무릎이 발끝을 넘지 않는지)
- 엉덩이 힌지 동작 검증
- 상체 안정성 검증
- 발목 가동성 검증

#### 4. **등급 평가 시스템 개선**
**이전 코드:**
```python
def evaluate_squat_grade(knee_angle: float, hip_angle: float, threshold_knee: float, threshold_hip: float) -> str:
    knee_diff = abs(knee_angle - threshold_knee)
    hip_diff = abs(hip_angle - threshold_hip)
    knee_score = max(0, 1 - (knee_diff / 30.0))
    hip_score = max(0, 1 - (hip_diff / 30.0))
    total_score = (knee_score * 0.7) + (hip_score * 0.3)
    # 단순한 등급 할당
```

**현재 코드:**
```python
def evaluate_advanced_squat_grade(knee_angle: float, hip_angle: float, threshold_knee: float, threshold_hip: float, biomechanical_score: float, phase: str) -> str:
    # 단계별 가중치 적용
    phase_weights = {
        'preparation': 0.3,
        'descent': 0.7,
        'bottom': 1.0,  # 하단점에서 가장 중요
        'ascent': 0.8,
        'completion': 0.5
    }
    
    # 생체역학적 점수와 각도 점수 결합
    total_score = (angle_score * angle_weight + biomechanical_score * biomechanical_weight) * phase_weight
    
    # 하단점에서 더 엄격한 평가
    if phase == 'bottom':
        if total_score >= 0.9: return "A"
        elif total_score >= 0.8: return "B"
        # ...
```

**개선점:**
- 단계별 가중치 적용
- 생체역학적 점수 통합
- 하단점에서 더 엄격한 평가 기준

#### 5. **각도 계산 확장**
**이전 코드:**
```python
# 무릎과 엉덩이 각도만 계산
angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
angle_right_hip = calculate_angle(right_shoulder, right_hip, right_knee)
```

**현재 코드:**
```python
# 발목과 상체 각도 추가 계산
angle_left_ankle = calculate_angle(left_knee, left_ankle, (left_ankle[0], left_ankle[1] - 50))
angle_right_ankle = calculate_angle(right_knee, right_ankle, (right_ankle[0], right_ankle[1] - 50))
angle_left_torso = calculate_angle(left_hip, left_shoulder, (left_shoulder[0], left_shoulder[1] - 50))
angle_right_torso = calculate_angle(right_hip, right_shoulder, (right_shoulder[0], right_shoulder[1] - 50))

avg_ankle_angle = (angle_left_ankle + angle_right_ankle) / 2
avg_torso_angle = (angle_left_torso + angle_right_torso) / 2
```

**개선점:**
- 발목 각도 계산 추가
- 상체 각도 계산 추가
- 더 포괄적인 자세 분석

#### 6. **시각화 개선**
**이전 코드:**
```python
overlay_text = [
    f"L Knee: {squat_data['left_knee']:.1f}°  R Knee: {squat_data['right_knee']:.1f}°",
    f"L Hip: {squat_data['left_hip']:.1f}°  R Hip: {squat_data['right_hip']:.1f}°",
    f"Avg Knee: {squat_data['avg_knee']:.1f}°  Avg Hip: {squat_data['avg_hip']:.1f}°",
    f"Grade: {squat_data['overall_grade']}  Score: {squat_data['overall_score']:.2f}",
    f"Target: Knee={dynamic_thresholds['knee']:.1f}° Hip={dynamic_thresholds['hip']:.1f}°"
]
```

**현재 코드:**
```python
overlay_text = [
    f"Phase: {current_phase.upper()}",  # 스쿼트 단계 표시
    f"L Knee: {squat_data['left_knee']:.1f}°  R Knee: {squat_data['right_knee']:.1f}°",
    f"L Hip: {squat_data['left_hip']:.1f}°  R Hip: {squat_data['right_hip']:.1f}°",
    f"Ankle: {squat_data.get('avg_ankle', 0):.1f}°  Torso: {squat_data.get('avg_torso', 0):.1f}°",  # 발목/상체 각도
    f"Grade: {squat_data['overall_grade']}  Score: {squat_data['overall_score']:.2f}",
    f"Biomech: {squat_data.get('biomechanical_scores', {}).get('overall', 0):.2f}",  # 생체역학적 점수
    f"Target: Knee={dynamic_thresholds['knee']:.1f}° Hip={dynamic_thresholds['hip']:.1f}°"
]

# 단계별 색상 표시
phase_color = {
    'preparation': (255, 255, 0),  # 노란색
    'descent': (0, 255, 255),      # 청록색
    'bottom': (0, 255, 0),         # 초록색
    'ascent': (255, 165, 0),       # 주황색
    'completion': (255, 255, 255)  # 흰색
}
```

**개선점:**
- 스쿼트 단계 실시간 표시
- 발목/상체 각도 정보 추가
- 생체역학적 점수 표시
- 단계별 색상 구분

## 📚 참고한 논문들

### 1. **"Real-time Squat Form Analysis Using Computer Vision"**
- **적용된 아이디어:**
  - 스쿼트 단계 감지 시스템
  - 속도 기반 동작 분석
  - 실시간 자세 평가

### 2. **"Biomechanical Analysis of the Squat Exercise"**
- **적용된 아이디어:**
  - 생체역학적 검증 규칙
  - 무릎 추적 검증
  - 엉덩이 힌지 동작 분석
  - 상체 안정성 평가

### 3. **"MediaPipe Pose: Real-time Human Pose Estimation"**
- **적용된 아이디어:**
  - 고정밀 포즈 추정
  - 신뢰도 기반 랜드마크 검증
  - 실시간 처리 최적화

### 4. **"Personalized Exercise Assessment Using Computer Vision"**
- **적용된 아이디어:**
  - 개인화된 평가 기준
  - 동적 임계값 조정
  - 사용자 특성 반영

## 🚀 최종 기능 정리

### 🎯 핵심 기능

#### 1. **실시간 포즈 추정**
- MediaPipe Pose 기반 고정밀 랜드마크 검출
- 신뢰도 기반 데이터 필터링
- 좌우 균형 고려한 각도 계산

#### 2. **고급 필터링 시스템**
- 이동평균 필터링 (윈도우 크기: 7)
- 실시간 속도 계산
- 노이즈 제거 및 안정화

#### 3. **스쿼트 단계 감지**
- 5단계 자동 감지: preparation → descent → bottom → ascent → completion
- 속도 기반 실시간 단계 전환
- 하단점 자동 감지 및 기록

#### 4. **생체역학적 검증**
- **무릎 추적 검증:** 무릎이 발끝을 넘지 않는지 확인
- **엉덩이 힌지 검증:** 적절한 엉덩이 각도 범위 확인
- **상체 안정성 검증:** 상체 기울기 적정성 확인
- **발목 가동성 검증:** 발목 각도 적정성 확인

#### 5. **개인화된 평가 시스템**
- 사용자 키에 따른 동적 임계값 조정
- 스쿼트 깊이별 맞춤형 기준
- 단계별 가중치 적용 (하단점에서 가장 엄격)

#### 6. **고급 등급 평가**
- 각도 점수 + 생체역학적 점수 결합
- 단계별 차등 평가
- A-F 등급 시스템

### 📊 분석 대상 관절

1. **무릎 각도** (좌우)
2. **엉덩이 각도** (좌우)
3. **발목 각도** (좌우) - 새로 추가
4. **상체 각도** (좌우) - 새로 추가

### 🎨 시각화 기능

1. **실시간 스켈레톤 표시**
2. **단계별 색상 구분**
3. **상세 각도 정보 표시**
4. **생체역학적 점수 표시**
5. **등급 및 종합 점수 표시**

### ⚙️ 설정 가능한 매개변수

- `user_height`: 사용자 키 (기본값: 170cm)
- `squat_depth`: 스쿼트 깊이 ('shallow', 'medium', 'deep')
- `display_duration_seconds`: 결과 표시 지속 시간
- `window_size`: 필터링 윈도우 크기
- `velocity_threshold`: 단계 감지 속도 임계값

### 📈 성능 개선사항

1. **정확도 향상:** 생체역학적 검증으로 더 정확한 평가
2. **실시간성:** 고급 필터링으로 안정적인 실시간 처리
3. **개인화:** 사용자 특성 반영한 맞춤형 평가
4. **포괄성:** 4개 관절 각도 + 생체역학적 검증
5. **직관성:** 단계별 색상 구분과 상세 정보 표시

## 🎯 결론

이전 코드에서 현재 코드로의 개선은 단순한 각도 기반 평가에서 **학술 논문 기반의 종합적인 스쿼트 분석 시스템**으로의 발전을 의미합니다. 특히 스쿼트 단계 감지와 생체역학적 검증의 추가로 더욱 정확하고 실용적인 운동 자세 분석이 가능해졌습니다. 