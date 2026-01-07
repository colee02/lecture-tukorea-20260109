import gymnasium as gym
import numpy as np
import random

# SARSA 에이전트 클래스 정의
class SARSAAgent: 
    def __init__(self, env):         # Agent 초기화
        self.env = env 
        self.state_bins = np.array([10, 100])  # 상태를 이산화할 구간 개수

        # 실수 형식의 상태를 이산화 (binning)
        n_states = np.round((env.observation_space.high-env.observation_space.low)*self.state_bins).astype(int)+1
        self.q = np.zeros([n_states[0], n_states[1], env.action_space.n])  # Q값 초기화

    def _disc_state(self, state):    # 주어진 상태를 이산화
        return np.round((state-self.env.observation_space.low)*self.state_bins).astype(int)

    def act(self, state, e):         # 다음 행동 결정
        if random.random() >= e:                    # 지금까지 학습한 최선의 행동
            s_disc = self._disc_state(state)                  # 상태를 이산화
            return np.argmax(self.q[s_disc[0], s_disc[1]])    # Q값이 최대인 행동
        else: return self.env.action_space.sample() # 가끔 무작위로 행동

    def learn(self, state, action, reward, next_state, e):  # Q값 학습
        s_disc = self._disc_state(state)        # 현재 상태 이산화
        ns_disc = self._disc_state(next_state)  # 다음 상태 이산화 
        ns_action = self.act(next_state, e)     # 다음 상태의 다음 행동

        delta = reward + 0.9 * self.q[ns_disc[0], ns_disc[1], ns_action] - self.q[s_disc[0], s_disc[1], action]
        self.q[s_disc[0], s_disc[1], action] += 0.1 * delta   # Q값 갱신

# 학습 환경 생성
train_env = gym.make(
    "MountainCar-v0",
    render_mode="rgb_array",
    max_episode_steps=400
)

# SARSA 에이전트 생성
agent = SARSAAgent(train_env) 

print("학습 시작...")

for epoch in range(5000): # 5000회 반복 학습
    state = train_env.reset()[0] # 환경 상태 초기화
    done, trunc = False, False

    while not done and not trunc: # 한 에피소드 진행
        action = agent.act(state, e=0.1) # 10% 확률로 무작위 행동
        next_state, reward, done, trunc, _ = train_env.step(action) # 행동 수행 및 결과 관찰

        agent.learn(state, action, reward, next_state, e=0.1) # Q값 학습
        state = next_state # 상태 갱신

train_env.close()

print("학습 완료!")
print("테스트 영상 녹화 중...")

# 테스트 환경 생성
test_env = gym.make(
    "MountainCar-v0",
    render_mode="rgb_array",
    max_episode_steps=400
)

# 테스트 환경은 영상 녹화
test_env = gym.wrappers.RecordVideo(
    test_env,
    video_folder="videos",    # 저장 폴더
    episode_trigger=lambda episode_id: True  
)

# 테스트 진행 및 녹화
obs, _ = test_env.reset()
done, trunc = False, False
while not done and not trunc:
    obs, _, done, trunc, _ = test_env.step(agent.act(obs, e=0))

test_env.close()