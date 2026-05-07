# SPIM V6 与 RL 增强采样主线技术说明

## 0. 这份说明的范围

这份文档只基于当前仓库中可见的代码与产物来还原实现主线，不按记忆补论文，也不拿过期报告代替代码。

- [proven] `SPIM` 相关主入口在：
  - `src/scripts/run_spim_family_sweep.py`
  - `src/scripts/run_spim_teacher_imitation_rl_pilot.py`
  - `src/scripts/run_spim_policy_eval.py`
  - `src/scripts/run_spim_policy_eval_strict.py`
- [proven] `SPIM V6` 在当前代码里的精确定义，是 `run_spim_family_sweep.py` 中的 `hsr_soft_scenario_posterior_v6`，其核心差别是：
  - 在 `v3` 的“软场景后验 + EMA”之上，额外把 trigger 节点在第一次决策前注入为一次正见证。
- [proven] 仓库里还有另一条“采样增强”线，即 `run_clean_navigator_v1.py` 的 availability-aware case sampler；它和 `SPIM` 共享一些底层 rollout/evidence 组件，但不是同一条算法主线。
- [proven] 因为你把 `SPIM V6` 和 `RL 增强采样策略` 连在一起提，这里正文主线采用 `SPIM-native teacher -> student imitation/RL` 这一条；最后会单独补充 `clean navigator` 的 availability sampler，避免概念串线。

## 1. 先做名词消歧

### 1.1 `SPIM V6` 不是 Data V6

- [proven] 仓库同时存在：
  - 数据层的 `src/data/v6/*`
  - `SPIM` 教师后验族中的 `v6`
- [proven] 这里的 `SPIM V6` 指的是教师后验版本号，不是数据格式版本号。

### 1.2 这里的“RL 增强采样”具体指什么

- [proven] 在 `SPIM` 主线上，RL 并不是“训练一个普通单步分类器”。
- [proven] 它做的是：
  1. 用 `SPIM` 教师后验把一个 case 在每一轮压缩成“可行动状态”。
  2. 让学生策略在这个状态上做一个 3-pick、无放回、set-level 的采样决策。
  3. 用 terminal-task reward 直接训练学生，而不是只做 posterior mimic。
- [proven] 也就是说，`RL 增强采样` 的对象是“每轮要采哪 3 个点”，不是“belief 头本身”。

## 2. 当前代码里的总系统边界

### 2.1 底层任务是什么

- [proven] 每个 case 都是一张局部候选子图，真正目标是在预算内直接采到真实 source。
- [proven] 统一成功定义在多个 `SPIM` 产物里都被锁成：
  - `B30 budget` 下的 `direct source hit`
- [proven] 当前 `SPIM` 主线协议是：
  - 10 轮
  - 每轮 3 个动作
  - 每轮 45 分钟
  - 总预算 30 次采样

### 2.2 运行时状态从哪里来

- [proven] 所有 `SPIM` 变体都依赖同一组底层组件：
  - `PracticalRollout`：`src/scripts/audit/utils_practical_rollout.py`
  - `ObservationWitnessHistory`：`src/modeling/evidence/two_channel_clean.py`
  - `CleanTwoChannelEvidenceEnv`：`src/modeling/evidence/two_channel_clean.py`
  - `make_rollout_state()`：`src/scripts/run_reasoner_same_case_stronger_source_overfit.py`
- [proven] `load_runtime_context()` 从上游冻结产物里读取：
  - exact136 replayable case panel
  - 数据集资产
  - `num_episodes`
  - `action_budget`
  - `episode_duration_min`
  - `frontier_role_mode`

### 2.3 每一轮 rollout 真实发生了什么

对任意 case，运行协议是：

1. 初始化 `PracticalRollout`。
2. `revealed_mask = 0`，`current_episode = 0`，`current_time_min = 0`。
3. 每轮开始时用 `observe_current_state()` 取当前部分观测、oracle 观测、physics context。
4. 用 `build_state_bundle()` / `make_rollout_state()` 把当前轮压成统一状态。
5. 教师或学生从该状态里选出 3 个还没采过的节点。
6. `rollout.step_with_actions(...)` 真正执行采样。
7. 把新采到的正/安全见证写进 `ObservationWitnessHistory`。
8. 如果 3 个动作里已经包含真实 source，则本 case 立刻成功并终止；否则进入下一轮。

## 3. 底层物理与观测语义

### 3.1 `PracticalRollout` 的状态变量

- [proven] `PracticalRollout` 内部维护：
  - `revealed_mask`
  - `current_episode`
  - `current_time_min`
  - `history_steps`
  - 子图节点的全局 id 映射 `g_ids`
- [proven] 采样动作永远是“本子图 local node index”。
- [proven] 每轮时间推进规则是：
  - `current_episode += 1`
  - `current_time_min += episode_duration_min`

### 3.2 观测是什么

- [proven] `_build_observation_state()` 用当前 snapshot 的 `x_raw[:, t_idx, :]` 构造：
  - `observed_flag`
  - `toxic_positive_flag`
  - `toxic_negative_flag`
  - `chlorine_deviation`
  - `freshness`
- [proven] 这里的“负见证”定义非常直接：
  - 被采到且浓度 `<= 0.1` 的点就是 safe witness。
- [proven] 一次采样结果最终会被记录成 `WitnessRecord`：
  - 节点 local/global id
  - 绝对时间
  - 轮次
  - 绝对 snapshot idx
  - label = `positive` 或 `safe`
  - confidence
  - 当时的 `phys_ctx`

### 3.3 动态可达距离的真正语义

- [proven] `DynamicReachabilityRuleModule` 明确采用“反向传播图”来计算 causal distance。
- [proven] 注释写得非常清楚：
  - 目标是算 `Dist(Source, Obs)`。
  - 实现方法是在反向图上从 `Obs` 做 Dijkstra。
- [proven] 因此 `compute_distance_matrix(seed_indices = witness_nodes, ...)` 返回的矩阵第 `(candidate_source, witness)` 元，就是“候选源到该见证点”的传播时间。
- [proven] 这点在 `SPIM` 的 scenario error 和 log-likelihood 里都直接使用，不能写反。

## 4. `SPIM` 教师族共用的状态压缩方式

### 4.1 候选集合如何定义

教师后验不是对所有子图节点都分布化，而是先做 candidate filtering。

- [proven] `_build_clean_candidate_mask()` 的逻辑是：
  - 以 `phys_ctx.feasible_mask` 为基础；若缺失则退回 `valid_mask`
  - 去掉 `confirmed_non_source_mask`
  - 去掉已经采过的 `revealed_mask`
  - 如果存在 trigger 节点，再额外要求该节点对 trigger 可达
- [proven] 因为 `ConstraintState` 在当前 `SPIM` rollout 里基本只把“已采样/不重采样”接进来，所以最主要的硬过滤仍然是：
  - feasible
  - 未采样
  - trigger 可达

### 4.2 `belief` 为什么不是一次性 posterior，而是递推量

- [proven] `PaperLikeHSRState` 只维护两件事：
  - `source_prior`
  - `trigger_seeded_positive`
- [proven] 这表示每一轮不是从零开始，而是：
  1. 先根据当前 history 生成一轮新的“瞬时 posterior / pseudo-posterior”
  2. 再和上一轮 `source_prior` 做混合
  3. 得到新的 belief
- [proven] 这就是代码里的 EMA flavor。

## 5. `SPIM V3` 和 `SPIM V6` 的核心定义

### 5.1 共同骨架：soft scenario posterior

- [proven] `v3` 和 `v6` 都调用 `_soft_scenario_posterior(...)` 的同一套主公式。
- [proven] 这套公式的输入是：
  - 当前 rollout 状态
  - 历史见证集合
  - trigger 节点
  - 候选源集合
  - onset offset grid
  - `alpha`
  - `time_tol_min`
  - `beta`

### 5.2 场景网格怎么定义

- [proven] 当前实现把 onset grid 固定成：
  - `[-episode_duration_min, 0, +episode_duration_min]`
- [proven] 在主线 B30 配置下，`episode_duration_min = 45`，所以 onset grid 实际上是：
  - `[-45, 0, +45]` 分钟

### 5.3 单个 `(候选源, onset)` 场景如何打分

对每个候选源 `s` 与 onset offset `o`，代码会遍历所有历史见证 `j`：

1. 取 `arrival(s -> j)`，用的是上面说的反向图 Dijkstra 结果。
2. 计算 `slack = (t_obs_j - o) - arrival(s -> j)`。
3. 用 `sigmoid(slack / time_tol)` 把它变成“该场景下此 witness 应该为 positive 的软概率”。
4. 如果当前 witness 真的是 positive，则目标值是 1；如果是 safe，则目标值是 0。
5. 对该 witness 的场景误差记为 `|expected_positive - observed_positive|`。
6. 对所有 witness 求和，得到 `scenario_error(s, o)`。

- [proven] `v3/v6` 用的不是 top-k scenario hard count，而是所有场景的软加权。
- [proven] 具体权重是：
  - `weight(s, o) = exp(-beta * shifted_error(s, o))`
  - 其中 `shifted_error = error - min(error)`

### 5.4 如何从场景权重变成 source posterior

1. 对同一个 source 的不同 onset 场景权重求和。
2. 得到该 source 的 `src_weight`。
3. 归一化，得到 `p_hat(source)`。
4. 若当前还没有历史 prior，则 prior = 候选集合上的均匀分布。
5. 否则 prior = 上一轮 `paper_state.source_prior`。
6. 最终 belief 为：
   - `belief = alpha * p_hat + (1 - alpha) * prior`
7. 再在 candidate mask 上重新归一化。

- [proven] 当前默认 `alpha = 0.55`，`beta = 2.0`，`time_tol_min = 30.0`。

### 5.5 `V6` 相比 `V3` 多了什么

- [proven] `V6` 的唯一本体改动就在 `_soft_scenario_posterior_v6()`：
  - 它先调用 `_inject_trigger_positive_witness_once(...)`
  - 然后完全复用 `v3` 的 soft-scenario posterior
- [proven] trigger 注入只发生一次。
- [proven] 注入内容是：
  - witness 节点 = `global_trigger_node`
  - label = `positive`
  - confidence = `1.0`
  - 时间 = 当前轮开始时的 `time_min`
  - `phys_ctx = 当前 state["phys_ctx"]`

### 5.6 这一步的真实含义

自然语言上，`V6` 的假设是：

- 告警触发器本身在 `tau = 0` 时已经构成一次“确定为正”的初始见证。
- 后续 adaptive sampling 不需要再去“发现 trigger 是否为正”，而是从这个已知正见证出发去反推 source。

这会直接改变：

1. 候选源对 trigger 的到达时间解释。
2. 第一轮 belief 的质量。
3. 后续 EMA prior 的起点。

## 6. 教师如何把 posterior 变成动作

### 6.1 教师动作不是采样，而是贪心 top-k

- [proven] 教师动作函数是 `_pick_topk_unsampled(...)`。
- [proven] 它做的事非常简单：
  - 按 belief 从高到低排序
  - 过滤掉 mask 外节点
  - 过滤掉已经 `revealed` 的节点
  - 取前 `k=3`

所以教师 policy 不是 stochastic planner，而是：

- `posterior_greedy`
- 每轮 3-pick
- 无放回

### 6.2 `teacher_full` 和 `teacher_slate`

- [proven] 严格评估器 `run_spim_policy_eval_strict.py` 同时支持：
  - `teacher`
  - `teacher_slate`
- [proven] `teacher_slate` 不是换 belief，而是先用和学生相同的 bounded slate 限制候选，再在 slate 内按 posterior 贪心。
- [proven] 这让评估能回答两个不同问题：
  - 学生是否优于原始 full posterior-greedy teacher
  - 学生是否优于“同等 slate 约束下的 teacher”

## 7. 学生策略如何表示“3 个采样动作”

### 7.1 不是 3 个独立分类器，而是 set-level autoregressive 3-pick

- [proven] `SpimNativePolicy.act()` 是顺序无放回决策：
  - 第 1 个动作从当前可用节点里选
  - 选过的节点从可用集合中删除
  - 第 2 个动作在剩余集合里再选
  - 第 3 个动作同理
- [proven] 因此动作空间不是“从 N 个节点独立打 3 次标签”，而是“带顺序约束的 size-3 set”。

### 7.2 当前主线模型结构

截至仓库内可见的 `20260416_posterior_v6_swap_seed45_v1/strongest_rl_v6`：

- [proven] `architecture = baseline_mlp`
- [proven] `policy_arch = separate_heads`
- [proven] `hidden_dim = 128`
- [proven] `policy_mlp_depth = 2`
- [proven] `value_mlp_depth = 3`
- [proven] `value_head_width_mult = 2.0`
- [proven] `candidate_encoder = none`
- [proven] `critic_trunk_depth = 0`

也就是说当前主线不是 transformer/gat/graphsage，而是最基础的一条：

1. 给每个候选节点拼上 local feature。
2. 给整轮状态拼一个 global feature。
3. 每一轮用同一个 MLP 框架顺序地产出 3 个动作。
4. 另有一个 value 头输出整轮状态价值。

## 8. 学生状态表示：它到底看到了什么

### 8.1 Global feature

- [proven] 当前主线 global feature 一共 12 维：
  - `round_index_norm`
  - `remaining_budget_norm`
  - `candidate_count_norm`
  - `candidate_ratio`
  - `positive_count_norm`
  - `negative_count_norm`
  - `elapsed_time_norm`
  - `posterior_entropy_norm`
  - `mass_cover_0p7_ratio`
  - `top1_mass`
  - `top3_mass`
  - `top1_top2_margin`

### 8.2 Local feature

- [proven] 当前主线 local feature 基础维度一共 9 维：
  - `posterior_mass`
  - `posterior_rank_percentile`
  - `expected_positive_prob`
  - `disagreement_score`
  - `distance_to_trigger_norm`
  - `distance_to_nearest_positive_norm`
  - `distance_to_nearest_negative_norm`
  - `legal_flag`
  - `sampled_flag`

### 8.3 这些 feature 是怎么来的

`build_spim_native_state()` 的逻辑可以还原为：

1. 从教师 belief 提取 posterior 统计量：
   - entropy
   - top-k mass
   - coverage ratio
   - top1-top2 margin
2. 从历史见证中提取：
   - 正见证个数
   - 安全见证个数
   - 到最近正/负见证的动态距离
3. 从 trigger、正见证、负见证、top posterior 源集合出发，在动态图上再算一批距离矩阵。
4. 对每个节点估计：
   - 它如果是真源，当前时刻有多大概率解释已见到的正见证
5. 令
   - `disagreement_score = min(expected_positive_prob, 1 - expected_positive_prob)`
   - 这相当于“最值得 disambiguate 的地方”。

### 8.4 受控 slate 是怎么做的

学生不是直接在所有可用节点上做 3-pick，而是先构造一个 bounded slate。

- [proven] `build_controlled_slate_mask()` 先从当前可用节点里抽一个大小为 `slate_size` 的候选集。
- [proven] 当前主线的 slate 参数是：
  - `slate_size = 10`
  - `top_posterior_k = 8`
  - `high_disagreement_k = 1`
  - `novelty_k = 1`

构造顺序是：

1. 先按 posterior 取高置信节点。
2. 再按 disagreement 取一批高歧义节点。
3. 再按 novelty 分数补一批不太冗余的节点。
4. 不够就继续按 posterior 填满。

这一步的目的不是改变任务定义，而是：

- 把 RL 决策从“全图 size-3 组合爆炸”压缩到“在一个小 slate 内做 set decision”。

## 9. 奖励是怎么定义的

### 9.1 主线 reward family

- [proven] 当前 `SPIM V6` 强化采样主线使用的是 `reward_r0_terminal_step`。
- [proven] 公式在 `_compute_reward_by_family()`：
  - `base_reward = step_penalty * selected_count + hit_reward * 1[source_hit]`
- [proven] 当前主线参数是：
  - `hit_reward = 1.0`
  - `step_penalty = -1/30`

对 B30 协议下每轮 3 个动作来说：

- 一轮没命中时基础奖励是 `-0.1`
- 命中 source 的那一轮基础奖励是 `1 - 0.1 = 0.9`

### 9.2 仓库里还支持 shaping，但主线没用

- [proven] 代码还支持：
  - `reward_r1_cover_shrink`
  - `reward_r2_topk_scenario_error_improve`
  - `reward_r3_cover_plus_error`
- [proven] 这些 shaping 通过 belief cover ratio 和 top-k scenario error 的改善来加成。
- [proven] 但当前 `strongest_rl_v6` 锁定产物没有启用这些 shaping；它用的是纯任务奖励 `r0`。

## 10. 训练主线：从 teacher 到 RL

### 10.1 训练脚本总流程

`run_spim_teacher_imitation_rl_pilot.py` 的主流程是：

1. 解析 runtime。
2. 载入训练集 case。
3. 选定 teacher family。
4. 先跑 teacher exact136 评估，得到 anchor。
5. 如果 `rl_init_mode == teacher_warm_start`，先收集 teacher transitions 做 BC。
6. 然后进入 RL rollout + PPO update。
7. 每个 epoch 在 held-out exact136 panel 上评估。
8. 用 held-out 成功率选 best epoch。

### 10.2 BC 阶段到底学什么

- [proven] `_compute_bc_loss()` 是 slot-wise teacher forcing：
  - slot 1 学 teacher 第 1 个动作
  - 删除该动作
  - slot 2 学 teacher 第 2 个动作
  - 再删除
  - slot 3 学 teacher 第 3 个动作
- [proven] 这不是 set matching，也不是 Hungarian matching；它严格遵守 teacher 的顺序动作。

### 10.3 PPO 阶段到底学什么

- [proven] `ppo_update()` 对每条 transition 计算：
  - `new_log_prob`
  - `old_log_prob`
  - `ratio = exp(new - old)`
  - clipped surrogate
  - `value_loss = MSE(new_value, target_return)`
  - entropy bonus
- [proven] target return 来自 `_prepare_ppo_targets()`：
  - 先对每个 case 的奖励序列做 discounted return
  - 再减去 baseline return（如果启用了相对 baseline）
- [proven] 当前 `strongest_rl_v6` 主线使用：
  - `advantage_baseline = value_only`
  - 所以没有再减 teacher return，只是用 `return - old_value`

### 10.4 当前 `strongest_rl_v6` 这条 run 的训练状态

- [proven] `summary.json` 里标明：
  - `rl_init_mode = random_init`
  - `rl_policy_mode = free`
  - `advantage_baseline = value_only`
- [proven] `bc_train_history.csv` 为空，`bc_summary` 实际对应 `random_init_student_pre_rl`。
- [proven] 所以这条 run 不是“teacher warm-start + RL”，而是：
  - 随机初始化学生
  - 直接靠 RL 在 `SPIM V6` 状态表示上学会 3-pick 采样

### 10.5 这条 run 实际发生了几轮 PPO

- [proven] `rl_train_history.csv` 记录了 3 个 epoch：
  - epoch 1 held-out success = `0.867647`
  - epoch 2 held-out success = `0.867647`
  - epoch 3 held-out success = `0.852941`
- [proven] best epoch 被选为第 1 轮。
- [partially proven] 脚本默认 `rl_epochs=4`、`rl_lr=1e-4`、`rl_gamma=0.97`、`clip_range=0.2`、`update_epochs=2`、`minibatch_size=64`，而当前 run 的 summary 没把所有 CLI 原样持久化；从代码与 history 看，这条 run 至少没有改掉 reward family 和 policy architecture，但我不能 100% 证明所有未显式持久化的参数都保持默认。

## 11. 当前仓库里“锁定主线”与“实验延伸”如何区分

### 11.1 `SPIM` 线里的时间顺序

- [proven] `run_spim_family_sweep.py` 先定义和比较教师族。
- [proven] `20260415_causal_exp3C_train4823_*` 是早期 multiseed 锁定线，teacher 还是 `v3`。
- [proven] `20260416_posterior_v6_swap_seed45_v1/strongest_rl_v6` 是把 teacher 切到 `v6` 后、在更强 value head v2 上做的后续锁定检查。
- [proven] `20260416_critic_value_report_v1` 说明 value head v2 相比旧 3C baseline 还有小幅增益。
- [proven] `20260416_v6_bc_regime_setint_b2c_v1` 看起来是更晚的补充实验，而不是已经替代主线的锁定产物。

### 11.2 当前最应该当“SPIM V6 + RL 采样”主参考的产物

如果你的目标是复现实验主干，而不是追每一支旁路实验，那么当前最该盯的是：

- 教师定义：
  - `src/scripts/run_spim_family_sweep.py`
  - `hsr_soft_scenario_posterior_v6`
- 训练主入口：
  - `src/scripts/run_spim_teacher_imitation_rl_pilot.py`
- 当前可见的 `V6` 锁定产物：
  - `artifacts/spim_set_level_rl_mainline/20260416_posterior_v6_swap_seed45_v1/strongest_rl_v6/summary.json`

## 12. 当前可证明的效果读数

### 12.1 早期 multiseed 3C 锁定线

- [proven] `20260415_causal_exp3C_train4823_multiseed_lock_v1/aggregate_stats.json` 显示：
  - `rl_sr mean = 0.920142`
- [proven] `per_run_summary.json` 显示 seed42/45/46 都优于对应 teacher_full。
- [proven] 但这条锁定线的 teacher 还是 `v3`，不是 `v6`。

### 12.2 当前 `V6` 交换后的 seed45 锁定线

来自 `strongest_rl_v6/summary.json` 与 `stage0_anchor_compare.json`：

- [proven] exact136 B30 teacher (`v6`)：
  - `success_rate = 0.867647`
- [proven] exact136 B30 random init pre-RL：
  - `success_rate = 0.441176`
- [proven] exact136 B30 RL final：
  - `success_rate = 0.867647`
- [proven] strict val B30：
  - teacher_full = `0.902037`
  - RL = `0.903007`
  - delta = `+0.000970`

结论是：

- [proven] 在当前仓库可见的 `V6` 锁定检查里，RL 至少能追平并略高于 `V6` teacher。
- [partially proven] 但 `V6` 的 multiseed 正式锁定目前没有像 `v3 3C` 那样完整铺开，所以如果你要写“当前最稳的多 seed 证据”，仍应把 `20260415_causal_exp3C_*` 和 `20260416_posterior_v6_swap_*` 区分开。

## 13. 如果别人要按自然语言复现，最小闭环应该怎么做

### 13.1 先复现教师 `SPIM V6`

1. 使用当前仓库的数据加载与 `PracticalRollout`。
2. 每个 case 设定：
   - 10 轮
   - 每轮 3 采样
   - 每轮 45 分钟
3. 初始 `history = []`，`revealed_mask = 0`。
4. 第一轮开始前，把 `global_trigger_node` 注入 history 为一条 `positive` witness，且只注入一次。
5. 每轮对每个候选源、每个 onset offset `{-45,0,+45}` 计算 scenario error。
6. 用 `exp(-beta * shifted_error)` 作为场景权重，按 source 聚合。
7. 与上一轮 prior 用 `alpha=0.55` 混合。
8. 在 candidate 集上归一化。
9. 每轮按 posterior 贪心取 top-3 unsampled 节点。
10. 命中真实 source 即终止。

### 13.2 再复现学生 RL 采样

1. 每轮先用教师 belief 构造 12 维 global feature 和 9 维 local feature。
2. 从可用节点里构造大小 10 的 slate：
   - posterior 8
   - disagreement 1
   - novelty 1
3. 学生策略在这个 slate 内做 3 次顺序无放回选择。
4. 每轮 reward：
   - 命中则 `+1`
   - 每选一个动作 `-1/30`
5. 对每个 case 的奖励序列做 discounted return。
6. 用 PPO clipped surrogate 更新策略，用 value head 回归 return。
7. 在 held-out exact136 panel 上按 success rate 选 best epoch。

## 14. 补充：与 `clean navigator` 的 availability sampler 是什么关系

### 14.1 不是同一条算法

- [proven] `clean navigator` 的入口是 `src/scripts/diagnostics/run_clean_navigator_v1.py`，不是 `SPIM`。
- [proven] 那条线的“availability sampler”作用在“训练 case 抽样”，不是作用在“每轮 source posterior 决策”。

### 14.2 它具体做了什么

- [proven] 它先离线对每个 train case 做一个 `oracle probe`：
  - 看这个 case 有没有机会暴露正见证
  - 有没有 pair signal
  - 有没有 reward/unresolved 改善
- [proven] 然后给每个 case 一个 `sampling_weight`。
- [proven] `sample_train_cases()` 在 `sampler_mode = availability` 时用：
  - `effective_weights = focus * avail_weights + (1 - focus) * uniform_weights`
  - 按带放回抽样选择本 epoch 训练 case
- [proven] 当前 clean lane 最佳官方产物不是“最新 availability 变体”，而是：
  - `artifacts/clean_navigator_v1/stage_rolecmp_rolebias/summary.json`
- [proven] 仓库的 `current_position_on_rubric.md` 明确写了：
  - 它优于更新但退化的 `stage_sampler_availability10_rolebias_pairfrontier`

所以如果你说的“RL 增强采样”是 `clean navigator` 这条线，那么它的核心是“availability-aware 训练 case 采样 + 3-slot clean actor-critic”；但如果你说的是和 `SPIM V6` 同一条主线，那么应当理解为本文前面展开的 `SPIM-native set-level RL sampling`。

## 15. 最后给一个一句话主线定义

当前仓库里最准确的自然语言主线可以压成一句：

- `SPIM V6` 是一个“把 trigger 当作一次已知正见证注入后，再用动态传播时间一致性对 source×onset 场景做软加权并跨轮 EMA 更新”的教师后验；
- `RL 增强采样` 是“把这个教师 belief 压成一个 bounded slate 上的 set-level 3-pick 状态，再用 PPO 直接优化预算内命中真实 source 的采样策略”。

