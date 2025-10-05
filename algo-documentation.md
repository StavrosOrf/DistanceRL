# RtgRecDistRL Algorithm Documentation

## Abstract
Return-to-go Recursive Distance Reinforcement Learning (RtgRecDistRL) is an actor–metric method that learns a deterministic policy by leveraging a representation network trained to respect recursive distance constraints induced by multi-step returns. This report formalizes the algorithmic components implemented in `dist_rl/rtg_distRL.py`, analyzes their mathematical properties, and situates the method within the broader literature on representation learning for control. We provide derivations for the recursive cosine loss, detail the policy optimization objective based on return-to-go neighbors, discuss implementation particulars such as Ornstein–Uhlenbeck exploration and target network updates, and outline potential extensions.

## 1. Introduction
Recent reinforcement learning (RL) research has sought to couple policy learning with auxiliary objectives that shape representation spaces, thereby enhancing sample efficiency and stability [1,2]. Distance-based RL algorithms posit that states and actions with similar downstream returns should be embedded close together in latent space [3]. RtgRecDistRL advances this idea by combining a return-to-go (RTG) aware replay buffer with a recursive distance loss that bootstraps from the actor’s target dynamics. The resulting embedding network supports non-parametric policy improvement driven by cosine-similarity neighborhoods rather than explicit Q-value regression.

The implementation analyzed in this document is provided in `dist_rl/rtg_distRL.py`. Supporting modules `dist_rl/models.py`, `dist_rl/loss.py`, and `dist_rl/utils.py` supply neural architectures, loss functions, and buffer mechanics, respectively. This report offers a mathematically grounded walkthrough of the algorithm’s components, culminating in an end-to-end training description.

## 2. Background and Notation
We consider a Markov decision process (MDP) $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ with state space $\mathcal{S}$, action space $\mathcal{A}$, transition density $p(s'\mid s,a)$, reward function $r(s,a)$, and discount factor $\gamma \in (0,1)$ [4]. A policy $\pi(a\mid s)$ induces trajectories $\tau = (s_0,a_0,r_0,\ldots)$ with return-to-go (RTG) at time $t$ defined as $G_t = \sum_{k=0}^{T-t-1} \gamma^{k} r_{t+k}$. RtgRecDistRL focuses on continuous control problems with deterministic actors $\mu_\theta(s)$.

The algorithm maintains two primary neural modules:
1. **Actor** $\mu_\theta: \mathbb{R}^{d_s} \rightarrow \mathbb{R}^{d_a}$ implemented as a bounded MLP with hyperbolic tangent output scaled to environment action limits (`Actor` class).
2. **Distance encoder** $f_\phi: \mathbb{R}^{d_s + d_a} \rightarrow \mathbb{R}^{d_h}$ producing latent embeddings for state–action pairs (`Distance` class).

We denote normalized embeddings as $\hat{z} = f_\phi(s,a) / \lVert f_\phi(s,a) \rVert_2$. Cosine similarities between pairs $((s_i,a_i),(s_j,a_j))$ are $S_{ij} = \hat{z}_i^\top \hat{z}_j \in [-1,1]$. The replay buffer stores transitions $(s_t, a_t, r_t, s_{t+1}, d_t)$ along with Monte Carlo RTGs and $n$-step returns (`RTGRolloutBuffer`).

## 3. Return-to-Go Replay Buffer
### 3.1 Data structures
The buffer maintains tensors for observations, actions, rewards, dones, RTGs, and $n$-step returns, each of length `buffer_size`. Entries are written in cyclic fashion with pointer `self.ptr` and entry count `self.entry_count` (`RTGRolloutBuffer.__init__`). During interaction, `add` records the current transition and appends the index to the current-episode list `self._ep_idx` (`RTGRolloutBuffer.add`). Upon episode termination (`done=True`), `_backfill_episode` computes targets:

- **Monte Carlo RTG**: For episode indices $\{i_0,\ldots,i_{T-1}\}$, the method iterates backward: $G \leftarrow r_{i_t} + \gamma G$.
- **$n$-step returns**: For each time $t$ within the episode, the truncated sum $R_t^{(n)} = \sum_{k=0}^{\min(n, T-t)-1} \gamma^k r_{t+k}$ is stored.

These targets offer two complementary signals: the full RTG captures long-horizon reward structure, whereas truncated returns reduce variance and support bootstrap blending in the loss.

### 3.2 Sampling
`get_batch` selects indices uniformly from populated entries (with replacement if necessary). The method returns tensors $(\mathbf{s}, \mathbf{s}', \mathbf{a}, \mathbf{r}, \mathbf{d}, \mathbf{g}, \mathbf{R}^{(n)})$, aligning with the data needs of distance and policy updates.

## 4. Recursive $n$-Step Cosine Loss
### 4.1 Motivation
Traditional critics regress toward scalar $Q$-values, an approach sensitive to scale and overestimation [5]. RtgRecDistRL instead learns an embedding space where cosine similarity encodes proximity in expected future returns. The loss `recursive_nstep_cosine_loss` (Eq. (1)) penalizes discrepancies between current similarities and a target that interpolates RTG-derived relations and bootstrapped future similarities.

### 4.2 Loss Definition
Given a batch of embeddings $Z = [z_1,\ldots,z_B]^\top$ and next-step embeddings $Z^+ = [z_1^+,\ldots,z_B^+]^\top$, let $S = \hat{Z} \hat{Z}^\top$ and $S^+ = \hat{Z}^+ (\hat{Z}^+)^\top$ with row-wise normalization. Let $G_i$ denote the $n$-step return for sample $i$, and $d_i \in \{0,1\}$ indicate termination. The pairwise gap matrix $\Delta_{ij}$ uses a robust scale $\beta$ given by the $95$th percentile of $|G_i - G_j|$:
\[
\Delta_{ij} = \min\Bigl(1, \frac{|G_i - G_j|}{\beta} \Bigr).
\]
A shape parameter $\gamma_{shape} = v_\gamma$ adjusts the curvature of the cosine target
\[
T_{ij} = 1 - 2 \Delta_{ij}^{\gamma_{shape}}.
\]

Bootstrap mixing blends this target with discounted next similarities:
\[
Y_{ij} = (1-\lambda) T_{ij} + \lambda (1 - d_i) \gamma S_{ij}^+.
\]
The loss applies a Huber penalty (`F.smooth_l1_loss`) on off-diagonal elements:
\[
\mathcal{L}_{dist} = \frac{1}{|\mathcal{I}|} \sum_{(i,j) \in \mathcal{I}} \text{Huber}(S_{ij} - Y_{ij}; \delta),
\]
where $\mathcal{I} = \{(i,j): i \neq j\}$ and $\delta = 0.2$ (Eq. (2)). This objective enforces that embeddings respect both empirical RTG ordering and recursively bootstrapped structure from the target actor.

### 4.3 Optimization and Target Networks
The distance network parameters $\phi$ are updated via Adam with gradient clipping at unit norm. Target parameters $\phi^{-}$ mirror $\phi$ using Polyak averaging: $\phi^{-} \leftarrow \tau \phi + (1-\tau) \phi^{-}$ with $\tau=5\times 10^{-3}$. This low-pass filter stabilizes the recursive targets, akin to DDPG and TD3 [6,7].

## 5. Policy Improvement via Return-Aware Neighbors
Unlike policy gradient or actor–critic updates, RtgRecDistRL’s actor leverages the distance embedding as a non-parametric value estimator. The procedure `train_policy` operates as follows.

### 5.1 Candidate Pool and Embeddings
1. Sample a mini-batch of current states $\{s_i\}_{i=1}^B$ from the buffer.
2. Sample a larger candidate set $\{(\tilde{s}_m, \tilde{a}_m, \tilde{G}_m)\}_{m=1}^M$ with $M = \texttt{comp_samples}$.
3. Compute current actions $a_i = \mu_\theta(s_i)$ and embeddings $z_i = f_\phi(s_i,a_i)$.
4. Compute embeddings $\tilde{z}_m = f_\phi(\tilde{s}_m, \tilde{a}_m)$ without gradient.

All embeddings are L2-normalized to map to the hypersphere.

### 5.2 Similarity-Based Targets
The cosine similarity matrix $S = Z \tilde{Z}^\top$ yields, for each $i$, the top-$K$ neighbors by similarity (`torch.topk`). The associated RTGs $(\tilde{G}_{i,1},\ldots,\tilde{G}_{i,K})$ provide targets. After subtracting the row-wise mean (baseline) $\bar{G}_i$, scores become $\delta_{i,k} = \tilde{G}_{i,k} - \bar{G}_i$. A softmax transforms these centered returns into a target distribution:
\[
P^{\text{tgt}}_{i,k} = \frac{\exp(\delta_{i,k})}{\sum_{l=1}^K \exp(\delta_{i,l})}.
\]

Analogously, similarities are stabilized by subtracting the maximum per row and exponentiated to yield predicted probabilities:
\[
P^{\text{pred}}_{i,k} = \frac{\exp(S_{i,k} - \max_l S_{i,l})}{\sum_{l=1}^K \exp(S_{i,l} - \max_l S_{i,l})}.
\]
The actor minimizes the cross-entropy
\[
\mathcal{L}_{\pi} = - \frac{1}{B} \sum_{i=1}^B \sum_{k=1}^K P^{\text{tgt}}_{i,k} \log P^{\text{pred}}_{i,k}.
\]
This objective encourages the actor’s embeddings to align with high-RTG neighbors, effectively imitating high-performing trajectories stored in the buffer. Because `returns` can be either RTGs or $n$-step returns depending on the `rtg_enabled` flag, the policy can prioritize long-horizon or short-horizon signals.

### 5.3 Optimization Details
The distance encoder is frozen during actor updates to avoid collapse. Gradients are clipped at norm $10$, and Adam with learning rate $3 \times 10^{-4}$ optimizes $\theta$. After the update, target actors are Polyak-averaged with the same $\tau$ as the distance target. This interplay parallels the joint optimization of critic and actor in deterministic policy gradients but replaces scalar critics with metric embeddings.

## 6. Exploration and Data Collection
### 6.1 Action Selection
During rollouts, the actor’s deterministic action $a = \mu_\theta(s)$ is perturbed by exploration noise. Two options exist: Ornstein–Uhlenbeck (OU) noise or Gaussian noise with linearly decaying standard deviation. The OU process obeys the stochastic differential equation $dX_t = \theta (\mu - X_t) dt + \sigma dW_t$, discretized as
\[
X_{t+1} = X_t + \theta (\mu - X_t) \Delta t + \sigma \sqrt{\Delta t} \xi_t,
\]
where $\xi_t \sim \mathcal{N}(0, I)$. This process generates temporally correlated exploration suitable for physical control [6]. Noise is clipped to action bounds before being added. The noise schedule decays from $0.3$ to $0.05$ over $120{,}000$ steps to balance exploration and exploitation.

### 6.2 Training Loop
The main training loop alternates between environment interaction and learning updates once sufficient samples are collected. The agent delays distance and policy training until `val_training_start` and `policy_training_start`, respectively, ensuring the buffer contains diverse transitions. Evaluation runs are executed every `eval_freq` steps to monitor policy returns, with best-model tracking.

## 7. Algorithm Summary
The RtgRecDistRL algorithm can be summarized as Algorithm 1.

**Algorithm 1**: RtgRecDistRL
1. Initialize actor $\mu_\theta$, distance encoder $f_\phi$, targets $\mu_{\theta^-}, f_{\phi^-}$, OU noise, and RTG buffer.
2. For each environment step:
   - Execute $a_t = \text{clip}(\mu_\theta(s_t) + \epsilon_t)$ with OU or Gaussian noise.
   - Observe $(r_t, s_{t+1}, d_t)$ and push to buffer.
   - If episode ends, backfill RTGs and $n$-step returns.
   - If $t >$ `val_training_start`, sample batch and update $f_\phi$ minimizing $\mathcal{L}_{dist}$ using `recursive_nstep_cosine_loss`.
   - If $t >$ `policy_training_start`, sample batches and update $\mu_\theta$ minimizing $\mathcal{L}_{\pi}$.
   - Polyak-average target parameters.
   - Periodically evaluate policy without noise.

## 8. Connections to Existing Work
RtgRecDistRL relates to several research threads:
- **Deterministic policy gradients**: The use of OU noise and target networks echoes DDPG [6] and its successor TD3 [7], but RtgRecDistRL replaces explicit $Q$-function regression with metric learning.
- **Representation learning for value approximation**: The recursive cosine loss resembles temporal-difference metric learning approaches [8,9], where embeddings capture transition dynamics.
- **Return-conditioned policies**: By incorporating RTG targets, the algorithm aligns with return-conditioned RL paradigms such as Decision Transformer [10], albeit within an online actor–critic framework.
- **Non-parametric policy improvement**: Selecting actions via neighbor returns connects to imitation from experience replay and policy iteration via nearest neighbors [11].

## 9. Practical Considerations
### 9.1 Hyperparameters
Key hyperparameters include:
- `K` (n-step horizon): influences smoothing in $n$-step returns and distance targets.
- `comp_samples`: size of the candidate pool, balancing computational cost and neighborhood quality.
- `v_gamma`: shapes the target cosine; higher values emphasize penalizing medium gaps.
- `tau`: Polyak coefficient controlling target lag.

### 9.2 Computational Complexity
Distance updates require $\mathcal{O}(B^2)$ operations due to pairwise similarity matrices, a common drawback of metric learning. Policy updates incur $\mathcal{O}(B M)$ cost for similarity computation with the candidate pool. Efficient implementations can leverage matrix multiplications on GPUs.

### 9.3 Stability
Freezing the distance encoder during actor updates prevents representational drift. Gradient clipping and Huber losses further limit the influence of outliers. The 95th percentile scaling for gaps adaptively normalizes RTG differences, obviating manual tuning.

### 9.4 Extensions
Potential improvements include:
- Incorporating stochastic policies with reparameterized sampling for better exploration.
- Employing prioritized sampling weighted by RTG magnitudes.
- Integrating contrastive regularizers or temperature parameters in the cosine softmax.
- Extending to discrete action spaces using the `StochasticActor` class provided elsewhere in the repository.

## 10. Conclusion
RtgRecDistRL is a hybrid between deterministic actor–critic methods and metric-based representation learning. By anchoring policy updates in a learned latent geometry shaped by return-to-go structure, the algorithm avoids explicit value regression while retaining recursive bootstrapping. This technical report has detailed the mathematical foundations, implementation structure, and practical considerations necessary to understand and extend the algorithm.

## References
[1] M. G. Bellemare, W. Dabney, and R. Munos. "A Distributional Perspective on Reinforcement Learning." *International Conference on Machine Learning*, 2017.
[2] D. Silver et al. "Deterministic Policy Gradient Algorithms." *International Conference on Machine Learning*, 2014.
[3] J. B. Tenenbaum et al. "Learning Local Distance Functions." *Advances in Neural Information Processing Systems*, 2000.
[4] R. S. Sutton and A. G. Barto. *Reinforcement Learning: An Introduction*. MIT Press, 2018.
[5] H. Van Hasselt, A. Guez, and D. Silver. "Deep Reinforcement Learning with Double Q-learning." *AAAI Conference on Artificial Intelligence*, 2016.
[6] T. P. Lillicrap et al. "Continuous Control with Deep Reinforcement Learning." *International Conference on Learning Representations*, 2016.
[7] S. Fujimoto, H. Van Hoof, and D. Meger. "Addressing Function Approximation Error in Actor-Critic Methods." *International Conference on Machine Learning*, 2018.
[8] R. Agarwal et al. "Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning." *ICML*, 2021.
[9] C. Gelada et al. "DeepMDP: Learning Continuous Latent Space Models for Representation Learning." *International Conference on Machine Learning*, 2019.
[10] L. Chen et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling." *NeurIPS*, 2021.
[11] A. Barreto et al. "Successor Features for Transfer in Reinforcement Learning." *NeurIPS*, 2017.

## 11. Relationship to Repository Components
The repository includes multiple distance-based agents, such as `RecDistanceAgent` in `recursive_distRL.py` and stochastic variants in `stoch_rtg_distRL.py`. RtgRecDistRL inherits structural patterns from these agents while introducing RTG-aware sampling and the non-parametric policy objective. Notable distinctions include:

- **Buffer Design**: `RecDistanceAgent` employs `RolloutBuffer`, storing only single-step rewards. RtgRecDistRL’s `RTGRolloutBuffer` augments this with per-episode bookkeeping to compute Monte Carlo and $n$-step signals, enabling richer supervision for both the distance loss and policy target distributions.
- **Loss Function**: While `RecDistanceAgent` trains $f_\phi$ with `recursive_reward_aware_cosine_loss`, RtgRecDistRL selects `recursive_nstep_cosine_loss`, emphasizing return gaps derived from truncated sums rather than immediate rewards. This choice mitigates sensitivity to reward scaling and extends the horizon of supervision.
- **Policy Update**: The recursive agent combines cosine similarity with a soft-windowed gap transformation controlled by hyperparameter `beta`. In contrast, the RTG variant bypasses scalarized utilities in favor of neighborhood-based cross-entropy, reducing reliance on calibrated gap parameters.

Understanding these relationships clarifies how RtgRecDistRL serves as an evolution of earlier distance-based formulations within the project.

## 12. Theoretical Perspectives
### 12.1 Fixed Points of the Distance Embedding
Assuming deterministic dynamics and a stationary policy, the recursive loss admits fixed points when embeddings satisfy
\[
S_{ij} = (1-\lambda) T_{ij} + \lambda (1 - d_i) \gamma S_{ij}^+.
\]
For non-terminal transitions, recursive substitution yields
\[
S_{ij} = (1-\lambda) \sum_{t=0}^{\infty} \lambda^{t} \gamma^{t} T_{ij}^{(t)},
\]
where $T_{ij}^{(t)}$ denotes the target derived from $n$-step returns computed $t$ steps in the future. Consequently, the embedding approximates an exponentially smoothed average of gap-induced targets along the policy trajectory. Under bounded rewards and suitable contraction properties of $T_{ij}$, the recursion exhibits stability analogous to TD learning with eligibility traces.

### 12.2 Gradient Flow
Let $L = \mathcal{L}_{dist}$. Differentiating with respect to embeddings prior to normalization reveals two components: (i) gradients from cosine similarities and (ii) gradients from the normalization constraint. Concretely, if $z_i$ is the unnormalized vector and $\hat{z}_i = z_i / \lVert z_i \rVert$, then
\[
\frac{\partial S_{ij}}{\partial z_i} = \frac{1}{\lVert z_i \rVert} \Bigl( I - \hat{z}_i \hat{z}_i^\top \Bigr) \hat{z}_j.
\]
This projection matrix encourages updates tangential to the unit sphere, preserving norm. The Huber penalty ensures gradients grow linearly for large discrepancies, preventing explosion. Such geometric gradient flow is a hallmark of contrastive and metric-learning objectives and contributes to stable convergence.

### 12.3 Policy Improvement Interpretation
The cross-entropy loss can be interpreted as minimizing the Kullback–Leibler divergence $\mathrm{KL}(P^{\text{tgt}} \Vert P^{\text{pred}})$. Under mild regularity assumptions, the optimal actor embeddings align such that the cosine similarity to high-return exemplars dominates the softmax denominator. Because the candidate pool samples from replay, the policy implicitly performs a soft policy improvement step akin to Conservative Policy Iteration [12], with the softness determined by the exponential temperature implicit in the softmax. The row-wise centering of RTGs subtracts a baseline, ensuring invariance to global reward shifts.

## 13. Implementation Guidelines
### 13.1 Initialization and Seeding
The agent seeds Python, NumPy, and PyTorch random number generators through `set_seed`, ensuring reproducibility. When deploying on GPU, `torch.cuda.manual_seed_all` propagates the seed across devices. Practitioners should be aware that deterministic GPU behavior may require additional backend flags (not set in this implementation), particularly for convolutional operations.

### 13.2 Device Management
All tensors stored in the buffer reside on the specified device (CPU or GPU). Sampling operations return detached tensors, preventing gradient accumulation across reuse. When extending the algorithm, it is crucial to maintain device consistency, especially when introducing new statistics or augmentations to the buffer.

### 13.3 Logging
Weights & Biases integration occurs conditionally via `wandb_run`. During distance training, diagnostic scalars such as `mean_delta`, `mean_targets`, and gradient norms are logged, providing insight into embedding geometry. Policy updates track neighbor similarity statistics, helping diagnose issues like embedding collapse (low variance in `S_top`). When WandB is unavailable, developers can instrument additional `print` statements without affecting core logic.

### 13.4 Evaluation Protocol
Evaluation uses a separate environment clone `eval_env` without exploration noise. Episodes terminate upon either environment-provided `done` or `truncated`, accommodating Gymnasium’s time-limit termination signals. The implementation maintains `best_reward` to save high-performing models; users can extend this by serializing network weights when a new best is found.

### 13.5 Computational Footprint
The pairwise similarity computations are matrix multiplications scaling with $B^2$ and $BM$. For example, with $B=64$ and $M=256$, the primary distance matrix $S$ requires approximately $16{,}384$ dot products per update—tractable on modern GPUs. Nevertheless, practitioners should monitor memory usage when increasing hidden dimensionality or sample counts. Mixed-precision training may be introduced to mitigate memory pressure, though care must be taken with cosine normalization to avoid numerical instability.

## 14. Experimental Protocol Suggestions
While the repository does not ship with extensive evaluation scripts, the following protocol can guide empirical studies:

1. **Environment Suite**: Benchmark on MuJoCo tasks (e.g., HalfCheetah, Hopper, Walker2d) to compare against deterministic baselines like TD3. Use identical random seeds for fair comparison.
2. **Ablations**: Analyze the effect of disabling RTG targets (`rtg_enabled=False`), varying `K`, and substituting $n$-step returns with Monte Carlo RTGs in the distance loss. Track both episode returns and distance loss metrics to correlate embedding quality with policy performance.
3. **Representation Diagnostics**: Periodically project embeddings using t-SNE to visualize clustering by return magnitude. High-return actions should form coherent clusters if the metric is informative.
4. **Hyperparameter Sweeps**: Explore values for `v_gamma` and `comp_samples`. Higher `v_gamma` should sharpen the cosine target, potentially improving discrimination at the cost of sensitivity to noise.
5. **Offline Evaluation**: Because the policy update relies on replay, RtgRecDistRL may extend to offline RL scenarios. One can freeze data collection and continue policy updates, monitoring for overfitting by evaluating on held-out trajectories.

## 15. Future Research Directions
- **Analytical Guarantees**: Formalizing convergence of the metric recursion under function approximation remains an open problem. Borrowing techniques from contraction mapping analyses in distributional RL [1] could yield insights.
- **Adaptive Candidate Selection**: Instead of uniform sampling, one might prioritize candidates with high novelty or uncertainty, akin to prioritized experience replay.
- **Hybrid Critics**: Combining the metric loss with a lightweight scalar critic could blend the benefits of both worlds, offering more traditional TD targets alongside metric structure.
- **Multi-Agent Settings**: Extending the embedding to include other agents’ actions may facilitate coordination tasks where return gaps depend on joint behavior.
- **Hierarchical Policies**: Using the metric as a subgoal selector in hierarchical RL could enable coarse-to-fine planning where high-return embeddings identify valuable intermediate states.

## 16. Summary of Key Hyperparameters and Defaults
The following table summarizes default hyperparameters relevant to practitioners:

| Symbol / Name | Code Parameter | Default | Description |
|---------------|----------------|---------|-------------|
| $K$ | `K` | 5 | Horizon for $n$-step returns in buffer and loss recursion. |
| $B$ | `batch_size` | 64 | Mini-batch size for both distance and policy updates. |
| $M$ | `comp_samples` | 256 | Candidate pool size for neighbor-based actor update. |
| $\tau$ | `tau` | 0.005 | Polyak averaging coefficient for target networks. |
| $\gamma$ | `discount` | 0.99 | Discount factor for distance recursion. |
| $\gamma_{shape}$ | `v_gamma` | 1.0 | Shape parameter for cosine target. |
| $\sigma_0$ | `expl_sigma` | 0.3 | Initial exploration noise scale. |
| $\sigma_T$ | `expl_sigma_final` | 0.05 | Final exploration noise scale after decay. |
| $\lambda$ | `lam` (implicit) | 0.5 | Bootstrap mixing weight inside the distance loss. |
| $\delta$ | `huber_delta` | 0.2 | Huber threshold for distance loss residuals. |

This summary aids reproducibility and hyperparameter tuning for future experiments.

## 17. Concluding Remarks
RtgRecDistRL exemplifies a modern trend in RL towards embedding-based critics and data-driven policy improvements. Its modular design permits experimentation with alternative metric losses, buffer schemas, and policy objectives. By disentangling representation learning from explicit value regression, the algorithm provides a fertile platform for exploring how geometric constraints can inform control. We hope this documentation equips researchers and practitioners with the insight necessary to deploy, analyze, and extend the method across diverse domains.

## Additional References
[12] J. Kakade and J. Langford. "Approximately Optimal Approximate Reinforcement Learning." *International Conference on Machine Learning*, 2002.
