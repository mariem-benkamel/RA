## Replacement Content for `test3.pdf`

This file provides ready-to-insert replacement text for the requested parts of the document.
It is written to keep the same section numbering style used in `test3.pdf`.

### Figure 3 (replacement note)
Replace **"Figure 3: Federated Learning Architecture"** with the **general architecture figure from article [27]** (instead of a bus-specific or over-specialized visual).

Suggested caption:

**Figure 3: General Federated Learning Architecture (Source: [27]).**

---

## 5.2.1 By Data Distribution
Federated learning can be categorized according to how data is partitioned across participants. In general, three settings are used: horizontal federated learning (HFL), vertical federated learning (VFL), and federated transfer learning (FTL).

- **Horizontal FL (HFL)**: participants share the same feature space, but each participant owns different samples.
- **Vertical FL (VFL)**: participants share the same entities (or overlapping entities), but each participant owns different features.
- **Federated Transfer Learning (FTL)**: participants have limited overlap in both samples and features, so transfer mechanisms are used.

In our use case (predictive maintenance for bus fleets), **HFL is the most natural baseline**, because each bus typically generates similar sensor types (temperature, vibration, battery, mileage), while local trip data differs from one vehicle to another.

**Conclusion for our case:**
- **HFL works best as the default setting** for fleet-scale predictive maintenance.
- **VFL is useful** when cross-organization feature fusion is needed (e.g., operator data + infrastructure data).
- **FTL is relevant** when fleets, cities, or operators have very different sensor stacks and low overlap.

---

## 5.2.2 By System Architecture
Federated learning can also be classified by coordination architecture. In general, three architectures are common:

1. **Centralized FL**: a central server orchestrates rounds and aggregation.
2. **Hierarchical FL**: intermediate aggregators (e.g., depots/edges) aggregate locally before forwarding to the cloud.
3. **Decentralized FL**: clients exchange updates peer-to-peer without a single global coordinator.

For predictive maintenance in bus fleets, architecture selection depends on connectivity, latency, and operational structure.

**Conclusion for our case:**
- **Centralized FL** is simplest to deploy and monitor for a first production rollout.
- **Hierarchical FL** is often the best compromise at large scale (fleet -> depot -> cloud), reducing bandwidth and improving robustness.
- **Decentralized FL** is interesting for resilience research but is usually harder to govern in regulated transport operations.

---

## 5.4 Federated Learning Lifecycle
In general, federated learning follows an iterative lifecycle:
1. initialize a global model,
2. select available clients,
3. send model parameters,
4. run local training,
5. collect updates,
6. aggregate,
7. repeat until stopping criteria are met.

Two orchestration modes are commonly used:

- **Synchronous FL**: the server waits for all selected clients before aggregation.
  - Advantage: stable and easier-to-analyze rounds.
  - Limitation: stragglers delay progress.

- **Asynchronous FL**: the server updates when client updates arrive.
  - Advantage: better tolerance to intermittent connectivity.
  - Limitation: stale updates must be handled carefully.

For bus fleets, asynchronous or semi-asynchronous strategies are often practical because vehicles may disconnect across routes or depots.

---

## 5.5 Core Algorithms
Before discussing algorithms, we clarify terminology:

- A **model** is the predictive function being learned (e.g., MLP, LSTM, Transformer).
- An **algorithm** is the distributed training procedure that tells clients and server how to update and aggregate that model.

So, the model defines **what** is learned, while the algorithm defines **how** learning is coordinated across clients.

### 5.5.1 FedAvg
**How it works (intuitive):**
Each selected client trains the current global model locally for a few steps, then the server averages client models using client data size as weight.

**Simple math:**
\[
w_{t+1} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j}\, w_t^k
\]
where \(w_t^k\) is client \(k\)'s updated model and \(n_k\) is its sample count.

### 5.5.2 FedSGD
**How it works (intuitive):**
Clients compute one local gradient step and send gradients/updated weights each round.

**Simple math:**
\[
w_{t+1} = w_t - \eta \sum_{k \in S_t} \frac{n_k}{n}\, \nabla F_k(w_t)
\]

### 5.5.3 FedProx
**How it works (intuitive):**
FedProx stabilizes local training by penalizing local models that move too far from the global model.

**Simple math:**
\[
\min_w\; F_k(w) + \frac{\mu}{2}\|w - w_t\|^2
\]

### 5.5.4 SCAFFOLD
**How it works (intuitive):**
SCAFFOLD corrects client drift under non-IID data using control-variates that reduce biased local directions.

**Simple math (compact):**
\[
w \leftarrow w - \eta\big(\nabla F_k(w) - c_k + c\big)
\]

### 5.5.5 FedNova
**How it works (intuitive):**
FedNova normalizes updates so clients with different local step counts do not bias aggregation.

**Simple math:**
\[
\Delta w_k^{\text{norm}} = \frac{1}{\tau_k}\Delta w_k
\]

### 5.5.6 DAAFL
**How it works (intuitive):**
DAAFL reweights contributions using both data quantity and staleness-aware participation weighting.

**Simple math:**
\[
\alpha_k^t \propto n_k \cdot f(s_k^t)
\]
where \(s_k^t\) is staleness (time since the client last participated effectively), and a simple choice is \(f(s)=\frac{1}{1+s}\), so updates from clients with higher staleness receive less weight.

### 5.5.7 FedSA
**How it works (intuitive):**
FedSA combines synchronous warm-up with asynchronous updates and reduces local epochs for stale clients.

**Simple math:**
\[
E_k = \max(1, E_{\max} - \lambda s_k)
\]

---

## 5.6 Choice Criteria
Algorithm and architecture selection should be based on operational constraints, not only benchmark accuracy.

Main criteria:
1. **Data heterogeneity** (IID vs non-IID severity)
2. **System heterogeneity** (device compute, memory, and power)
3. **Connectivity reliability** (stable, intermittent, or sparse)
4. **Communication budget** (uplink/downlink constraints)
5. **Privacy and governance constraints**
6. **Deployment complexity and maintainability**
7. **Convergence considerations**

### Convergence considerations
Convergence behavior depends on:
- local epoch count,
- participation rate,
- staleness (asynchronous settings),
- degree of non-IID data,
- and optimizer hyperparameters.

In fleet maintenance scenarios, convergence must be evaluated with both **global performance** and **fairness across vehicle subgroups** (e.g., old vs new buses, urban vs suburban routes), to avoid overfitting to dominant participation patterns.

---

## 6 Revised (Complete Rewrite)
### 6.1 Problem Framing for Bus Fleet Predictive Maintenance
The objective is to estimate failure risk or remaining useful life from onboard telemetry while preserving data locality at vehicle or depot level.

### 6.2 Data and Client Definition
A client can be a bus, an edge unit, or a depot server. Typical input signals include engine/thermal measurements, vibration, battery indicators, maintenance logs, and route context.

### 6.3 Baseline Pipeline
1. Local preprocessing and feature validation.
2. Local model update on recent windowed data.
3. Secure transmission of model updates.
4. Server-side aggregation and redeployment.

### 6.4 Recommended FL Configuration for This Use Case
- Start with **HFL + centralized (or hierarchical) FedAvg** as operational baseline.
- Introduce **FedProx or SCAFFOLD** when non-IID drift degrades stability.
- Move to **semi/asynchronous variants** when connectivity instability becomes dominant.

### 6.5 Evaluation Protocol
Evaluate with:
- predictive metrics (e.g., F1, AUROC, MAE depending on task),
- communication cost per round,
- convergence speed,
- robustness to client dropout,
- and subgroup fairness across fleet segments.

### 6.6 Operational and Security Considerations
Use authenticated update channels, secure aggregation when available, and privacy-preserving mechanisms (e.g., clipping/noise) based on regulatory requirements and acceptable utility loss.

### 6.7 Practical Conclusion
For a real fleet rollout, begin with a simple, monitorable architecture and progressively add robustness mechanisms only when justified by measured instability, drift, or communication failures.
