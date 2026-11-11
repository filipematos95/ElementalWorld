# HybridEcosystem: Computational Architecture & Theoretical Framework

A hybrid Cellular Automata (CA) and Agent-Based Model (ABM) framework implemented in TensorFlow for simulating multi-elemental biogeochemical ecosystems, stoichiometric niches, spatial nutrient transport, climate seasonality, and multi-scale disturbance regimes.

---

## Act I: The Bio-Elemental Foundation

### 1.1 The Elementome Vector Definition
The simulation tracks bio-elemental stoichiometry across two linked elemental spaces:

1. **Organic Elemental Vector ($\mathbf{N}_{\text{org}} = 11$):**
   $$\mathbf{E}_{\text{org}} = [\text{C}, \text{N}, \text{P}, \text{K}, \text{O}, \text{Ca}, \text{Mg}, \text{S}, \text{Fe}, \text{Mn}, \text{Zn}]$$
   Every individual organism $i$ maintains an internal elemental stoichiometry vector $E_i \in \mathbb{R}^{11}$ strictly normalized to sum to 1.0:
   $$\sum_{k=1}^{11} E_{i,k} = 1.0$$

2. **Soil Inorganic Nutrients ($\mathbf{N}_{\text{soil}} = 10$):**
   $$\mathbf{E}_{\text{soil}} = [\text{N}, \text{P}, \text{K}, \text{O}, \text{Ca}, \text{Mg}, \text{S}, \text{Fe}, \text{Mn}, \text{Zn}]$$
   Carbon ($\text{C}$) is excluded from soil pools as it is acquired via atmospheric fixation during growth. An indexing mapping vector `org_to_soil_idx` projects organic elemental pools to their soil counterparts.

---

### 1.2 Elemental Dissimilarity & Stoichiometric Diversity
Elemental diversity across agents and species is quantified using Mahalanobis dissimilarity metrics ($ED_m$), which account for covariance between multiple elemental concentrations:

* **Agent-Level Dissimilarity Index:**
  Evaluates individual-level stoichiometric variance relative to the population mean stoichiometry ($\bar{E}$):
  $$D_{m}(i, j) = \sqrt{(E_i - E_j)^T \mathbf{\Sigma}_{\text{pop}}^{-1} (E_i - E_j)}$$
  $$ED_m^{\text{agent}} = \frac{1}{\mathbf{N}_{\text{org}}} \sum_{i} \sum_{j} \min(p_i, p_j) D_m(i, j)$$
  where $p_i = \frac{M_i}{\sum M}$ is the relative biomass weight of agent $i$.

* **Species-Level Dissimilarity Index:**
  Evaluates inter-specific stoichiometric divergence by calculating weighted distances between species mean elemental centroids ($\bar{E}_s$):
  $$ED_m^{\text{species}} = \frac{1}{\mathbf{N}_{\text{org}}} \sum_{s_1} \sum_{s_2} \min(p_{s1}, p_{s2}) \sqrt{(\bar{E}_{s1} - \bar{E}_{s2})^T \mathbf{\Sigma}_{\text{spp}}^{-1} (\bar{E}_{s1} - \bar{E}_{s2})}$$

---

### 1.3 Biogeochemical Niche Space
Each species $s \in \{1, \dots, N_{\text{spp}}\}$ occupies a distinct biogeochemical niche defined in $11$-dimensional elemental space by:

1. **Niche Center ($\mathbf{\mu}_s \in \mathbb{R}^{11}$):** Normalized target stoichiometry representing optimal physiological composition.
2. **Tolerance Covariance Matrix ($\mathbf{\Sigma}_s \in \mathbb{R}^{11 \times 11}$):** Defines stoichiometric flexibility and co-variation tolerances among elements.

---

### 1.4 Niche Fitness Mechanics
Physiological performance is governed by an agent's stoichiometric deviation from its species niche center, evaluated via Mahalanobis distance ($D_M$):

$$D_M(E_i, \mathbf{\mu}_s) = \sqrt{(E_i - \mathbf{\mu}_s)^T \mathbf{\Sigma}_s^{-1} (E_i - \mathbf{\mu}_s)}$$

**Niche Fitness Function ($f_i \in [0, 1]$):**
$$f_i = \text{clip}\left(1.0 - \left(\frac{D_M(E_i, \mathbf{\mu}_s)}{\sigma_{\text{threshold}}}\right)^2, \, 0.0, \, 1.0\right)$$
where $\sigma_{\text{threshold}} = 3.0$. Fitness reaches $1.0$ when internal stoichiometry perfectly aligns with the species niche optimum, and drops quadratically to $0.0$ as stoichiometric imbalance grows.

---

## Act II: Spatial & Computational Architecture

### 2.1 Hybrid Coupling: Biotic ABM + Abiotic CA Grid
The simulation couples two computational layers operating over a discrete spatial grid of size $H \times W$:

1. **Biotic Layer (Agent-Based Tensor):**
   Agents are stored in a contiguous 2D tensor buffer $\mathbf{A} \in \mathbb{R}^{\text{MAX\_AGENTS} \times 16}$:
   $$\mathbf{A}_i = [y_i, \, x_i, \, \text{species}_i, \, M_i, \, E_{i,0}, \dots, E_{i,10}, \, \text{alive}_i, \, \text{age}_i]$$
   Position $(y_i, x_i)$ is stored as continuous coordinates mapped to grid cells.

2. **Abiotic Layer (Cellular Automata Grid Variable):**
   Soil chemical states are stored in a 3D variable tensor $\mathbf{S} \in \mathbb{R}^{H \times W \times 20}$:
   $$\mathbf{S}(y, x) = [\mathbf{I}(y, x), \; \mathbf{O}(y, x)]$$
   where $\mathbf{I} \in \mathbb{R}^{10}$ represents available inorganic nutrients, and $\mathbf{O} \in \mathbb{R}^{10}$ represents soil organic nutrient pools.

---

### 2.2 Soil Nutrient Pools & Chemical States
* **Inorganic Pool ($\mathbf{I}$):** Mineralized, biologically available nutrients ($\text{N}, \text{P}, \text{K}, \dots, \text{Zn}$).
* **Organic Pool ($\mathbf{O}$):** Immobilized organic matter derived from plant turnover and dead biomass, awaiting microbial mineralization.

---

### 2.3 Carrying Capacities & Physical Space Constraints
Cellular crowdedness is constrained by two distinct carrying capacity mechanisms:

1. **Biomass Capacity Constraint ($K_{\text{biomass}}$):**
   Space availability factor $S(y,x)$ reduces individual growth as local biomass density approaches $K_{\text{biomass}} = 1.5$:
   $$S(y, x) = \max\left(0.0, \, 1.0 - \frac{\sum_{i \in (y,x)} M_i}{K_{\text{biomass}}}\right)$$

2. **Occupancy Limit (`MAX_AGENTS_PER_CELL`):**
   Hard cap on individual count per cell ($N_{\text{max\_cell}} = 10$). Seed establishment fails if a target cell already contains $\ge 10$ live agents.

---

### 2.4 Soil Nutrient Transport & Spatial Diffusion
Spatial diffusion of inorganic soil nutrients across neighboring cells is calculated every timestep using 2D depthwise convolution with symmetric padding:

$$\mathbf{I}_{\text{diffused}} = \text{Conv2D}\left(\mathbf{I}, \, \mathbf{K}_{\text{diff}}, \, \text{padding}=\text{'SAME'}\right)$$

where $\mathbf{K}_{\text{diff}}$ is a $3 \times 3$ isotropic diffusion kernel:
$$\mathbf{K}_{\text{diff}} = \begin{bmatrix} 0.05 & 0.10 & 0.05 \\ 0.10 & 0.40 & 0.10 \\ 0.05 & 0.10 & 0.05 \end{bmatrix}$$

**Inorganic Soil Influx & Drift:**
Soil receives background inorganic inputs subject to stochastic compositional drift ($\sigma_{\text{drift}} = 0.08$):
$$\mathbf{I}_{\text{new}} = \mathbf{I}_{\text{diffused}} + \left(\mathbf{R}_{\text{drifted}} \times \text{soil\_input\_rate}\right)$$

---

## Act III: Metabolism, Growth Kinetics & Nutrient Uptake

### 3.1 Target Growth & Temperature Modifiers
Desired biomass growth ($\Delta M_{\text{desired}}$) is driven by niche fitness ($f_i$), individual biomass ($M_i$), base growth rate ($\mu = 0.7$), and local temperature modulation ($T(y, x)$):

$$\Delta M_{\text{desired}} = f_i \cdot \mu \cdot M_i \cdot \left(1.0 + \gamma_{\text{growth}} \cdot (T(y,x) - 0.5)\right)$$

Target uptake is divided into:
* **Carbon Demand (Atmospheric Fixation):** $\Delta C_{\text{demand}} = \Delta M_{\text{desired}} \cdot E_{i,\text{C}}$
* **Soil Element Demands ($\mathbf{D}_{\text{soil}} \in \mathbb{R}^{10}$):** Remaining growth requirement proportioned according to species niche preference stoichiometry:
$$\mathbf{D}_{\text{soil}} = (\Delta M_{\text{desired}} - \Delta C_{\text{demand}}) \cdot \mathbf{\hat{\mu}}_{s,\text{soil}}$$

---

### 3.2 Michaelis-Menten Uptake Kinetics & Deficits
Soil nutrient uptake ($\mathbf{U}_{\text{soil}} \in \mathbb{R}^{10}$) follows saturating Michaelis-Menten kinetics ($K_m = 0.1$):

$$\mathbf{U}_{\text{soil}} = \mathbf{D}_{\text{soil}} \odot \left( \frac{\mathbf{I}_{\text{available}}}{\mathbf{I}_{\text{available}} + K_m} \right)$$

where $\mathbf{I}_{\text{available}} = \mathbf{I}_{\text{new}} \odot \text{soil\_availability\_rate}$.

**Nutrient Deficit Tracking:**
Unfulfilled nutrient demand is accumulated as a per-species deficit tensor:
$$\mathbf{Deficit}_s = \sum_{i \in \text{spp } s} \max(0.0, \, \mathbf{D}_{\text{soil}, i} - \mathbf{U}_{\text{soil}, i})$$

---

### 3.3 Internal Quotas & Liebig's Law of the Minimum
Elements absorbed from soil and atmospheric carbon form an updated internal elemental pool:
$$\mathbf{P}_{\text{elem}} = M_i \cdot E_i + \mathbf{U}_{\text{full}}$$

Internal elemental quota relative to biomass:
$$Q_k = \frac{P_{\text{elem}, k}}{M_i + \epsilon}$$

Realized growth is bottlenecked by the most limiting element according to **Liebig's Law of the Minimum** relative to a minimum quota threshold ($Q_{\min} = 0.01$):
$$g_{\text{limit}} = \min_{k} \left( \max\left(0.0, \, 1.0 - \frac{Q_{\min}}{Q_k + \epsilon}\right) \right)$$

**Realized Growth Rate:**
$$\Delta M_{\text{realized}} = \Delta M_{\text{actual}} \cdot g_{\text{limit}} \cdot S(y, x)$$

---

### 3.4 Respiration, Net Biomass & Demographic Noise
Metabolic maintenance respiration drains biomass as a function of temperature:
$$R_{\text{maint}} = M_i \cdot r_{\text{resp}} \cdot \left(1.0 + \gamma_{\text{resp}} \cdot (T(y,x) - 0.5)\right)$$

**Net Biomass Update:**
$$M_{\text{final}} = M_i + \Delta M_{\text{realized}} - R_{\text{maint}} + \mathcal{N}(0, \sigma_{\text{demo}})$$
where $r_{\text{resp}} = 0.02$ and $\sigma_{\text{demo}} = 0.003$.

---

## Act IV: Demographics, Climate & Multi-Scale Disturbances

### 4.1 Demographic Dynamics, Senescence & Seed Dispersal

1. **Age Accumulation & Sigmoidal Senescence:**
   Agent age increments by $1.0$ each step. Mortality probability increases exponentially/sigmoidally as age exceeds expected maximum age ($M_{\text{age}} = 500.0$):
   $$P(\text{senescence}) = \frac{1}{1 + \exp\left(-\frac{\text{age} - M_{\text{age}}}{0.1 \cdot M_{\text{age}}}\right)}$$
   An agent dies if $M_{\text{final}} \le 0.01$ or if a uniform random roll $U(0,1) < P(\text{senescence})$.

2. **Fertility & Seed Production:**
   Live agents with $M_{\text{alive}} > C_{\text{seed}}$ ($C_{\text{seed}} = 0.3$) produce seeds with probability scaled by niche fitness:
   $$P(\text{seed}) = 0.1 \cdot f_i$$
   Parent biomass is deducted by $C_{\text{seed}}$ upon reproduction.

3. **Power-Law Dispersal Radius & Offspring Establishment:**
   Dispersal radius $R_{\text{disp}}$ scales with species seed mass ($M_{\text{seed}, s}$):
   $$R_{\text{disp}} = \text{seed\_range\_scale} \cdot (M_{\text{seed}, s} + \epsilon)^\alpha$$
   Offspring land at integer displacement $(\Delta y, \Delta x) \in [-R_{\text{disp}}, R_{\text{disp}}]$ using periodic boundary conditions (`% H`, `% W`).
   Establishment succeeds only if:
   * Target cell biomass density allows space: $U(0, 1) < \left(1.0 - \frac{\text{Density}_{\text{target}}}{K_{\text{biomass}}}\right)$
   * Current cell agent count $< \text{MAX\_AGENTS\_PER\_CELL}$.

---

### 4.2 Biogeochemical Recycling & Soil Mineralization

1. **Litter Turnover & Mortality Influx:**
   Metabolic turnover ($r_{\text{turnover}} = 0.02$) and dead agent biomass enter local soil organic pools:
   $$\mathbf{Loss}_{\text{full}} = \left(M_{\text{dead}} + M_{\text{alive}} \cdot r_{\text{turnover}}\right) \otimes Q_i$$
   Organic nutrients update locally:
   $$\mathbf{O}_{\text{tot}}(y,x) = \mathbf{O}(y,x) + \mathbf{Loss}_{\text{full}}[\text{org\_to\_soil\_idx}]$$

2. **Temperature-Modulated Soil Mineralization:**
   Organic nutrients are mineralized into inorganic pools based on local temperature:
   $$\mathbf{Flux}_{\text{mineral}} = \mathbf{O}_{\text{tot}}(y,x) \cdot r_{\text{min}} \cdot \left(1.0 + \gamma_{\text{min}} \cdot (T(y,x) - 0.5)\right)$$
   $$\mathbf{O}_{\text{final}} = \mathbf{O}_{\text{tot}} - \mathbf{Flux}_{\text{mineral}}$$
   $$\mathbf{I}_{\text{final}} = \mathbf{I}_{\text{new}} + \mathbf{Flux}_{\text{mineral}} - \mathbf{U}_{\text{soil}}$$

---

### 4.3 Climate Seasonality & Spatial Environmental Fields

1. **Sinusoidal Thermal Seasonality:**
   Global baseline temperature cycles sinusoidally over time step $t$:
   $$T_{\text{global}}(t) = T_{\text{mean}} + T_{\text{amp}} \cdot \sin\left(\frac{2\pi (t + \phi)}{T_{\text{period}}}\right)$$
   Spatial microclimates modulate global temperature across grid cells:
   $$T(y, x, t) = \text{clip}\left(T_{\text{global}}(t) + \gamma_{\text{spatial}} \cdot (\mathbf{E}_{\text{field}}(y,x) - 0.5), \, 0.0, \, 1.0\right)$$

2. **Persistent AR(1) Spatial Fields:**
   Two smooth 2D fields evolve continuously via autoregressive persistence and 2D average pooling:
   * **`env_field` ($\mathbf{E}_{\text{field}}$):** Persistent background spatial heterogeneity ($P_{\text{env}} = 0.85$).
   * **`shock_field` ($\mathbf{S}_{\text{field}}$):** Spatial catastrophe susceptibility field ($P_{\text{shock}} = 0.85$).
   $$\mathbf{F}_{t+1} = \text{Norm}\left(\rho \mathbf{F}_t + (1 - \rho) \cdot \text{AvgPool2D}(\mathcal{N}(0, 1))\right)$$

---

### 4.4 Multi-Scale Disturbance Regimes
The ecosystem experiences four distinct disturbance mechanisms targeting abiotic nutrients or phenotypic vulnerability:

#### 1. Spatial Patch Disturbances (Abiotic Nutrient Depletion)
* **Trigger:** Stochastic occurrence with probability $p_{\text{disturb}} = 0.01$ per step at random coordinate $(c_y, c_x)$.
* **Mechanics:** Strips inorganic nutrients within radius $R_{\text{disturb}} = 8$ modulated by local `env_field`:
$$\mathbf{I}_{\text{disturbed}}(y,x) = \mathbf{I}_{\text{new}}(y,x) \cdot \left(1.0 - \text{patch}(y,x) \cdot \text{strength} \cdot \mathbf{E}_{\text{field}}(y,x)\right)$$

#### 2. Systemic Catastrophes
* **Trigger:** Occurs periodically every `catastrophe_interval` (default: $200$ steps).
* **Mechanics:** Mass mortality shock scaled spatially by `shock_field`:
$$P(\text{die}_{\text{cat}}) = \text{mortality}_{\text{cat}} \cdot \mathbf{S}_{\text{field}}(y, x)$$

#### 3. Weak Disturbances (Fitness-Selective Environmental Filtering)
* **Trigger:** Occurs periodically every `weak_disturbance_interval`.
* **Mechanics:** Selectively culls organisms poorly adapted to their biogeochemical niche (low fitness $f_i$):
$$P(\text{die}_{\text{weak}}) = \text{mortality}_{\text{weak}} \cdot \mathbf{E}_{\text{field}}(y,x) \cdot (1.0 - f_i)^{\alpha_{\text{weak}}}$$
* **Ecological Impact:** Stabilizing selection; clears maladapted individuals from fragile environmental microclimates.

#### 4. Strong Disturbances (Biomass-Selective Competitive Resets)
* **Trigger:** Occurs periodically every `strong_disturbance_interval`.
* **Mechanics:** Selectively targets large, biomass-dominant individuals based on relative mass score ($M_{\text{score}} \in [0, 1]$):
$$M_{\text{score}, i} = \frac{M_i - M_{\min}}{M_{\max} - M_{\min} + \epsilon}$$
$$P(\text{die}_{\text{strong}}) = \text{mortality}_{\text{strong}} \cdot \mathbf{S}_{\text{field}}(y,x) \cdot (M_{\text{score}, i})^{\beta_{\text{strong}}}$$
* **Ecological Impact:** Competitive reset; prunes hyper-dominant agents holding massive nutrient quotas, freeing up space and resources to maintain species co-existence.

### Summary Table of Disturbance Dynamics

| Disturbance Type | Target Layer | Core Driver / Conditioning Factor | Ecological Role |
| :--- | :--- | :--- | :--- |
| **Patch Disturbance** | Abiotic (Soil) | `env_field` & Euclidean Distance Radius ($R=8$) | Localized resource pulses and nutrient depletion patches. |
| **Catastrophe** | Biotic (Agents) | `shock_field` & Fixed Interval Mortality | Periodic system-wide population bottlenecks. |
| **Weak Disturbance** | Biotic (Agents) | **Maladaptation** ($(1-f)^\alpha$) & `env_field` | Environmental filtering; culls organisms out of their niche. |
| **Strong Disturbance** | Biotic (Agents) | **Biomass Dominance** ($\text{MassScore}^\beta$) & `shock_field` | Competitive reset; culls hyper-dominant agents to maintain diversity. |