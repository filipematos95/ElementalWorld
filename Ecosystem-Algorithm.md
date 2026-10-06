# Hybrid Ecosystem Model: Algorithm Description

**Version**: 1.0  
**Date**: February 24, 2026  
**Model Type**: Individual-Based Spatial Ecosystem Simulation

---

## 1. Overview

The **HybridEcosystem** model simulates multi-species plant competition on a 2D spatial grid with explicit biogeochemical cycling. Each time step advances both **soil nutrient dynamics** (physics) and **agent lifecycles** (biology) using vectorized TensorFlow operations for GPU acceleration.

---

## 2. Data Structures

### 2.1 Spatial Grid
- **Dimensions**: H × W pixels
- **Soil State** (H × W × 8):
  - Channels 0-3: Inorganic [N, P, K, O] (plant-available)
  - Channels 4-7: Organic [N, P, K, O] (litter pool)

### 2.2 Agent Buffer
- **Shape**: (MAX_AGENTS, 10)
- **Columns**:
  - 0: y-coordinate (row)
  - 1: x-coordinate (column)
  - 2: species_id (integer)
  - 3: mass (biomass)
  - 4-8: elementome [C, N, P, K, O] (ratios, sum=1.0)
  - 9: alive (boolean flag)

### 2.3 Species Parameters
- **niche_centers** (N_species × 5): Optimal elementome
- **niche_left** (N_species × 5): Tolerance for deficiency
- **niche_right** (N_species × 5): Tolerance for excess

---

## 3. The `step()` Algorithm

### Input
- `fitness_metric` ∈ {`'euclidean'`, `'chebyshev'`, `'cosine'`}

### Output
- Updated `self.soil`, `self.agents`, `self.n_agents`

---

### **PHASE 1: SOIL PHYSICS**

#### 1.1 Nutrient Diffusion
```
FOR each inorganic nutrient i ∈ {N, P, K, O}:
    inorg_diffused[i] ← convolve(inorg[i], diff_kernel)
```
- **Kernel**: 3×3 Gaussian (center=0.4, neighbors=0.1/0.05)
- **Boundary**: Symmetric padding (Neumann)

#### 1.2 External Input
```
inorg_new ← inorg_diffused + soil_input_rate × soil_base_ratio
```
- Represents atmospheric deposition, rainfall

---

### **PHASE 2: AGENT BIOLOGY**

#### 2.1 Filter Active Agents
```
active_mask ← agents[:, 9] > 0.5
active_data ← gather(agents, where(active_mask))
```

IF no active agents:
```
    APPLY mineralization: inorg ← inorg_new + mineralization_rate × org
    RETURN n_agents
```

---

#### 2.2 Biogeochemical Niche Fitness

**For each active agent j:**

```
delta ← elementome[j] - niche_center[species[j]]
tolerance ← IF delta[i] < 0 THEN niche_left[species[j], i] 
                            ELSE niche_right[species[j], i]
norm_dev ← |delta| / tolerance
```

**Fitness Calculation:**

- **Euclidean** (Gaussian penalty):
  ```
  fitness[j] ← exp(-0.5 × Σ(norm_dev²))
  ```

- **Chebyshev** (Liebig's Law):
  ```
  fitness[j] ← exp(-2.3 × max(norm_dev)²)
  ```

- **Cosine** (Angle similarity):
  ```
  weights ← 1 / tolerance²
  similarity ← (Σ weights × elementome[j] × center) / 
               (||elementome[j]||_w × ||center||_w)
  fitness[j] ← max(0, similarity)^20
  ```

**Output**: `fitness[j] ∈ [0, 1]`

---

#### 2.3 Resource Uptake

**Desired Growth:**
```
desired_growth[j] ← fitness[j] × growth_rate × mass[j]
```

**Carbon Uptake** (atmospheric, unlimited):
```
c_uptake[j] ← desired_growth[j] × elementome_C[j]
```

**Soil Nutrient Uptake** (N, P, K, O):
```
remaining[j] ← desired_growth[j] - c_uptake[j]
soil_available ← inorg_new[coords[j], :]
soil_ratio ← soil_available / Σ(soil_available)

FOR each nutrient i ∈ {N, P, K, O}:
    desired[i] ← remaining[j] × soil_ratio[i]

IF ANY(desired > soil_available):
    actual_growth[j] ← 0  # All-or-nothing rule
ELSE:
    actual_growth[j] ← desired_growth[j]
    uptake[N,P,K,O] ← desired
```

**Update Element Pools:**
```
pool_C ← mass[j] × elementome_C[j] + c_uptake[j]
pool_N ← mass[j] × elementome_N[j] + uptake_N[j]
... (similarly for P, K, O)
```

---

#### 2.4 Growth Limitation

**Quota Limitation:**
```
FOR each element i ∈ {C, N, P, K, O}:
    Q[i] ← pool[i] / mass[j]
    quota_factor[i] ← max(0, 1 - MIN_QUOTA / Q[i])

limit_quota ← min(quota_factor)
```

**Space Limitation:**
```
local_biomass ← Σ(mass of all agents at coords[j])
space_factor ← max(0, 1 - local_biomass / K_biomass)
```

**Realized Growth:**
```
realized_growth[j] ← actual_growth[j] × limit_quota × space_factor
```

**Maintenance Respiration:**
```
maintenance[j] ← respiration_rate × mass[j]
```

**Final Mass:**
```
new_mass[j] ← mass[j] + realized_growth[j] - maintenance[j]
alive[j] ← new_mass[j] > 0.01
```

---

#### 2.5 Recycling (Mortality + Turnover)

**Death Flux:**
```
IF alive[j] == FALSE:
    dead_mass[j] ← max(0, new_mass[j])
ELSE:
    dead_mass[j] ← 0

turnover_mass[j] ← turnover_rate × new_mass[j] × alive[j]
total_loss[j] ← dead_mass[j] + turnover_mass[j]
```

**Element Recycling:**
```
FOR each element i ∈ {C, N, P, K, O}:
    loss[i] ← total_loss[j] × Q[i]
    pool[i] ← pool[i] - loss[i]
    
    # Scatter to organic soil at coords[j]
    org[coords[j], i] += loss[i]
```

**Update Stoichiometry:**
```
surviving_mass[j] ← new_mass[j] - turnover_mass[j]

FOR each element i:
    elementome[j, i] ← pool[i] / surviving_mass[j]
    elementome[j, i] ← clip(elementome[j, i], 0.01, 0.99)

# Renormalize to sum = 1.0
elementome[j] ← elementome[j] / Σ(elementome[j])
```

---

#### 2.6 Soil Nutrient Update

**Mineralization** (organic → inorganic):
```
FOR each nutrient i ∈ {N, P, K, O}:
    flux[i] ← mineralization_rate × org[:, :, i]
    org[:, :, i] ← org[:, :, i] - flux[i]
    inorg[:, :, i] ← inorg_new[:, :, i] + flux[i] - uptake_grid[i]
```

**Prevent negative concentrations:**
```
inorg ← max(0, inorg)
```

---

#### 2.7 Reproduction

**Fertility Check:**
```
FOR each agent j WHERE alive[j] == TRUE:
    IF mass[j] > seed_cost AND random() < 0.1:
        mass[j] ← mass[j] - seed_cost
        
        # Spawn offspring
        new_y ← (y[j] + random_int(-1, 2)) mod H
        new_x ← (x[j] + random_int(-1, 2)) mod W
        
        CREATE new_agent:
            coords ← (new_y, new_x)
            species ← species[j]
            mass ← seed_mass
            elementome ← elementome[j]  # Inherit parent ratios
            alive ← TRUE
        
        IF buffer_not_full:
            ADD new_agent to agents buffer
            n_agents ← n_agents + 1
```

---

#### 2.8 Dead Agent Cleanup

**Compact Buffer:**
```
living_mask ← (agents[:, 9] > 0.5) AND (index < n_agents)
living_agents ← agents[living_mask]

agents ← [living_agents, zeros(MAX_AGENTS - count(living_agents), 10)]
n_agents ← count(living_agents)
```

---

## 4. Parameter Summary

| Parameter | Symbol | Typical Value | Role |
|-----------|--------|---------------|------|
| Growth rate | `r` | 0.15 | Max biomass increase per step |
| Respiration | `m` | 0.01 | Maintenance cost (constant) |
| Turnover | `τ` | 0.03 | Litterfall rate (alive agents) |
| Mineralization | `μ` | 0.05 | Decomposition rate (organic → inorganic) |
| Seed cost | `S_cost` | 0.02 | Biomass deducted from parent |
| Seed mass | `S_mass` | 0.02 | Initial offspring biomass |
| Carrying capacity | `K` | 2.5 | Max biomass per pixel |
| Soil input | `I` | 1.5 | External nutrient deposition rate |

---

## 5. Computational Complexity

**Per Time Step:**
- Soil diffusion: O(H × W)
- Agent operations: O(n_agents)
- Grid scattering: O(n_agents)

**Total**: O(n_agents + H × W)

**Parallelization**: All operations are vectorized using TensorFlow → GPU acceleration possible.

---

## 6. Key Design Features

1. **Asymmetric Niche Tolerance**: Different sensitivities to nutrient deficiency vs excess
2. **Multi-Metric Fitness**: Supports Euclidean (smooth), Chebyshev (strict), Cosine (angle-based)
3. **Quota-Based Growth**: All elements must meet minimum ratio (no partial growth)
4. **Dual Mortality**: Starvation (fin_mass < 0.01) + routine turnover
5. **Local Dispersal**: ±1 pixel prevents unrealistic global mixing
6. **Closed Nutrient Cycles**: All element losses return to soil

---

## 7. Pseudo-Code Summary

```
ALGORITHM: Ecosystem.step(fitness_metric)

# Phase 1: Soil
inorg ← diffuse(inorg) + external_input
IF no_agents: RETURN after mineralization

# Phase 2: Agents
FOR each active agent j:
    fitness ← compute_niche_fitness(elementome[j], niche_params[species[j]], metric)
    actual_growth ← uptake_resources(fitness, mass[j], inorg[coords[j]])
    realized_growth ← apply_limits(actual_growth, quota, space)
    new_mass ← mass[j] + realized_growth - maintenance
    alive ← new_mass > threshold
    
    IF NOT alive OR turnover_event:
        RECYCLE elements to org[coords[j]]
    
    IF fertile AND lucky:
        SPAWN offspring at nearby pixel

# Phase 3: Soil Update
inorg ← inorg + mineralize(org) - total_uptake

# Phase 4: Cleanup
COMPACT agent buffer (remove dead)

RETURN n_agents
```

---

## 8. References

- **Niche Theory**: Hutchinson, G.E. (1957). *Concluding remarks.* Cold Spring Harbor Symposia.
- **Liebig's Law**: Chebyshev fitness implementation
- **Droop Model**: Quota-based growth limitation
- **Lotka-Volterra**: Density dependence mechanism

---

**End of Document**
