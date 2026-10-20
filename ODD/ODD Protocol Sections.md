# ODD Protocol (ODD 2020): Sections Explained

The **ODD protocol** (*Overview, Design Concepts, and Details*) is a standardized framework for describing agent-based and simulation models. Its goal is to improve model transparency, reproducibility, and communication among researchers.

The protocol is organized into three main sections:a

1. Overview
2. Design Concepts
3. Details

---

# 1. Overview

This section describes what the model is, what it does, and which entities it contains.

## 1.1 Purpose and Patterns

### Purpose

Describes the scientific objectives and motivations of the model.

Questions typically addressed:

- Why was the model developed?
- Which hypotheses are being tested?
- What scientific questions are being investigated?

**Example**

> The model investigates whether permafrost warming can trigger critical transitions in ecosystem carbon cycling.

### Patterns

Describes empirical or theoretical patterns that the model reproduces or seeks to explain.

Examples include:

- Temporal trends
- Spatial patterns
- Statistical relationships
- Experimental observations

**Example**

> The model reproduces observed increases in ecosystem respiration under warming conditions.

---

## 1.2 Entities, State Variables, and Scales

### Entities

All objects represented in the model.

Examples:

- Landscape cells
- Individual plants
- Trees
- Herbivores
- Households

### State Variables

Variables that characterize each entity.

| Entity | State Variables |
|----------|----------|
| Cell | Temperature, soil moisture, nutrients |
| Plant | Biomass, age, species, carbon storage |

### Scales

Defines the model resolution.

**Temporal scale**

- Daily
- Monthly
- Annual

**Spatial scale**

- Individual-based
- Grid cells
- Landscapes

**Organizational scale**

- Individuals
- Populations
- Ecosystems

**Example**

> The model operates on a 100 × 100 grid and updates annually for 500 years.

---

## 1.3 Process Overview and Scheduling

Describes:

- Which processes occur
- The order in which they occur
- Their frequency

Typical annual schedule:

1. Climate update
2. Plant growth
3. Competition
4. Mortality
5. Decomposition
6. Carbon accounting

This section focuses on workflow and timing rather than equations.

---

# 2. Design Concepts

This section describes the conceptual mechanisms that generate model behavior.

ODD 2020 includes eleven design concepts.

---

## 2.1 Basic Principles

Describes the theories or concepts underlying the model.

Examples:

- Resource competition theory
- Metabolic theory
- Resilience theory
- Alternative stable states
- Niche theory

**Example**

> Ecosystem dynamics emerge from interactions between vegetation growth, nutrient cycling, and climate forcing.

---

## 2.2 Emergence

Describes system-level properties that arise from lower-level interactions.

Examples:

- Forest dieback
- Carbon source transitions
- Tipping points
- Species coexistence

Emergent properties are not explicitly programmed but result from model processes.

---

## 2.3 Adaptation

Describes how agents modify behavior in response to environmental conditions.

Examples:

- Resource allocation changes
- Behavioral adjustments
- Movement decisions

If adaptation is absent, state:

> Adaptation is not represented.

---

## 2.4 Objectives

Describes goals pursued by agents.

Examples:

- Maximize growth
- Maximize fitness
- Acquire resources
- Minimize mortality

Many ecological models do not include explicit objectives.

---

## 2.5 Learning

Describes whether agents learn from past experience.

Examples:

- Memory-based decisions
- Reinforcement learning
- Evolutionary adaptation

If absent:

> Learning is not represented.

---

## 2.6 Prediction

Describes whether agents anticipate future conditions.

Examples:

- Human planning
- Crop management decisions
- Economic forecasting

Most ecological ABMs do not include prediction.

---

## 2.7 Sensing

Describes what information agents can perceive.

Examples:

- Temperature
- Resource availability
- Local competition
- Neighbor density

Sensing determines the information available for decision-making.

---

## 2.8 Interaction

Describes how entities influence each other.

Examples:

- Competition
- Facilitation
- Predation
- Nutrient sharing

**Example**

> Plant growth decreases with increasing neighboring biomass due to competition for resources.

---

## 2.9 Stochasticity

Describes all random components of the model.

Examples:

- Random mortality
- Climate variability
- Trait initialization
- Seed dispersal

This section should specify:

- Which processes are stochastic
- Probability distributions used
- Why stochasticity is included

---

## 2.10 Collectives

Describes groups that behave as a unit.

Examples:

- Herds
- Colonies
- Social groups

If absent:

> Collectives are not represented.

---

## 2.11 Observation

Describes the outputs recorded during simulations.

Examples:

- Biomass
- Species richness
- GPP
- Reco
- NEP
- Spatial variance
- Lag-1 autocorrelation (AR(1))
- Interannual variability (IAV)

This section explains how model outputs are linked to analyses and hypotheses.

---

# 3. Details

This section provides the technical implementation of the model.

---

## 3.1 Initialization

Describes model starting conditions.

Examples:

- Initial biomass
- Initial nutrient pools
- Initial species distributions
- Initial climate conditions

**Example**

> Simulations begin from a mature forest equilibrium at a mean annual temperature of -5°C.

---

## 3.2 Input Data

Describes external data supplied to the model during simulations.

Examples:

- Climate observations
- Soil maps
- Remote sensing products
- Experimental forcing data

If external data are not used:

> The model operates without external input data.

---

## 3.3 Submodels

Provides detailed descriptions of all model processes.

Typically includes:

- Equations
- Algorithms
- Parameters
- Process-specific assumptions

Examples:

### Plant Growth

\[
GPP = f(T, N, L)
\]

where:

- \(T\) = temperature
- \(N\) = nutrient availability
- \(L\) = light availability

### Mortality

\[
P_{mort} = 1 - e^{-kS}
\]

### Decomposition

\[
Reco = R_0 e^{\beta T}
\]

### Nutrient Cycling

\[
N_{t+1} = N_t + M - D
\]

Submodels are usually the most technical and extensive part of an ODD description.

---

# Typical Reviewer Questions and Corresponding ODD Sections

| Reviewer Question | ODD Section |
|------------------|-------------|
| Why was the model built? | Purpose |
| What entities are represented? | Entities, State Variables, and Scales |
| How does the simulation run? | Process Overview and Scheduling |
| Which ecological theory supports the model? | Basic Principles |
| Where do emergent dynamics originate? | Emergence |
| Which processes are random? | Stochasticity |
| What outputs are analyzed? | Observation |
| How are the processes implemented mathematically? | Submodels |

---

# ODD Sections Most Important for Ecosystem Tipping Point Models

For models focused on resilience, critical transitions, permafrost thaw, forest dieback, and carbon cycle dynamics, particular attention should be given to:

1. Purpose and Patterns
2. Basic Principles
3. Emergence
4. Interaction
5. Stochasticity
6. Observation
7. Submodels

These sections explain how ecosystem-scale regime shifts emerge from local ecological processes and how resilience indicators such as AR(1), variance, and interannual variability are calculated and analyzed.