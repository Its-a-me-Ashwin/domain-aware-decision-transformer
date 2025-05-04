# Domain-Aware Decision Transformer (DADT)

This repository contains the official code and experimental framework for the paper:

**"Domain-Aware Decision Transformers for Generalizable Policy Learning"**

We present a transformer-based architecture that conditions policy predictions on both the environment state and a latent domain configuration. This design enables robust generalization across diverse physics parameters and environments in reinforcement learning tasks.

## Overview

Prior approaches to handling domain variability in RL fall into two main categories:

- **Domain Randomization**: Train a single policy over a large distribution of domain parameters, hoping the resulting policy generalizes reasonably across the entire distribution.
- **Domain-Specific Optimization**: Train a high-performing policy for one specific environment with precise simulation fidelity, which often fails to generalize outside its trained domain.

We propose an alternative method:
> Learn a domain-conditioned transformer policy that adapts to the current environment by either observing or inferring the underlying domain configuration.

## Environments

The model is evaluated on three simulated environments with domain variability:

1. **Ball-on-Plate Balancer**  
   The agent must center a ball on a tiltable plate. Domains vary by surface friction, external wind force, and action scaling.

2. **Quadrupedal Locomotion (Unitree Go1)**  
   The robot must walk forward on terrains with varying slope, friction, weight, and torque constraints.

3. **[Environment 3 Placeholder]**  
   Additional benchmarks can be included for further testing of generalization behavior.

Each environment includes a range of domain configurations that affect the transition dynamics.

## Model Architecture

The proposed model extends the Decision Transformer to include domain conditioning. At each timestep, the model takes as input:

