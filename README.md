# Active Perception Framework for Flow State Estimation in Autonomous Underwater Vehicles

## Overview

In this project, we've developed an advanced Rao-Blackwellized Particle Filter (RBPF) framework designed to enhance the precision of flow state estimation and vehicle localization in dynamic environments. Our approach draws upon and expands the foundational work presented by [Chang et al. 2022](https://ieeexplore.ieee.org/abstract/document/9679478).

### Approaches

1. **Cascaded Filters Approach**:
   - This method employs a combination of Kalman Filter (KF) and Particle Filter (PF) in a sequential manner, where the KF is dedicated to flow state estimation and the PF focuses on vehicle state estimation.

2. **RBPF (Rao-Blackwellized Particle Filter)**:
   - In this approach, each particle is equipped with its own KF, enabling it to predict and estimate the flow state independently. This strategy notably improves the framework's overall performance by leveraging the strengths of both filtering techniques.

### Configuration

To activate or deactivate the RBPF feature:

- Navigate to the `test_params.yaml` file.
- Toggle the `rbpf` parameter accordingly.

## Installation Instructions

Follow these steps to install the package:

```bash
cd ~
git clone git@github.com:rakeshv24/rob599-mobile-robotics.git
sudo -H pip3 install -e ~/rob599-mobile-robotics
