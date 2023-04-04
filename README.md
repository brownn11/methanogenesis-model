# methanogenesis-model
Simple model of methanogenesis ($3H_2 + CO \leftarrow CH_4 + H_2O$) in a tube-in-tube flow reactor. Users can control tube length and radius, reaction rate, initial concentrations, membrane permeability, and use of a carrier gas. 

### Structure:

The tube starts with an initial concentration for $CO$ in the gas tube, and 3x that concentration for $H_2$. It uses a Gaussian diffusion along with membrane permeability to begin diffusion from the gas phase to the liquid phase. Once in the liquid phase, the reaction ($3H_2 + CO \leftarrow CH_4 + H_2O$) can begin ($H_2O$ is ignored as this occurs in the liquid phase). The amount of produced $CH_4$ is diffused across the liquid phase and begins to diffuse into the gas phase until reaching equilibrium. This continues until either all $H_2$ and $CO$ is consumed, or until the flow reaches the end of the tube.
