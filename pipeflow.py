# Calculate the flow rate of water in a pipe given the pipe diameter, length, and pressure difference

import numpy as np

g = 9.81  
rho = 1000  
mu = 0.001  

diameter = 0.05  
length = 10  
pressure_diff = 5000  

area = np.pi * (diameter / 2)**2
velocity = (pressure_diff / (2 * g * length * (rho / mu)))**0.5
flow_rate = area * velocity

print(f"Flow rate: {flow_rate:.2f} m^3/s")
