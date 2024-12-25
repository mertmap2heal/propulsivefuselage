import numpy as np
import matplotlib.pyplot as plt

# Constants and parameters for Airbus A320-200
p = 0.3  # Average air density between FL330 and FL390 (kg/m^3)
A = 2.3  # Disk area (m^2)
v1 = 231  # Free-stream velocity - Cruise speed of the A320 (m/s)
vi = 200  # Induced velocity through the disk (m/s)
ηdisk = 0.95  # Disk efficiency
ηmotor = 0.95  # Motor efficiency
ηprop = 0.97  # Propeller efficiency
ηTotal = ηmotor * ηprop * ηdisk  # Total efficiency
v2 = v1 + 2 * vi  # Velocity far downstream

#---------------------------------------# 
# Section 1 - Basic Actuator Disk Model #
# Energy demand to drive the propulsive fuselage engine
""" 
Inlet size
Generated thrust
Efficiency
    -As a function of heat
    -As a function of inlet freestream velocity
Exhaust size
Fuel burn rate
"""
#---------------------------------------# 

class ActuatorDiskModel:
    def __init__(self, p, A, v1, vi, ηdisk, ηmotor, ηprop):
        self.p = p
        self.A = A
        self.v1 = v1
        self.vi = vi
        self.ηdisk = ηdisk
        self.ηmotor = ηmotor
        self.ηprop = ηprop
        self.ηTotal = ηmotor * ηprop * ηdisk
        self.v2 = v1 + 2 * vi

    def calculate_mass_flow_rate(self):
        """Calculate the mass flow rate (mdot)."""
        return self.p * self.A * self.vi

    def calculate_thrust(self, mdot):
        """Calculate the thrust (T)."""
        return mdot * (self.v2 - self.v1)

    def calculate_power_disk(self, T):
        """Calculate the power required at the disk (P_disk)."""
        return T * (self.v1 + self.vi)

    def calculate_total_power(self, P_disk):
        """Calculate the total electrical power required (P_total)."""
        return P_disk / self.ηTotal

    def display_results(self):
        """All calculations and the results."""
        mdot = self.calculate_mass_flow_rate()
        T = self.calculate_thrust(mdot)
        P_disk = self.calculate_power_disk(T)
        P_total = self.calculate_total_power(P_disk)
        
        print("Mass flow rate (mdot):", mdot, "kg/s")
        print("Thrust (T):", T, "N")
        print("Power required at the disk (P_disk):", P_disk, "W")
        print("Total efficiency (ηTotal):", self.ηTotal)
        print("Total electrical power required (P_total):", P_total, "W")

        return mdot, T, P_disk, P_total

print('---------Chapter 1: Basic Actuator Disk Model------------------------')
BasicModelValues = ActuatorDiskModel(p, A, v1, vi, ηdisk, ηmotor, ηprop)
mdot, T, P_disk, P_total = BasicModelValues.display_results()
print('---------------------------------------------------------------------')
print('---------Chapter 2: Derived Values------------------------')


#---------------------------------------# 
# Section 1 - Basic Actuator Disk Model # 
#---------------------------------------# 


#--------------------------------------------------# 
# Section 2 - Visualization of Actuator Disk Model # 
#--------------------------------------------------# 

x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)

V_before = v1 * np.ones_like(X)
V_disk = vi * np.exp(-X**2 * 5)  
V_after = BasicModelValues.v2 * np.exp(-(X-1)**2 * 5)  

V_field = V_before * (X < -0.5) + V_disk * ((X >= -0.5) & (X <= 0.5)) + V_after * (X > 0.5)

plt.figure(figsize=(12, 6))
plt.contourf(X, Y, V_field, levels=50, cmap='coolwarm')
plt.colorbar(label="Velocity (m/s)")

plt.plot([0, 0], [-1, 1], color='black', linewidth=2, label="Actuator Disk")

# streamlines to indicate flow direction
U_flow = V_field  # X-velocity components
V_flow = np.zeros_like(U_flow)  # No Y-direction flow
plt.streamplot(X, Y, U_flow, V_flow, color="white", linewidth=0.5, density=2)

plt.title("Flow Visualization - Actuator Disk Model")
plt.xlabel("Distance (arbitrary units)")
plt.ylabel("Height (arbitrary units)")
plt.legend()
plt.grid(False)
plt.show()

#--------------------------------------------------# 
# Section 2 - Visualization of Actuator Disk Model # 
#--------------------------------------------------# 

