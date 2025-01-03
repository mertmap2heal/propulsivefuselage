import numpy as np
import matplotlib.pyplot as plt
import math

A = 2.3  # Disk area (m^2)
v1 = 231  # Free-stream velocity - Cruise speed of the A320 (m/s)
vi = 200  # Induced velocity through the disk (m/s)
ηdisk = 0.95  # Disk efficiency
ηmotor = 0.95  # Motor efficiency
ηprop = 0.97  # Propeller efficiency
ηTotal = ηmotor * ηprop * ηdisk  # Total efficiency
v2 = v1 + 2 * vi  # Velocity far downstream

def calculate_atmospheric_properties(FL):
    # Constants
    g_s = 9.80665  # m/s^2
    R = 287.05  # J/(kg*K)

    # Reference values
    T_MSL = 288.15  # K
    p_MSL = 101325  # Pa
    rho_MSL = 1.225  # kg/m^3

    H_MSL = 0  # m
    H_G11 = 11000  # m
    H_G20 = 20000  # m

    T_11 = 216.65  # K
    p_11 = 22632  # Pa
    rho_11 = 0.364  # kg/m^3

    T_20 = 216.65  # K
    p_20 = 5474.88  # Pa
    rho_20 = 0.088  # kg/m^3

    gamma_Tropo = -0.0065  # K/m
    gamma_LowerStr = 0.0  # K/m (Lower stratosphere is isothermal)
    gamma_UpperStr = 0.001  # K/m (Upper stratosphere lapse rate)

    n_tropo = 1 / (1 + gamma_Tropo * R / g_s)
    n_lower = 1  # Isothermal layer
    n_upper = 1 / (1 + gamma_UpperStr * R / g_s)

    # Convert FL to height (in meters)
    H_G = FL * 100

    if H_G <= H_G11:  # Troposphere
        T = T_MSL * (1 - ((n_tropo - 1) / n_tropo) * (g_s / (R * T_MSL)) * H_G)
        p = p_MSL * (1 - ((n_tropo - 1) / n_tropo) * (g_s / (R * T_MSL)) * H_G) ** (n_tropo / (n_tropo - 1))
        rho = rho_MSL * (1 - ((n_tropo - 1) / n_tropo) * (g_s / (R * T_MSL)) * H_G) ** (1 / (n_tropo - 1))

    elif H_G <= H_G20:  # Lower Stratosphere 
        T = T_11
        p = p_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))
        rho = rho_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))

    else:  # Upper Stratosphere
        T = T_20 * (1 - ((n_upper - 1) / n_upper) * (g_s / (R * T_20)) * (H_G - H_G20))
        p = p_20 * (1 - ((n_upper - 1) / n_upper) * (g_s / (R * T_20)) * (H_G - H_G20)) ** (n_upper / (n_upper - 1))
        rho = rho_20 * (1 - ((n_upper - 1) / n_upper) * (g_s / (R * T_20)) * (H_G - H_G20)) ** (1 / (n_upper - 1))

    return T, p, rho


FL = float(input("Enter Flight Level (FL): "))  # e.g., FL350 for 35000 feet
T, p, rho = calculate_atmospheric_properties(FL)
print('---------Chapter 1: Flight Conditions------------------------')
print(f"Temperature: {T:.2f} K")
print(f"Pressure: {p:.2f} Pa")
print(f"Density: {rho:.6f} kg/m^3")


#---------------------------------------# 
# Section 1 - Basic Actuator Disk Model #
# Energy demand to drive the propulsive fuselage engine
#---------------------------------------# 

class ActuatorDiskModel:
    def __init__(self, rho, A, v1, vi, ηdisk, ηmotor, ηprop):
        self.rho = rho
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
        return self.rho * self.A * self.vi

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

print('---------Chapter 2: Basic Actuator Disk Model------------------------')
BasicModelValues = ActuatorDiskModel(rho, A, v1, vi, ηdisk, ηmotor, ηprop)
mdot, T, P_disk, P_total = BasicModelValues.display_results()
print('---------------------------------------------------------------------')

#--------------------------------------------------# 
# Section 3 - Drag Generation by BLI Engine #
#--------------------------------------------------# 
print('---------Chapter 3: Drag Generated by BLI Engine------------------------')


class DragbyBLIEngine:
    def __init__(self, lnac, dnac, T, p, v1):
        self.lnac = lnac  # Nacelle length (m)
        self.dnac = dnac  # Nacelle diameter (m)
        self.T = T        # Temperature (K)
        self.p = p        # Pressure (Pa)
        self.v1 = v1      # Free-stream velocity (m/s)
        
    def calculate_zero_lift_drag(self):
        """Calculate zero lift drag based on nacelle properties."""
        #--------Dynamic Viscosity----------------#
        mu = (18.27 * 10**-6) * (411.15 / (self.T + 120)) * (self.T / 291.15) ** 1.5  # Sutherland's law
        k = 10 * 10**-6  # Surface finish coefficient for aluminum

        #--------Reynolds Number------------------#
        Re = (self.p * self.v1 * self.lnac) / mu
        Re0 = 38 * (self.lnac / k) ** 1.053  # Cutoff Reynolds number
        if Re > Re0:
            Re = Re0
            print("Reynolds Number exceeded cutoff Reynolds Number.")

        # Friction drag coefficient for laminar flow
        Cf = 1.328 / math.sqrt(Re)

        #--------Nacelle Parasite Drag Area-------#
        f = self.lnac / self.dnac
        Fnac = 1 + 0.35 / f  # Nacelle parasite drag factor
        Snacwet = math.pi * self.dnac * self.lnac  # Wetted area of the nacelle
        fnacparasite = Cf * Fnac * Snacwet  # Nacelle Parasite Drag

        # Zero lift drag
        Dzero = 0.5 * self.p * self.v1**2 * fnacparasite
        return Dzero

# Example input values for the drag calculation
lnac = 2.0  # Length of the nacelle (m)
dnac = 0.5  # Diameter of the nacelle (m)

BLIEngine = DragbyBLIEngine(lnac, dnac, T, p, v1)
Dzero = BLIEngine.calculate_zero_lift_drag()

 
print(f"Zero Lift Drag (Dzero): {Dzero:.2f} N")
print('---------------------------------------------------------------------')


#--------------------------------------------------# 
# Section 4 - Visualization of Actuator Disk Model #
#--------------------------------------------------# 
 
 
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)

 
V_before = v1 * np.ones_like(X)
V_disk = vi * ((X > -0.1) & (X < 0.1))  # Distinct vi region, showing velocity drop on the actuator line
V_after = BasicModelValues.v2 * ((X >= 0.1) & (X <= 0.5))  # Ensure v2 starts just after the actuator disk
V_far = v1 * (X > 0.5)  # Free stream velocity resumes as v1 after the v2 region

# Combine velocity fields
V_field = V_before * (X < -0.1) + V_disk * ((X >= -0.1) & (X <= 0.1)) + V_after * ((X > 0.1) & (X <= 0.5)) + V_far

# Plot the updated visualization
plt.figure(figsize=(12, 6))
plt.contourf(X, Y, V_field, levels=50, cmap='RdYlBu_r')  # Adjusted color map for better visibility of vi
plt.colorbar(label="Velocity (m/s)")
plt.plot([0, 0], [-1, 1], color='black', linewidth=2, label="Actuator Disk")

# Add streamlines to indicate flow direction
U_flow = V_field  # X-velocity components
V_flow = np.zeros_like(U_flow)  # No Y-direction flow
plt.streamplot(X, Y, U_flow, V_flow, color="white", linewidth=0.5, density=2)

plt.title("Updated Flow Visualization - Actuator Disk Model")
plt.xlabel("Distance (arbitrary units)")
plt.ylabel("Height (arbitrary units)")
plt.legend()
plt.grid(False)
plt.show()

print('---------End of Actuator Disk and Drag Analysis------------------------')
