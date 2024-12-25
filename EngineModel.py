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

# ===============================
# Chapter 1: Most Basic Actuator Disk Model
# ===============================

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

    def display_results(self):
        """All calculations and the results."""
        mdot = self.calculate_mass_flow_rate()
        T = self.calculate_thrust(mdot)
        P_disk = self.calculate_power_disk(T)

        print("Mass flow rate (mdot):", mdot, "kg/s")
        print("Thrust (T):", T, "N")
        print("Power required at the disk (P_disk):", P_disk, "W")
        print("Total efficiency (ηTotal):", self.ηTotal)
        return mdot, T, P_disk


# ===============================
# Chapter 2: Derived Values from the Basic Model
# ===============================

class DerivedValues:
    def __init__(self, mdot, T, P_disk, ηTotal):
        self.mdot = mdot
        self.ηTotal = ηTotal
        self.T = T
        self.P_disk = P_disk

    def calculate_total_power(self):
        """Calculate the total electrical power required (P_total)."""
        return self.P_disk / self.ηTotal

    def display_results(self):
        """Display all calculations and the results."""
        P_total = self.calculate_total_power()
        print("Total electrical power required (P_total):", P_total, "W")
        return


# ===============================
# Chapter 2: Derived Values from the Basic Model
# =============================== 



print('---------Chapter 1: Basic Actuator Disk Model------------------------')
BasicModelValues = ActuatorDiskModel(p, A, v1, vi, ηdisk, ηmotor, ηprop)
mdot, T, P_disk = BasicModelValues.display_results()  
print('---------Chapter 2: Derived Values------------------------')
Values = DerivedValues(mdot, T, P_disk, BasicModelValues.ηTotal)
Values.display_results()
 