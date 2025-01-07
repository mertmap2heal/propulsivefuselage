import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

#---------------------------------------# 
# Section 1 - Flight Conditions #
#---------------------------------------# 

class FlightConditions:
    def calculate_atmospheric_properties(self, FL):
        # Constants
        g_s = 9.80665  # m/s^2
        R = 287.05  # J/(kg*K)

        # Reference values
        T_MSL = 288.15  # K
        p_MSL = 101325  # Pa
        rho_MSL = 1.225  # kg/m^3

        H_G11 = 11000  # m
        H_G20 = 20000  # m

        T_11 = 216.65  # K
        p_11 = 22632  # Pa
        rho_11 = 0.364  # kg/m^3

        T_20 = 216.65  # K
        p_20 = 5474.88  # Pa
        rho_20 = 0.088  # kg/m^3

        gamma_Tropo = -0.0065  # K/m
        gamma_UpperStr = 0.001  # K/m (Upper stratosphere lapse rate)

        # Convert FL to height (in meters)
        H_G = FL * 100

        if H_G <= H_G11:  # Troposphere
            T = T_MSL + gamma_Tropo * H_G
            p = p_MSL * (T / T_MSL) ** (-g_s / (gamma_Tropo * R))
            rho = rho_MSL * (T / T_MSL) ** (-g_s / (gamma_Tropo * R) - 1)

        elif H_G <= H_G20:  # Lower Stratosphere
            T = T_11
            p = p_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))
            rho = rho_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))

        else:  # Upper Stratosphere
            T = T_20 + gamma_UpperStr * (H_G - H_G20)
            p = p_20 * (T / T_20) ** (-g_s / (gamma_UpperStr * R))
            rho = rho_20 * (T / T_20) ** (-(g_s / (gamma_UpperStr * R) + 1))

        return T, p, rho

    def calculate_free_stream_velocity(self, Mach, FL):
        kappa = 1.4  # Ratio of specific heats for air, constant
        R = 287.05  # J/(kg*K)
        
        # Temperature property from the previous method
        T, _, _ = self.calculate_atmospheric_properties(FL)
        # Speed of sound
        a = math.sqrt(kappa * R * T)
        
        # Free-stream velocity
        v_freestream = Mach * a
        v_inlet = v_freestream
        return v_inlet, v_freestream


FL = float(input("Enter Flight Level (FL): "))  
Mach = float(input("Enter Mach Number: "))  

flight_conditions = FlightConditions()
T, p, rho = flight_conditions.calculate_atmospheric_properties(FL)
v_inlet, v_freestream = flight_conditions.calculate_free_stream_velocity(Mach, FL)
 

print('---------Chapter 1: Flight Conditions-----------------------------------')
print(f"Temperature: {T:.2f} K")
print(f"Pressure: {p:.2f} Pa")
print(f"Density: {rho:.6f} kg/m³")
print(f'Free-stream velocity: {v_freestream:.2f} m/s')
print('------------------------------------------------------------------------')

#---------------------------------------# 
# Section 2 - Nacelle Geometry #
#---------------------------------------# 
class NacelleParameters:
    def __init__(self):
        self.A_inlet = 2  # Engine Inlet Area (m^2)
        self.v_inlet = 230  # Free-stream velocity - Av. Cruise speed of the A320 (m/s)
        self.ηdisk = 0.95  # Disk efficiency
        self.ηmotor = 0.95  # Motor efficiency
        self.ηprop = 0.97  # Propeller efficiency
        self.ηTotal = self.ηmotor * self.ηprop * self.ηdisk  # Total efficiency
        self.nac_length = 2.286  # Nacelle length (m), from engine spec
        
    def variable_parameters(self, v_inlet, A_inlet):

        A_disk = self.A_inlet / 0.9
        Inlet_radius = math.sqrt(self.A_inlet / math.pi)
        Disk_radius = math.sqrt(A_disk / math.pi)
        v_disk = (self.A_inlet * self.v_inlet) / A_disk

        v_exhaust = self.v_inlet + 2 * v_disk  # Comes from actuator disk theory
        A_exhaust = A_disk * (v_disk / v_exhaust)
        Exhaust_radius = math.sqrt(A_exhaust / math.pi)

        print('---------Chapter 2: Nacelle Geometry -----------------------------------')
       
        print('Inlet Radius:', Inlet_radius, 'm')
        print('Inlet Area:', A_inlet, 'm^2')
        print('Inlet Velocity:', v_inlet, 'm/s')

        print('Disk Radius:', Disk_radius, 'm')
        print('Disk Area:', A_disk, 'm^2')
        print('Disk Velocity:', v_disk, 'm/s')

        print('Exhaust Radius:', Exhaust_radius, 'm')
        print('Exhaust Area:', A_exhaust, 'm^2')
        print('Exhaust Velocity:', v_exhaust, 'm/s')
        print('-------------------------------------------------------------------------')
    
        return self.A_inlet, A_disk, A_exhaust, Inlet_radius, Exhaust_radius, v_disk, v_exhaust, self.v_inlet, Disk_radius

nacelle = NacelleParameters()  
A_inlet = nacelle.A_inlet

#---------------------------------------# 
# Section 3 - Basic Actuator Disk Model #
#---------------------------------------# 

class ActuatorDiskModel:
    def __init__(self, rho, A_disk, v_inlet, v_disk, ηdisk, ηmotor, ηprop):
        self.rho = rho
        self.A_disk= A_disk
        self.v_inlet = v_inlet
        self.v_disk = v_disk
        self.ηdisk = ηdisk
        self.ηmotor = ηmotor
        self.ηprop = ηprop
        self.ηTotal = ηmotor * ηprop * ηdisk
        self.v_exhaust = v_inlet + 2 * v_disk

    def calculate_mass_flow_rate(self):
        """Calculate the mass flow rate (mdot)."""
        return self.rho * self.A_disk* self.v_disk

    def calculate_thrust(self, mdot):
        """Calculate the thrust (T)."""
        return mdot * (self.v_exhaust - self.v_inlet)

    def calculate_power_disk(self, T):
        """Calculate the power required at the disk (P_disk)."""
        return T * (self.v_inlet + self.v_disk)

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

        return mdot, T, P_disk, P_total, self.A_disk


# From Chapter 1: FlightConditions
rho = flight_conditions.calculate_atmospheric_properties(FL)[2]   
v_inlet = v_inlet  

# From Chapter 2: NacelleParameters
A_inlet = nacelle.A_inlet  
results_nacelle = nacelle.variable_parameters(v_inlet, A_inlet)  
A_disk = results_nacelle[1]   
v_disk = results_nacelle[5]   
ηdisk = nacelle.ηdisk   
ηmotor = nacelle.ηmotor  
ηprop = nacelle.ηprop  
 
print('---------Chapter 3: Basic Actuator Disk Model------------------------')
BasicModelValues = ActuatorDiskModel(rho, A_disk, v_inlet, v_disk, ηdisk, ηmotor, ηprop)


mdot, T, P_disk, P_total, A_disk = BasicModelValues.display_results()

#--------------------------------------------------# 
# Section 4 - Drag Generation by BLI Engine #
#--------------------------------------------------# 

class DragbyBLIEngine:
    def __init__(self, flight_conditions, nacelle_params, FL, Mach):
        # From FlightConditions Class
        self.T, self.p, _ = flight_conditions.calculate_atmospheric_properties(FL)  # Temperature and pressure
        self.v_inlet, _ = flight_conditions.calculate_free_stream_velocity(Mach, FL)  # Free-stream velocity

        # From NacelleParameters Class
        self.nac_length = nacelle_params.nac_length  # Nacelle length
        self.A_inlet = nacelle_params.A_inlet  # Inlet area
        self.inlet_radius = math.sqrt(self.A_inlet / math.pi)  # Nacelle diameter derived from inlet area
        
    def calculate_zero_lift_drag(self):
        # Dynamic Viscosity using Sutherland's law
        mu = (18.27 * 10**-6) * (411.15 / (self.T + 120)) * (self.T / 291.15) ** 1.5
        k = 10 * 10**-6  # Surface finish coefficient for aluminum

        # Reynolds Number
        Re = (self.p * self.v_inlet * self.nac_length) / mu
        Re0 = 38 * (self.nac_length / k) ** 1.053  # Cutoff Reynolds number
        if Re > Re0:
            Re = Re0
            print("Reynolds Number exceeded cutoff Reynolds Number.")

        # Friction drag coefficient for laminar flow
        Cf = 1.328 / math.sqrt(Re)

        # Nacelle Parasite Drag Area
        f = self.nac_length / (2 * self.inlet_radius)
        Fnac = 1 + 0.35 / f  # Nacelle parasite drag factor
        Snacwet = math.pi * 2 * self.inlet_radius * self.nac_length  # Wetted area of the nacelle
        fnacparasite = Cf * Fnac * Snacwet  # Nacelle Parasite Drag

        # Zero lift drag
        Dzero = 0.5 * self.p * self.v_inlet**2 * fnacparasite
        return Dzero

# Initialize the DragbyBLIEngine class
bli_engine = DragbyBLIEngine(flight_conditions, nacelle, FL, Mach)

# Calculate zero-lift drag
Dzero = bli_engine.calculate_zero_lift_drag()
 
print('---------Chapter 4: Drag Generated by BLI Engine------------------------')
print(f"Zero Lift Drag (Dzero): {Dzero:.2f} N")
print('------------------------------------------------------------------------')


#--------------------------------------------------# 
# Section 5 - Visualization of the Engine Model #
#--------------------------------------------------# 

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


class NacelleVisualization:
    def __init__(self, A_inlet, A_disk, A_exhaust, v_inlet, v_disk, v_exhaust, nac_length, inlet_radius, disk_radius, exhaust_radius):
        self.A_inlet = A_inlet
        self.A_disk = A_disk
        self.A_exhaust = A_exhaust
        self.v_inlet = v_inlet
        self.v_disk = v_disk
        self.v_exhaust = v_exhaust
        self.nac_length = nac_length
        self.inlet_radius = inlet_radius
        self.disk_radius = disk_radius
        self.exhaust_radius = exhaust_radius

        # Geometry lengths
        self.extra_length = 2
        self.l_intake = 1.5
        self.l_engine = 2.5
        self.l_exhaust = self.nac_length - (self.l_intake + self.l_engine)
        self.disk_location = self.l_intake + 0.5  # Disk is inside the engine section

    def calculate_geometry(self):
        self.x = np.linspace(-self.extra_length, self.nac_length + self.extra_length, 700)

        # Outer radius profile
        self.outer_radius = np.piecewise(
            self.x,
            [
                self.x < 0,
                (self.x >= 0) & (self.x < self.l_intake),
                (self.x >= self.l_intake) & (self.x < self.disk_location),
                (self.x >= self.disk_location) & (self.x < self.l_intake + self.l_engine),
                (self.x >= self.l_intake + self.l_engine) & (self.x <= self.nac_length),
                self.x > self.nac_length
            ],
            [
                lambda x: self.inlet_radius,
                lambda x: self.inlet_radius + (self.disk_radius - self.inlet_radius) * (x / self.l_intake),
                lambda x: self.disk_radius,
                lambda x: self.disk_radius - (self.disk_radius - self.exhaust_radius) * ((x - self.disk_location) / (self.l_engine - (self.disk_location - self.l_intake))),
                lambda x: self.exhaust_radius,
                lambda x: self.exhaust_radius
            ]
        )

        # Velocity profile
        self.velocities = np.piecewise(
            self.x,
            [
                self.x < 0,
                (self.x >= 0) & (self.x < self.l_intake),
                (self.x >= self.l_intake) & (self.x < self.disk_location),
                (self.x >= self.disk_location) & (self.x < self.l_intake + self.l_engine),
                (self.x >= self.l_intake + self.l_engine) & (self.x <= self.nac_length),
                self.x > self.nac_length
            ],
            [
                lambda x: self.v_inlet,
                lambda x: self.v_inlet - 10,
                lambda x: self.v_disk - 20,
                lambda x: self.v_disk + 10,
                lambda x: self.v_exhaust,
                lambda x: self.v_exhaust
            ]
        )

    def plot_geometry(self, canvas_frame):
        self.calculate_geometry()

        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.cm.plasma
        norm = Normalize(vmin=min(self.velocities), vmax=max(self.velocities))
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        ax.fill_between(self.x, -self.outer_radius, self.outer_radius, color="lightgray", alpha=0.5, label="Nacelle Geometry")
        ax.plot(self.x, self.outer_radius, color="darkred", linewidth=2)
        ax.plot(self.x, -self.outer_radius, color="darkred", linewidth=2)

        for i in range(len(self.x) - 1):
            ax.fill_between(
                [self.x[i], self.x[i + 1]],
                [-self.outer_radius[i], -self.outer_radius[i + 1]],
                [self.outer_radius[i], self.outer_radius[i + 1]],
                color=cmap(norm(self.velocities[i])),
                edgecolor="none",
                alpha=0.7
            )

        fan_x = self.disk_location
        ax.plot([fan_x, fan_x], [-self.disk_radius, self.disk_radius], color="black", linewidth=2, linestyle="--", label="Fan (Disk Location)")

        exhaust_start = self.l_intake + self.l_engine
        ax.plot([exhaust_start, exhaust_start], [-self.exhaust_radius, self.exhaust_radius], color="orange", linewidth=2, linestyle="--", label="Exhaust Boundary")
        ax.fill_between([exhaust_start, self.nac_length], [-self.exhaust_radius, -self.exhaust_radius], [self.exhaust_radius, self.exhaust_radius], color="orange", alpha=0.3, label="Exhaust Air")

        ax.text(-self.extra_length / 2, self.inlet_radius + 0.3, f"Inlet\nArea: {self.A_inlet:.2f} m²", color="blue", fontsize=10, ha="center")
        ax.text(fan_x, self.disk_radius + 0.3, f"Fan (Disk)\nArea: {self.A_disk:.2f} m²", color="green", fontsize=10, ha="center")
        ax.text((exhaust_start + self.nac_length) / 2, self.exhaust_radius + 0.3, f"Exhaust\nArea: {self.A_exhaust:.2f} m²", color="orange", fontsize=10, ha="center")

        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Velocity (m/s)', fontsize=12)

        ax.set_title("2D Nacelle Geometry with Velocity and Section Representation", fontsize=16)
        ax.set_xlabel("Length (m)", fontsize=12)
        ax.set_ylabel("Radius (m)", fontsize=12)
        ax.legend()
        ax.grid()

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


class NacelleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nacelle Visualization")

        # Initialize FL and Mach as instance variables
        self.FL = None
        self.Mach = None

        # Input Frame
        input_frame = tk.Frame(root, padx=10, pady=10)
        input_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(input_frame, text="Enter Flight Level (FL):").pack()
        self.fl_entry = tk.Entry(input_frame)
        self.fl_entry.pack()

        tk.Label(input_frame, text="Enter Mach Number:").pack()
        self.mach_entry = tk.Entry(input_frame)
        self.mach_entry.pack()

        submit_btn = tk.Button(input_frame, text="Visualize", command=self.visualize)
        submit_btn.pack(pady=10)

        # Canvas Frame for Plot
        self.canvas_frame = tk.Frame(root, padx=10, pady=10)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Text Output Frame
        self.output_frame = tk.Frame(root, padx=10, pady=10)
        self.output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(self.output_frame, wrap=tk.WORD, width=40, height=30)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def visualize(self):
        # Clear previous output
        self.output_text.delete("1.0", tk.END)

        # Get user inputs
        self.FL = float(self.fl_entry.get())
        self.Mach = float(self.mach_entry.get())

        # Initialize classes
        flight_conditions = FlightConditions()
        nacelle = NacelleParameters()

        # Chapter 1: Flight Conditions
        T, p, rho = flight_conditions.calculate_atmospheric_properties(self.FL)
        v_inlet, v_freestream = flight_conditions.calculate_free_stream_velocity(self.Mach, self.FL)
        self.output_text.insert(tk.END, 'Chapter 1: Flight Conditions\n')
        self.output_text.insert(tk.END, f"Temperature: {T:.2f} K\n")
        self.output_text.insert(tk.END, f"Pressure: {p:.2f} Pa\n")
        self.output_text.insert(tk.END, f"Density: {rho:.6f} kg/m³\n")
        self.output_text.insert(tk.END, f'Free-stream velocity: {v_freestream:.2f} m/s\n')
        self.output_text.insert(tk.END, '------------------------------------------------------------------------\n')

        # Chapter 2: Nacelle Parameters
        A_inlet = nacelle.A_inlet
        results_nacelle = nacelle.variable_parameters(v_inlet, A_inlet)
        A_disk = results_nacelle[1]
        v_disk = results_nacelle[5]
        v_exhaust = results_nacelle[6]
        self.output_text.insert(tk.END, 'Chapter 2: Nacelle Geometry\n')
        self.output_text.insert(tk.END, f"Inlet Radius: {results_nacelle[3]:.2f} m\n")
        self.output_text.insert(tk.END, f"Inlet Area: {results_nacelle[0]:.2f} m²\n")
        self.output_text.insert(tk.END, f"Disk Area: {A_disk:.2f} m²\n")
        self.output_text.insert(tk.END, f"Exhaust Area: {results_nacelle[2]:.2f} m²\n")
        self.output_text.insert(tk.END, '------------------------------------------------------------------------\n')

        # Chapter 3: Actuator Disk Model
        actuator_model = ActuatorDiskModel(rho, A_disk, v_inlet, v_disk, 
                                           nacelle.ηdisk, nacelle.ηmotor, nacelle.ηprop)
        mdot, T, P_disk, P_total, _ = actuator_model.display_results()
        self.output_text.insert(tk.END, 'Chapter 3: Basic Actuator Disk Model\n')
        self.output_text.insert(tk.END, f"Mass flow rate (mdot): {mdot:.2f} kg/s\n")
        self.output_text.insert(tk.END, f"Thrust (T): {T:.2f} N\n")
        self.output_text.insert(tk.END, f"Power required at the disk (P_disk): {P_disk:.2f} W\n")
        self.output_text.insert(tk.END, f"Total efficiency (ηTotal): {nacelle.ηTotal:.2f}\n")
        self.output_text.insert(tk.END, f"Total electrical power required (P_total): {P_total:.2f} W\n")
        self.output_text.insert(tk.END, '------------------------------------------------------------------------\n')

        # Chapter 4: Drag by BLI Engine
        bli_engine = DragbyBLIEngine(flight_conditions, nacelle, self.FL, self.Mach)
        Dzero = bli_engine.calculate_zero_lift_drag()
        self.output_text.insert(tk.END, 'Chapter 4: Drag Generated by BLI Engine\n')
        self.output_text.insert(tk.END, f"Zero Lift Drag (Dzero): {Dzero:.2f} N\n")
        self.output_text.insert(tk.END, '------------------------------------------------------------------------\n')

        # Visualization
        visualization = NacelleVisualization(
            A_inlet=nacelle.A_inlet,
            A_disk=A_disk,
            A_exhaust=results_nacelle[2],
            v_inlet=v_inlet,
            v_disk=v_disk,
            v_exhaust=v_exhaust,
            nac_length=nacelle.nac_length,
            inlet_radius=results_nacelle[3],
            disk_radius=results_nacelle[8],
            exhaust_radius=results_nacelle[4]
        )
        visualization.plot_geometry(self.canvas_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = NacelleApp(root)
    root.mainloop()

