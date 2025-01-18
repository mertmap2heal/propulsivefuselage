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

        n_trop = 1.235 # n constant at Troposphere
        n_uStr = 0.001 # n constant at Upper Stratosphere

        H_G11 = 11000  # m
        H_G20 = 20000  # m

        T_11 = 216.65  # K
        p_11 = 22632  # Pa
        rho_11 = 0.364  # kg/m^3

        T_20 = 216.65  # K
        p_20 = 5474.88  # Pa
        rho_20 = 0.088  # kg/m^3

        gamma_Tropo = -0.0065  # K/m
        gamma_UpperStr = 0.001  # K/m

        H_G = FL * 0.3048  # Convert feet to meters

        if H_G <= H_G11:  # Troposphere
            T = T_MSL * (1 + (gamma_Tropo / T_MSL) * H_G)
            p = p_MSL * (1 + (gamma_Tropo / T_MSL) * H_G) ** (n_trop / (n_trop - 1))
            rho = rho_MSL * (1 + (gamma_Tropo / T_MSL) * H_G) ** (1 / (n_trop - 1))

        elif H_G <= H_G20:  # Lower Stratosphere
            T = T_11
            p = p_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))
            rho = rho_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))

        else:  # Upper Stratosphere
            T = T_20 * (1 + (gamma_UpperStr / T_20) * (H_G - H_G20))
            p = p_20 * (1 + (gamma_UpperStr / T_20) * (H_G - H_G20)) ** (n_uStr / (n_uStr - 1))
            rho = rho_20 * (1 - ((n_uStr - 1) / n_uStr) * (g_s / (R * T_20)) * (H_G - H_G20)) ** (1 / (n_uStr - 1))

        return T, p, rho

    def calculate_free_stream_velocity(self, Mach, FL):
        # Constants
        kappa = 1.4  # Adiabatic constant for air
        R = 287.05  # J/(kg*K)

        # Atmospheric properties
        T, _, _ = self.calculate_atmospheric_properties(FL)

        # Speed of sound
        a = math.sqrt(kappa * R * T)

        # Free-stream velocity
        v_freestream = Mach * a
        v_inlet = v_freestream

        return v_inlet, v_freestream

#---------------------------------------# 
# Section 2 - Nacelle Geometry #
#---------------------------------------# 
class NacelleParameters:
    def __init__(self, v_inlet, A_inlet):
        self.A_inlet = A_inlet # Engine Inlet Area (m^2)
        self.v_inlet = v_inlet # Inlet Velocity (m/s)
        self.ηdisk = 0.95 # Disk Efficiency
        self.ηmotor = 0.95  # Motor Efficiency
        self.ηprop = 0.97 # Propeller Efficiency
        self.ηTotal = self.ηmotor * self.ηprop * self.ηdisk
        self.nac_length = 2.8 # Nacelle Length (m)

    def variable_parameters(self):
        A_disk = self.A_inlet * 0.9 # Disk Area (m^2)
        Inlet_radius = math.sqrt(self.A_inlet / math.pi) # Inlet Radius (m)
        Disk_radius = math.sqrt(A_disk / math.pi) # Disk Radius (m)
        v_disk = (self.A_inlet * self.v_inlet) / A_disk # Disk Velocity (m/s)

        v_exhaust = 2 * v_disk-self.v_inlet   # Exhaust Velocity (m/s)
        A_exhaust = A_disk * (v_disk / v_exhaust) # Exhaust Area (m^2)
        Exhaust_radius = math.sqrt(A_exhaust / math.pi) # Exhaust Radius (m)
 
        return self.A_inlet, A_disk, A_exhaust, Inlet_radius, Exhaust_radius, v_disk, v_exhaust, self.v_inlet, Disk_radius

#---------------------------------------# 
# Section 3 - Basic Actuator Disk Model #
#---------------------------------------# 
class ActuatorDiskModel:
    def __init__(self, rho, A_disk, v_inlet, v_disk ):
        self.rho = rho # Air Density (kg/m^3)
        self.A_disk = A_disk # Disk Area (m^2)
        self.v_inlet = v_inlet # Inlet Velocity (m/s)
        self.v_disk = v_disk # Disk Velocity (m/s)
        self.ηdisk = 0.95 # Disk Efficiency
        self.ηmotor = 0.95  # Motor Efficiency
        self.ηprop = 0.97 # Propeller Efficiency
        self.ηTotal = self.ηmotor * self.ηprop * self.ηdisk
        self.v_exhaust = 2 *self.v_disk-self.v_inlet # Exhaust Velocity (m/s)


    def calculate_mass_flow_rate(self):   # Mass flow rate (mdot)
        mdot = self.rho * self.A_disk * self.v_disk
        return mdot

    def calculate_thrust(self, mdot): # Thrust (T)
        T = mdot * (self.v_exhaust - self.v_inlet)
        return T

    def calculate_power_disk(self, T): # Power required at the disk (P_disk)
        P_disk = T * (self.v_disk)*10**-3
        return P_disk

    def calculate_total_power(self, P_disk): # Total electrical power required (P_total) after the efficieny losses
        P_total = (P_disk) / (self.ηTotal)
        return P_total


    def display_results(self):
        mdot = self.calculate_mass_flow_rate()
        T = self.calculate_thrust(mdot)
        P_disk = self.calculate_power_disk(T)
        P_total = self.calculate_total_power(P_disk)

        return mdot, T, P_disk, P_total, self.A_disk


#---------------------------------------# 
# Section 4 - Drag Generation by BLI Engine #
#---------------------------------------# 
class DragbyBLIEngine:
    def __init__(self, flight_conditions, nacelle_params, FL, Mach):
        self.T, self.p, self.rho = flight_conditions.calculate_atmospheric_properties(FL)
        self.v_freestream, _ = flight_conditions.calculate_free_stream_velocity(Mach, FL)
        self.nac_length = nacelle_params.nac_length
        self.A_inlet = nacelle_params.A_inlet
        self.inlet_radius = math.sqrt(self.A_inlet / math.pi)
        
        print(f"Temperature (T): {self.T} K")
        print(f"Density (rho): {self.rho} kg/m^3")
        print(f"Free-stream velocity (v_freestream): {self.v_freestream} m/s")
        print(f"Nacelle length (nac_length): {self.nac_length} m")
        print(f"Inlet area (A_inlet): {self.A_inlet} m^2")
        print(f"Inlet radius (inlet_radius): {self.inlet_radius} m")

    def calculate_zero_lift_drag(self):
        mu = (18.27 * 10**-6) * (411.15 / (self.T + 120)) * (self.T / 291.15) ** 1.5 # Dynamic Viscosity
        k = 10 * 10**-6 # Roughness of the surface - Al surface

        Re = (self.rho * self.v_freestream * self.nac_length) / mu # Reynolds Number
        Re0 = 38 * (self.nac_length / k) ** 1.053 # Cut off Reynolds Number
        if Re > Re0:
            Re = Re0

        cf = 1.328 / math.sqrt(Re) # Skin Friction Coefficient
        f = self.nac_length / (2 * self.inlet_radius) # Fineness Ratio
        Fnac = 1 + (0.35 / f) 
        Cdzero=cf*Fnac # Zero Lift Drag Coefficient
        Snacwet = math.pi * 2 * self.inlet_radius * self.nac_length # Wetted Surface Area
        fnacparasite = cf * Fnac * Snacwet 
        Dzero = 0.5 * self.rho * self.v_freestream**2 * fnacparasite # Zero Lift Drag
 
        print(f"Cd0: {Cdzero}")
        print(f"Dynamic viscosity (mu): {mu} Pa·s")
        print(f"Reynolds number (Re): {Re}")
        print(f"Skin friction coefficient (cf): {cf}")

        print(f"rho: {self.rho}")
        print(f"freestream: {self.v_freestream}")

        print(f"Fineness ratio (f): {f}")
        print(f"Wetted surface area (Snacwet): {Snacwet} m^2")
        print(f"Parasite drag factor (fnacparasite): {fnacparasite}")
        print(f"Zero-lift drag (Dzero): {Dzero} N")
        
        
        return Dzero

#---------------------------------------# 
# Section 5 - Visualization of the Engine Model #
#---------------------------------------# 
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

        self.extra_length = 2
        self.l_intake = 1.5
        self.l_engine = 2.5
        self.l_exhaust = self.nac_length - (self.l_intake + self.l_engine)
        self.disk_location = self.l_intake + 0.5

    def calculate_geometry(self):
        self.x = np.linspace(-self.extra_length, self.nac_length + self.extra_length, 700)

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

        self.FL = None
        self.Mach = None
        self.A_inlet = None

        input_frame = tk.Frame(root, padx=10, pady=10)
        input_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(input_frame, text="Enter the Flight Height (Feet):").pack()
        self.fl_entry = tk.Entry(input_frame)
        self.fl_entry.pack()

        tk.Label(input_frame, text="Enter Mach Number:").pack()
        self.mach_entry = tk.Entry(input_frame)
        self.mach_entry.pack()

        tk.Label(input_frame, text="Enter Inlet Area (m²):").pack()
        self.area_entry = tk.Entry(input_frame)
        self.area_entry.pack()

        submit_btn = tk.Button(input_frame, text="Visualize", command=self.visualize)
        submit_btn.pack(pady=10)

        self.canvas_frame = tk.Frame(root, padx=10, pady=10)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.output_frame = tk.Frame(root, padx=10, pady=10)
        self.output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(self.output_frame, wrap=tk.WORD, width=40, height=30)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def visualize(self):
        self.output_text.delete("1.0", tk.END)
        self.FL = float(self.fl_entry.get())
        self.Mach = float(self.mach_entry.get())
        self.A_inlet = float(self.area_entry.get())

        flight_conditions = FlightConditions()
        T, p, rho = flight_conditions.calculate_atmospheric_properties(self.FL)
        v_inlet, v_freestream = flight_conditions.calculate_free_stream_velocity(self.Mach, self.FL)

        self.output_text.insert(tk.END, 'Chapter 1: Flight Conditions\n')
        self.output_text.insert(tk.END, f"Temperature: {T:.2f} K\n")
        self.output_text.insert(tk.END, f"Pressure: {p:.2f} Pa\n")
        self.output_text.insert(tk.END, f"Density: {rho:.6f} kg/m³\n")
        self.output_text.insert(tk.END, f"Free-stream velocity: {v_freestream:.2f} m/s\n")
        self.output_text.insert(tk.END, '-----------------------------\n')

        # Chapter 2: Nacelle Parameters
        nacelle = NacelleParameters(v_inlet, self.A_inlet)
        results_nacelle = nacelle.variable_parameters()
        A_disk = results_nacelle[1]
        v_disk = results_nacelle[5]
        v_exhaust = results_nacelle[6]

        self.output_text.insert(tk.END, 'Chapter 2: Nacelle Geometry\n')
        self.output_text.insert(tk.END, f"Inlet Radius: {results_nacelle[3]:.2f} m\n")
        self.output_text.insert(tk.END, f"Inlet Area: {results_nacelle[0]:.2f} m²\n")
        self.output_text.insert(tk.END, f"Inlet Velocity: {results_nacelle[7]:.2f} m/s\n")
        self.output_text.insert(tk.END, f"Disk Radius: {results_nacelle[8]:.2f} m\n")
        self.output_text.insert(tk.END, f"Disk Area: {A_disk:.2f} m²\n")
        self.output_text.insert(tk.END, f"Disk Velocity: {v_disk:.2f} m/s\n")
        self.output_text.insert(tk.END, f"Exhaust Radius: {results_nacelle[4]:.2f} m\n")
        self.output_text.insert(tk.END, f"Exhaust Area: {results_nacelle[2]:.2f} m²\n")
        self.output_text.insert(tk.END, f"Exhaust Velocity: {v_exhaust:.2f} m/s\n")
        self.output_text.insert(tk.END, '--------------------------------\n')

        actuator_model = ActuatorDiskModel(rho, A_disk, v_inlet, v_disk)


        mdot, T, P_disk, P_total, _ = actuator_model.display_results()

        self.output_text.insert(tk.END, 'Chapter 3: Basic Actuator Disk Model\n')
        self.output_text.insert(tk.END, f"Mass flow rate (mdot): {mdot:.2f} kg/s\n")
        self.output_text.insert(tk.END, f"Thrust (T): {T:.2f} N\n")
        self.output_text.insert(tk.END, f"Power required at the disk (P_disk): {P_disk:.2f} kW\n")
        self.output_text.insert(tk.END, f"Total efficiency (ηTotal): {nacelle.ηTotal:.2f}\n")
        self.output_text.insert(tk.END, f"Total electrical power required (P_total): {P_total:.2f} kW\n")
        self.output_text.insert(tk.END, '--------------------------------------\n')

        bli_engine = DragbyBLIEngine(flight_conditions, nacelle, self.FL, self.Mach)
        Dzero = bli_engine.calculate_zero_lift_drag()

        self.output_text.insert(tk.END, 'Chapter 4: Drag Generated by BLI Engine\n')
        self.output_text.insert(tk.END, f"Zero Lift Drag (Dzero): {Dzero:.2f} N\n")
        self.output_text.insert(tk.END, '---------------------------------------\n')

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
