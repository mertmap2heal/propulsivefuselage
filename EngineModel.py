import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp
import mplcursors
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


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
        self.A_inlet = A_inlet  # Engine Inlet Area (m²)
        self.v_inlet = v_inlet  # Inlet Velocity (m/s)
        self.ηdisk = 0.95  # Disk Efficiency
        self.ηmotor = 0.95  # Motor Efficiency
        self.ηprop = 0.97  # Propeller Efficiency
        self.ηTotal = self.ηmotor * self.ηprop * self.ηdisk
        self.nac_length = 2.8  # Nacelle Length (m)

    def variable_parameters(self, rho, p):
        A_disk = self.A_inlet * 0.9  # Disk Area (m²)
        Inlet_radius = math.sqrt(self.A_inlet / math.pi)  # Inlet Radius (m)
        Disk_radius = math.sqrt(A_disk / math.pi)  # Disk Radius (m)
        v_disk = (self.A_inlet * self.v_inlet) / A_disk  # Disk Velocity (m/s)
        v_exhaust = 2 * v_disk - self.v_inlet  # Exhaust Velocity (m/s)
        A_exhaust = A_disk * (v_disk / v_exhaust)  # Exhaust Area (m²)
        Exhaust_radius = math.sqrt(A_exhaust / math.pi)  # Exhaust Radius (m)
        
        delta_p =  0.5 * rho * (v_exhaust**2 - self.v_inlet**2)  
        P2 = p + delta_p
        Pressure_ratio = (P2 + 0.5 * rho * v_exhaust**2) / (p + 0.5 * rho * self.v_inlet**2)  # Pressure Ratio

        return (
            self.A_inlet, A_disk, A_exhaust, Inlet_radius, Exhaust_radius,
            v_disk, v_exhaust, self.v_inlet, Disk_radius, Pressure_ratio, delta_p, p  
        )
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
        return Dzero, Cdzero, mu
    
#---------------------------------------# 
# Section 5 - Mass Estimation of the Engine #
#---------------------------------------# 

class EngineMassEstimation:
    def __init__(self, v_inlet, A_inlet, rho, p, nac_length):
 
        # Instantiate NacelleParameters to extract necessary variables
        nacelle = NacelleParameters(v_inlet, A_inlet)
        results = nacelle.variable_parameters(rho, p)
        inlet_radius = results[3]  # Extract inlet radius from NacelleParameters

        # Shaft and rotor dimensions
        self.D_Shaft = (inlet_radius * 2) * 0.10  # Shaft diameter in meters
        self.l_Shaft = nac_length*0.4             # Shaft length in meters
        self.rho_Shaft = 1500                     # Shaft material density (kg/m^3) - Composite

        self.D_Rot = (inlet_radius * 2) * 0.85    # Rotor diameter in meters
        self.l_Rot = self.l_Shaft * 0.5           # Rotor length in meters
        self.rho_Rot = 1500                       # Rotor material density (kg/m^3) - Composite

        # Magnet properties
        self.p = 4                                # Number of pole pairs
        self.alpha_Mag = math.radians(30)         # Magnet angle in radians
        self.h_Mag = 0.01                         # Magnet height (m)
        self.rho_Mag = 4800                       # Magnet density (kg/m^3) - Ceramic

        # Stator properties
        self.D_Core_i = 0.06                      # Inner core diameter of stator (m)
        self.rho_Stator = 2800                    # Stator material density (kg/m^3)
        self.N_Slots = 36                         # Number of slots
        self.h_Slot = 0.025                       # Slot depth (m)
        self.w_Teeth = 0.005                      # Tooth width (m)
        self.delta_Slot = 0.0005                  # Slot depression depth (m)
        self.w_Slot = 0.003                       # Slot width (m)

        # Armature properties
        self.N_Phases = 3                         # Number of phases
        self.l_Conductor = 10                     # Total conductor length (m)
        self.A_Conductor = 1e-6                   # Conductor cross-sectional area (m^2)
        self.rho_Conductor = 2700                 # Conductor density (kg/m^3) - AL Conductor

        # Other properties
        self.k_Serv = 0.1                         # Service mass fraction

    def calculate_shaft_mass(self):
        return math.pi * (self.D_Shaft / 2) ** 2 * self.l_Shaft * self.rho_Shaft

    def calculate_rotor_mass(self):
        return ((self.D_Rot ** 2 - self.D_Shaft ** 2) * math.pi * self.l_Rot * self.rho_Rot) / 4

    def calculate_magnet_mass(self):
        return 0.5 * self.p * self.alpha_Mag * ((self.D_Rot / 2 + self.h_Mag) ** 2 - (self.D_Rot / 2) ** 2) * self.l_Rot * self.rho_Mag

    def calculate_stator_mass(self):
        iron_mass = (math.pi * self.l_Rot * (self.D_Rot ** 2 - self.D_Core_i ** 2) * self.rho_Stator) / 4
        teeth_mass = self.l_Rot * self.N_Slots * (
            (self.h_Slot * self.w_Teeth + math.pi * (self.D_Rot / self.N_Slots) * self.delta_Slot - self.w_Slot * self.delta_Slot)
            * self.rho_Stator
        )
        return iron_mass + teeth_mass

    def calculate_armature_mass(self):
        return self.N_Phases * self.l_Conductor * self.A_Conductor * self.rho_Conductor

    def calculate_service_mass(self, m_Shaft, m_Rot, m_Mag, m_Stator, m_Arm):
        return self.k_Serv * (m_Shaft + m_Rot + m_Mag + m_Stator + m_Arm)

    def calculate_total_motor_mass(self):
        m_Shaft = self.calculate_shaft_mass()
        m_Rot = self.calculate_rotor_mass()
        m_Mag = self.calculate_magnet_mass()
        m_Stator = self.calculate_stator_mass()
        m_Arm = self.calculate_armature_mass()
        m_Serv = self.calculate_service_mass(m_Shaft, m_Rot, m_Mag, m_Stator, m_Arm)
        return m_Shaft + m_Rot + m_Mag + m_Stator + m_Arm + m_Serv
    
#---------------------------------------#
# Section 6 - Fuselage Flow Simulation #
#---------------------------------------#
 
class Flow_around_fuselage:
    def __init__(self, v_freestream, Mach, rho, mu, delta_p, A_inlet, p,
                propulsor_position=33.0, eta_disk=0.95, 
                eta_motor=0.95, eta_prop=0.97):

        self.A_inlet = A_inlet
        A_disk = self.A_inlet * 0.9  # Disk Area (m²)
        disk_radius = math.sqrt(A_disk / math.pi)  # Disk Radius (m)
        self.disk_radius = disk_radius

        self.fuselage_length = 38
        self.fuselage_radius = 2.0    
        self.nose_length = 3.0
        self.tail_length = 10.0
        self.free_stream_velocity = v_freestream
        self.Mach = Mach
        self.rho = rho
        self.p = p
        self.mu = mu

        # Geometry creation 
        N = 1500
        self.x = np.linspace(0, self.fuselage_length, N)
        self.Re_x = self.rho * self.free_stream_velocity * self.x / self.mu
        self.y_upper = np.zeros(N)
        self.y_lower = np.zeros(N)
        
        for i, xi in enumerate(self.x):
            if xi <= self.nose_length:
                y = self.fuselage_radius * (1 - ((xi - self.nose_length)/self.nose_length)**2)
                self.y_upper[i] = y
                self.y_lower[i] = -y
            elif xi >= self.fuselage_length - self.tail_length:
                x_tail = xi - (self.fuselage_length - self.tail_length)
                y = self.fuselage_radius * (1 - (x_tail/self.tail_length)**2)
                self.y_upper[i] = y
                self.y_lower[i] = -y
            else:
                self.y_upper[i] = self.fuselage_radius
                self.y_lower[i] = -self.fuselage_radius

        self.R = np.abs(self.y_upper)

        # Get radius at propulsor position
        idx = np.argmin(np.abs(self.x - propulsor_position))
        R_prop = self.y_upper[idx]
        self.effective_A_disk = np.pi * (self.disk_radius**2 - R_prop**2)
        if self.effective_A_disk <= 0:
            raise ValueError("Disk radius must exceed fuselage radius at propulsor")

        self.A_disk = np.pi * self.disk_radius**2
        self.propulsor_position = propulsor_position

        v_local = self.get_local_velocity_at_propulsor()

        self.nacelle = NacelleParameters(
            v_inlet=v_local,  # BL-ingested velocity
            A_inlet=self.A_inlet  # Original engine inlet area
        )

        # Compute nacelle parameters
        all_params = self.nacelle.variable_parameters(rho=self.rho, p=self.p)
        self.delta_p = all_params[-2]  # Override input delta_p with computed value

        # Initialize actuator disk with PROPER velocities
        self.actuator_disk = ActuatorDiskModel(
            rho=self.rho,
            A_disk=self.effective_A_disk,
            v_inlet=v_local,        # From local velocity calculation
            v_disk=all_params[5],   # v_disk from NacelleParameters (index 5)
            eta_disk=eta_disk,
            eta_motor=eta_motor,
            eta_prop=eta_prop
        )

        # Remaining initialization 
        self.dx = self.x[1] - self.x[0]
        self.source_strength = self.source_strength_thin_body()
        
        # Boundary layer arrays 
        self.delta_99 = np.zeros_like(self.x)
        self.delta_star = np.zeros_like(self.x)
        self.theta = np.zeros_like(self.x)
        self.theta_star = np.zeros_like(self.x)
        self.tau_wall = np.zeros_like(self.x)
        self.nu_t = np.zeros_like(self.x)

        self.results_with_propulsor = None
        self.results_without_propulsor = None
        

    def source_strength_thin_body(self):
        dr2_dx = np.zeros_like(self.x)
        
        # Check Mach number first
        if self.Mach <= 0:
            raise ValueError("Mach number must be positive")
        elif self.Mach >= 1.0:
            raise ValueError("The model currently only supports subsonic flows (Mach < 1). "
                            f"Received Mach = {self.Mach:.2f}")

        # Only apply Prandtl-Glauert for 0.3 <= Mach < 1.0
        if 0.3 <= self.Mach < 1.0:
            beta = np.sqrt(1 - self.Mach**2)  #https://ntrs.nasa.gov/api/citations/19930093731/downloads/19930093731.pdf
            use_stretching = True
        else:  # Mach < 0.3 - If the mach is smaller than 0.3, the prandtl-glauert correction is not applied
            beta = 1.0 #https://ntrs.nasa.gov/api/citations/19930093731/downloads/19930093731.pdf
            use_stretching = False

        # Original geometry parameters (unchanged)
        nose_length = self.nose_length
        tail_start = self.fuselage_length - self.tail_length
        fuselage_radius = self.fuselage_radius

        for i, xi in enumerate(self.x):
            # Apply coordinate stretching only when needed
            current_xi = xi / beta if use_stretching else xi

            if current_xi <= nose_length:
                # Nose section  
                term = (current_xi - nose_length) / nose_length
                dr2_dx[i] = 2 * fuselage_radius**2 * (1 - term**2) * (-2 * term / nose_length)
                
            elif current_xi >= tail_start:
                # Tail section  
                x_tail = current_xi - tail_start
                term = x_tail / self.tail_length
                dr2_dx[i] = 2 * fuselage_radius**2 * (1 - term**2) * (-2 * term / self.tail_length)
                
            else:
                # Cylindrical section
                dr2_dx[i] = 0.0

        return self.free_stream_velocity * np.pi * dr2_dx
    
    def plot_fuselage_geometry(self, canvas_frame):
        # Create a figure for the fuselage vs source strength plot
        fig, ax1 = plt.subplots(figsize=(12, 4))
        
        # Use self.x, self.y_upper, self.y_lower, etc., instead of self.fuselage.x
        line_upper, = ax1.plot(self.x, self.y_upper, 'b', label='Fuselage')
        line_lower, = ax1.plot(self.x, self.y_lower, 'b')
        ax1.fill_between(self.x, self.y_upper, self.y_lower, color='lightblue', alpha=0.3)
        ax1.set_xlabel('Axial Position [m]')
        ax1.set_ylabel('Vertical Position [m]', fontsize=9, color='b')
        ax1.set_title('Source Strength Distribution')
        ax1.grid(True)

        ax2 = ax1.twinx()
        line_source, = ax2.plot(self.x, self.source_strength, 'r', label='Source Strength Q(x)')
        ax2.set_ylabel('Source Strength Q(x) [m²/s]', color='r', fontsize=9)

        ax1.legend(loc='lower left')
        ax2.legend(loc='lower right')
        fig.tight_layout()

        # Embed the figure into the provided canvas_frame
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def velocity_components_around_fuselage(self, X, Y, apply_mask=True):
        U = np.full(X.shape, self.free_stream_velocity, dtype=np.float64)
        V = np.zeros(Y.shape, dtype=np.float64)
        
        # Calculate velocity contributions from all sources
        for i in range(len(self.x)):
            if self.source_strength[i] == 0:
                continue
            dx = X - self.x[i]
            dy = Y
            r_sq = dx**2 + dy**2 + 1e-6  # Prevent division by zero
            U += (self.source_strength[i] * self.dx / (2 * np.pi)) * (dx / r_sq) #  Aerodynamics of aircrafts 1 lecture notes
            V += (self.source_strength[i] * self.dx / (2 * np.pi)) * (dy / r_sq)
        
        #  Under this part, masking is applied to fuselage
        if apply_mask:
            epsilon = 1e-6
            # Interpolate fuselage radius at every X,Y point
            R_interp = np.interp(X, self.x, self.R, 
                                left=0.0,   # No radius before x=0
                                right=0.0)  # No radius after x=L
            fuselage_mask = (np.abs(Y) <= R_interp + epsilon)
            # Apply mask
            U[fuselage_mask] = np.nan
            V[fuselage_mask] = np.nan
        
        return U, V

    def plot_velocity_streamlines(self, canvas_frame):
        # Clear previous widgets
        for widget in canvas_frame.winfo_children():
            widget.destroy()
            
        x = np.linspace(-10, self.fuselage_length + 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)
        U, V = self.velocity_components_around_fuselage(X, Y)
        
        # Replace NaN values with zero for plotting
        U = np.nan_to_num(U, nan=0.0)
        V = np.nan_to_num(V, nan=0.0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        strm = ax.streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), 
                             cmap='jet', linewidth=1, density=2, arrowsize=1)
        fig.colorbar(strm.lines, ax=ax, label='Velocity Magnitude (m/s)')
        ax.plot(self.x, self.y_upper, 'k', linewidth=2)
        ax.plot(self.x, self.y_lower, 'k', linewidth=2)
        ax.fill_between(self.x, self.y_upper, self.y_lower, color='lightgray', alpha=0.5)
        ax.set_xlabel('Axial Position [m]')
        ax.set_ylabel('Vertical Position [m]')
        ax.set_title('2D Velocity Field Around Fuselage')
        ax.set_aspect('equal')
        ax.grid(True)

        #  interactive cursor to read data
        cursor = mplcursors.cursor(strm.lines, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(f"x: {sel.target[0]:.2f}\ny: {sel.target[1]:.2f}"))
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
    def pressure_distribution(self):
        # Compute velocity components and magnitude
        U, V = self.velocity_components_around_fuselage(self.x, self.y_upper, apply_mask=False)
        velocity_magnitude = np.sqrt(U**2 + V**2)

        # Compute the incompressible pressure coefficient
        Cp_incompressible = 1 - (velocity_magnitude / self.free_stream_velocity)**2

        # compressibility correction if necessary
        if self.Mach > 0.3:
            Cp_compressible = Cp_incompressible / np.sqrt(1 - self.Mach**2)
        else:
            Cp_compressible = Cp_incompressible

        # apply propulsor effect if disk is active
        if self.disk_active:
            # Get local velocity at propulsor for pressure jump calculation
            v_local = self.get_local_velocity_at_propulsor()
            
            # Physically-based spread width 2x of the boundary layer thickness
            idx = np.argmin(np.abs(self.x - self.propulsor_position))
            spread_width = 2 * self.delta_99[idx] if self.delta_99[idx] > 0 else 10*self.dx
            
            # Compute pressure jump using local dynamic pressure
            dynamic_pressure = 0.5 * self.rho * v_local**2
            area_ratio = self.effective_A_disk / self.A_disk
            pressure_jump = (self.delta_p * area_ratio) / dynamic_pressure

            # Apply influence function
            influence = 0.5 * (1 + np.tanh((self.x - self.propulsor_position)/spread_width))
            Cp_compressible += pressure_jump * influence

        return Cp_incompressible, Cp_compressible

    def plot_pressure_distribution(self, canvas_frame):
        # Clear previous widgets
        for widget in canvas_frame.winfo_children():
            widget.destroy()
            
        # Get pressure data
        Cp_incompressible, Cp_compressible = self.pressure_distribution()  # Only 2 returns
        
        # Create figure and axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot pressure coefficients on primary axis
        line_incomp, = ax1.plot(self.x, Cp_incompressible, 'b', label='Incompressible Cp')
        line_comp, = ax1.plot(self.x, Cp_compressible, 'r', label='Compressible Cp')
        ax1.set_xlabel('Axial Position [m]')
        ax1.set_ylabel('Pressure Coefficient (Cp)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)
        
        # Create secondary axis for fuselage geometry
        ax2 = ax1.twinx()
        line_upper, = ax2.plot(self.x, self.y_upper, 'k-', label='Fuselage Upper')
        line_lower, = ax2.plot(self.x, self.y_lower, 'k-', label='Fuselage Lower')
        ax2.set_ylabel('Vertical Position [m]', color='k')
        ax2.tick_params(axis='y', labelcolor='k')
        
        # Combine legends
        lines = [line_incomp, line_comp, line_upper]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # Interactive cursor
        def on_hover(sel):
            x_val = sel.target[0]
            idx = np.argmin(np.abs(self.x - x_val))
            sel.annotation.set_text(
                f"x: {x_val:.2f}m\n"
                f"Incomp Cp: {Cp_incompressible[idx]:.2f}\n"
                f"Comp Cp: {Cp_compressible[idx]:.2f}\n"
                f"Fuselage Y: {self.y_upper[idx]:.2f}m"
            )
        
        cursor = mplcursors.cursor([line_incomp, line_comp, line_upper], hover=True)
        cursor.connect("add", on_hover)
        
        # Embed plot
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def compute_pressure_gradient(self):
        _, Cp_compressible = self.pressure_distribution()
        q = 0.5 * self.rho * self.free_stream_velocity**2  # Dynamic pressure
        dCp_dx = np.gradient(Cp_compressible, self.x)
        dp_dx = q * dCp_dx
        return dp_dx

    def solve_boundary_layer(self):
        dp_dx = self.compute_pressure_gradient()
        nu = self.mu / self.rho

        # Initial conditions (laminar flat plate), initial Re_x < 5e5 for laminar start
        x_initial = max(0.01, (5e5 * self.mu) / (self.rho * self.free_stream_velocity) * 0.9)
        
        Re_x_initial = self.rho * self.free_stream_velocity * x_initial / self.mu
        
        if Re_x_initial < 5e5:  # Laminar - Blasius solution
            theta_initial = 0.664 * np.sqrt(nu * x_initial / self.free_stream_velocity)
            delta_99_initial = 5.0 * x_initial / np.sqrt(Re_x_initial)
        else:  # Turbulent - Prandtls 1/7th power law
            theta_initial = 0.016 * x_initial / (Re_x_initial ** (1/7))
            delta_99_initial = 0.16 * x_initial / (Re_x_initial ** (1/7))

        # Solve ODE system
        sol = solve_ivp(
            self._boundary_layer_ode_system,
            [x_initial, self.x[-1]],
            [delta_99_initial, theta_initial],
            args=(dp_dx,),
            t_eval=self.x[self.x >= x_initial],
            method='LSODA',
            atol=1e-6,  # Absolute tolerance
            rtol=1e-6   # Relative tolerance
        )
        
        self.delta_99 = np.interp(self.x, sol.t, sol.y[0])
        self.theta = np.interp(self.x, sol.t, sol.y[1])
        self.delta_star = self._compute_displacement_thickness()
        self.tau_wall = self._compute_wall_shear_stress()

    def boundary_layer_velocity_profile(self, x_pos, y):
        """
        Compute velocity at (x, y) considering boundary layer effects.
        
        Uses a laminar profile if the local Reynolds number is below 5e5,
        otherwise applies the turbulent 1/7th power law.
        """
        # Find the index corresponding to x_pos
        idx = np.argmin(np.abs(self.x - x_pos))
        delta_99 = self.delta_99[idx]
        U_e = self._get_edge_velocity_at(idx)  
        
        # If y is outside the boundary layer, return the freestream velocity
        if y > delta_99:
            return U_e
        
        # Determine local Reynolds number at x_pos
        Re_local = self.rho * U_e * self.x[idx] / self.mu
        
        if Re_local < 5e5:
            # Laminar boundary layer: a simple parabolic profile
            # This profile satisfies: u=0 at y=0 and u=U_e at y=delta_99
            return U_e * (2 * (y / delta_99) - (y / delta_99) ** 2)
        else:
            # Turbulent boundary layer: 1/7th power law profile
            return U_e * (y / delta_99) ** (1/7)
    

    def get_local_velocity_at_propulsor(self):
        """Compute mass-averaged velocity over actuator disk area."""
        # Get propulsor parameters
        idx = np.argmin(np.abs(self.x - self.propulsor_position))
        R_inner = self.y_upper[idx]  # Fuselage radius
        R_outer = self.disk_radius # Disk radius (Covers the fuselage)
        delta = self.delta_99[idx]   # Boundary layer thickness

        
        if R_outer > R_inner + delta: # In case, the disk is larger than the boundary layer+fuselage radius, then it covers both boundary layer and freestream area
            # Split into boundary layer and freestream regions
            r_bl = np.linspace(R_inner, R_inner + delta, 50) # The distance between the fuselage and the boundary layer is divided into 50 points
            r_fs = np.linspace(R_inner + delta, R_outer, 50) # The distance between the boundary layer and and the disk is divided into 50 points
        else:
            # Entire disk within boundary layer, divide the region 100 points
            r_bl = np.linspace(R_inner, R_outer, 100)  
            r_fs = np.array([])

        # Compute velocities in boundary layer region
        velocities_bl = [self.boundary_layer_velocity_profile(self.propulsor_position, r - R_inner) 
                        for r in r_bl]

        # Compute velocities in freestream region
        if len(r_fs) > 0:  # Means there s a freestream region
            X_fs = np.full_like(r_fs, self.propulsor_position)  #Takes r_fs as a template and returns a new array of the same shape and data type 
            Y_fs = r_fs #Radial distance 
            U_fs, _ = self.velocity_components_around_fuselage(X_fs, Y_fs, apply_mask=False) #We extract a U value from another function 
        else: 
            U_fs = np.array([])  # Means that entire disk is within the boundary layer, thats why the array is empty


        # Area-weighted integration
        integral_bl = np.trapz(velocities_bl * r_bl, r_bl) if len(r_bl) > 0 else 0.0  # If there is a boundary layer, multiply the velocity with each radial portion 
        integral_fs = np.trapz(U_fs * r_fs, r_fs) if len(r_fs) > 0 else 0.0  #Integration over the free stream region, if there is no free stream region, the integral is 0 
        total_area = np.pi * (R_outer**2 - R_inner**2)

        return (2 * (integral_bl + integral_fs)) / total_area #Total mass flow rate / Total area = Mass averaged velocity

    def compute_skin_friction(self, i): 
        Re_x = self.Re_x[i]
        if Re_x < 5e5:  # Laminar (from https://fluidmech.onlineflowcalculator.com/White/Chapter7)
            Cf = 0.664 / np.sqrt(Re_x + 1e-10)  
        else:  # Turbulent (from https://fluidmech.onlineflowcalculator.com/White/Chapter7)
            Cf = 0.027 / (Re_x ** (1/7))   
        return Cf

    def _boundary_layer_ode_system(self, x, y, dp_dx):
        theta, delta_99 = y
        eps = 1e-10  # Small constant to avoid division by zero
        R_min = 0.1  # Minimum radius threshold 

        # Ensure U_e is positive
        U_e = max(self.free_stream_velocity, eps)

        # Interpolate the pressure gradient at the current x position
        current_dp_dx = np.interp(x, self.x, dp_dx)

        # Geometry parameters
        i = np.clip(np.searchsorted(self.x, x), 1, len(self.x) - 2)  # Avoid edge indices
        R = np.maximum(self.R[i], eps)  # Ensure R is always positive

        # Compute dR/dx using central difference, handle boundaries
        if 0 < i < len(self.x) - 1:
            dR_dx = (self.R[i + 1] - self.R[i - 1]) / (self.x[i + 1] - self.x[i - 1])
        else:
            dR_dx = 0.0

        # Suppress dR_dx contribution when R is below threshold to avoid instability
        if R < R_min:
            dR_dx = 0.0
        R_safe = max(R, R_min)  # Use R_min to prevent division by small R

        # Flow regime parameters
        Re_x = self.rho * U_e * x / (self.mu + eps)
        Cf = self.compute_skin_friction(i)
        H = 2.59 if Re_x < 5e5 else 1.29  # Shape factor, https://fluidmech.onlineflowcalculator.com/White/Chapter7/

        # Momentum thickness equation (axisymmetric)
        dtheta_dx = (Cf/2) + (theta/(self.rho * self.free_stream_velocity**2 + eps)) * (H + 2) * dp_dx[i] - (theta/R)* dR_dx + (self.mu/(self.rho * self.free_stream_velocity * R**2 + eps)) * theta
        # Delta99 equation based on empirical thickness ratios
        delta_theta_ratio = 5.0 / 0.664 if Re_x < 5e5 else 0.16 / 0.016
        ddelta99_dx = delta_theta_ratio * dtheta_dx - (delta_99 / R_safe) * dR_dx

        return [dtheta_dx, ddelta99_dx]
    
    def _compute_displacement_thickness(self): # https://fluidmech.onlineflowcalculator.com/White/Chapter7/
        H = np.where(self.Re_x < 5e5, 2.59, 1.29)
        return H * self.theta

    def _get_edge_velocity_at(self, i):
         
        # The free stream velocity can not be used, for the wall shear stress calculation (due to presence of the sources/sink) 
        X = np.array([self.x[i]])
        Y = np.array([self.R[i] * 1.001])  # Avoids NaN masking
        
        # Get velocity components without masking
        U, _ = self.velocity_components_around_fuselage(X, Y, apply_mask=False)
        
        return U[0]
    
    def _compute_wall_shear_stress(self):   # https://en.wikipedia.org/wiki/Skin_friction_drag
        tau_wall = np.zeros_like(self.x)
        
        for i in range(len(self.x)):
            # Get local edge velocity
            U_e = self._get_edge_velocity_at(i)
            
            # Compute local Reynolds number
            Re_x_local = self.rho * U_e * self.x[i] / self.mu
            
            # Skin friction coefficient
            if Re_x_local < 5e5:  # Laminar (Blasius)
                Cf = 0.664 / np.sqrt(Re_x_local)
            else:  # Turbulent (Prandtl-Schlichting)
                Cf = 0.027 / (Re_x_local ** (1/7))
            
            # Wall shear stress formula
            tau_wall[i] = Cf * 0.5 * self.rho * U_e**2  #  https://youtu.be/x_VhWhmJqrI  
        
        return tau_wall

    def plot_boundary_layer_thickness(self, canvas_frame):
        """Plot boundary layer thickness for both cases."""
        # Clear previous widgets
        for widget in canvas_frame.winfo_children():
            widget.destroy()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot fuselage geometry
        ax.plot(self.x, self.y_upper, 'k', linewidth=2, label='Fuselage')

        # Get results for both cases
        results_without = self.results_without_propulsor
        results_with = self.results_with_propulsor

        # Compute surface-normal δ99 for both cases
        dy_dx = np.gradient(self.y_upper, self.x)
        theta = np.arctan(dy_dx)
        delta_normal_without = results_without["delta_99"] * np.cos(theta)
        delta_normal_with = results_with["delta_99"] * np.cos(theta)

        # Plot filled regions for δ99
        ax.fill_between(
            self.x, 
            self.y_upper + delta_normal_without, 
            self.y_upper, 
            color='blue', alpha=0.3, 
            label='δ99 (Without Propulsor)'
        )
        ax.fill_between(
            self.x, 
            self.y_upper + delta_normal_with, 
            self.y_upper, 
            color='red', alpha=0.3, 
            label='δ99 (With Propulsor)'
        )

        # Add δ99 curves (absolute values)
        ax2 = ax.twinx()
        line_without, = ax2.plot(
            results_without["x"], 
            results_without["delta_99"], 
            'b--', 
            label='δ99 Without Propulsor'
        )
        line_with, = ax2.plot(
            results_with["x"], 
            results_with["delta_99"], 
            'r--', 
            label='δ99 With Propulsor'
        )
        ax2.set_ylabel('Absolute Boundary Layer Thickness [m]', color='k')

        # Add transition marker
        transition_idx = np.where(self.Re_x >= 5e5)[0][0]
        ax.axvline(
            self.x[transition_idx], 
            color='gray', 
            linestyle='--', 
            label=f'Transition (x={self.x[transition_idx]:.1f}m)'
        )

        # Labels and legends
        ax.set_xlabel('Axial Position [m]')
        ax.set_ylabel('Vertical Position [m]')
        ax.set_title('Boundary Layer Development Comparison')
        ax.grid(True)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Add interactive hover for δ99 curves
        def on_hover(sel):
            x_val = sel.target[0]
            # For "Without Propulsor" curve
            idx_without = np.argmin(np.abs(results_without["x"] - x_val))
            # For "With Propulsor" curve
            idx_with = np.argmin(np.abs(results_with["x"] - x_val))
            
            sel.annotation.set_text(
                f"x: {x_val:.2f} m\n"
                f"δ99 (Without): {results_without['delta_99'][idx_without]:.3f} m\n"
                f"δ99 (With): {results_with['delta_99'][idx_with]:.3f} m"
            )

        # Attach cursor to both lines
        cursor = mplcursors.cursor([line_without, line_with], hover=True)
        cursor.connect("add", on_hover)

        # Add metrics
        T_net = self.compute_net_thrust()
        D_red = self.compute_drag_reduction()
        PSC = self.compute_PSC()
        metrics_text = (
            f"Net Thrust: {T_net:.2f} N\n"
            f"Drag Reduction: {D_red:.2f} N\n"
            f"PSC: {PSC:.1%}"
        )
        ax.text(
            0.02, 0.95,
            metrics_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='wheat', alpha=0.5, edgecolor='black', boxstyle='round')
        )

        # Embed the figure and add toolbar
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas_widget = canvas.get_tk_widget()
        
        # Add navigation toolbar (zoom/pan)
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
    def run_simulation(self):
        """Run baseline and BLI cases with proper pressure field updates and result storage."""
        # =================================================================
        # Case 1: Baseline (no BLI engine)
        # =================================================================
        self.disk_active = False
        self.solve_boundary_layer()  # Solve without propulsor
        
        # Store baseline results with edge velocities
        self.results_without_propulsor = {
            "x": self.x.copy(),
            "delta_99": self.delta_99.copy(),
            "delta_star": self.delta_star.copy(),
            "theta": self.theta.copy(),
            "Cp": self.pressure_distribution()[1].copy(),
            "U_e": [self._get_edge_velocity_at(i) for i in range(len(self.x))]  # Critical for drag calc
        }

        # =================================================================
        # Case 2: BLI active
        # =================================================================
        self.disk_active = True
        
        # Update propulsor parameters using LOCAL velocity at disk location
        v_local = self.get_local_velocity_at_propulsor()
        self.nacelle.v_inlet = v_local  # Key BLI parameter
        
        # Recompute nacelle parameters with ingested flow
        all_params = self.nacelle.variable_parameters(rho=self.rho, p=self.p)
        
        # Initialize actuator disk with updated velocities
        self.actuator_disk = ActuatorDiskModel(
            rho=self.rho,
            A_disk=self.effective_A_disk,
            v_inlet=v_local,
            v_disk=all_params[5],  # v_disk from continuity
            eta_disk=self.eta_disk,
            eta_motor=self.eta_motor,
            eta_prop=self.eta_prop
        )
        
        # Force update of pressure field with BLI effects
        self.update_pressure_field()  # Critical step!
        
        # Re-solve boundary layer with new pressure distribution
        self.solve_boundary_layer()
        
        # Store BLI results with updated edge velocities
        self.results_with_propulsor = {
            "x": self.x.copy(),
            "delta_99": self.delta_99.copy(),
            "delta_star": self.delta_star.copy(),
            "theta": self.theta.copy(),
            "Cp": self.pressure_distribution()[1].copy(),
            "U_e": [self._get_edge_velocity_at(i) for i in range(len(self.x))]  # Updated U_e
        }
        self.T_net = self.actuator_disk.calculate_thrust()
        self.D_red = self.compute_drag_reduction()
        self.PSC = self.compute_PSC()

    def update_pressure_field(self):
        """Force recompute pressure distribution with BLI effects."""
        # Reset cached Cp values to trigger full recalculation
        if hasattr(self, "_Cp_incompressible"):
            del self._Cp_incompressible
        if hasattr(self, "_Cp_compressible"):
            del self._Cp_compressible
    
    # Explicitly recompute pressure distribution
    _ = self.pressure_distribution()  # Discard return values, just force update
        
    def activate_propulsor(self):
        """Activate the propulsor and update the boundary layer solution."""
        self.disk_active = True
        self.solve_boundary_layer()

    def deactivate_propulsor(self):
        """Deactivate the propulsor and update the boundary layer solution."""
        self.disk_active = False
        self.solve_boundary_layer()
 
    def compute_net_thrust(self):
        """Calculate net thrust using the ActuatorDiskModel."""
        if not self.disk_active:
            return 0.0
        return self.actuator_disk.calculate_thrust()

    def compute_drag_reduction(self):
        """Calculate drag reduction and return both drag values for reuse."""
        if not self.disk_active or self.results_without_propulsor is None:
            return 0.0, 0.0, 0.0  # Return tuple for base/bli/Δ drag
        
        # Get stored values
        theta_base = self.results_without_propulsor["theta"]
        theta_bli = self.results_with_propulsor["theta"]
        U_e_base = self.results_without_propulsor["U_e"]
        U_e_bli = self.results_with_propulsor["U_e"]
        
        # Compute integrands once
        integrand_base = 2 * np.pi * self.R * theta_base * np.array(U_e_base)**2
        integrand_bli = 2 * np.pi * self.R * theta_bli * np.array(U_e_bli)**2
        
        # Calculate all drag values
        drag_base = self.rho * np.trapz(integrand_base, self.x)
        drag_bli = self.rho * np.trapz(integrand_bli, self.x)
        drag_reduction = drag_base - drag_bli
        
        return drag_base, drag_bli, drag_reduction

    def compute_PSC(self):
        """Calculate Power Saving Coefficient using precomputed drag values."""
        if not self.disk_active or self.results_without_propulsor is None:
            return 0.0
        
        # Get drag values from single computation
        drag_base, drag_bli, _ = self.compute_drag_reduction()
        
        # Power calculations
        V = self.free_stream_velocity
        P_ref = drag_base * V  # Baseline power (W)
        P_drag_bli = drag_bli * V  # Power with BLI drag
        
        # BLI engine power (convert from kW to W)
        P_bli_engine = self.actuator_disk.calculate_total_power() * 1e3  
        P_required = P_drag_bli + P_bli_engine

        # Compute PSC
        if P_ref == 0:
            return 0.0
        return (P_ref - P_required) / P_ref


class ActuatorDiskModel:
    def __init__(self, rho, A_disk, v_inlet, v_disk, eta_disk=0.95, eta_motor=0.95, eta_prop=0.97):
        self.rho = rho  # Air Density (kg/m^3)
        self.A_disk = A_disk  # Disk Area (m^2)
        self.v_inlet = v_inlet  # Inlet Velocity (m/s)
        self.v_disk = v_disk  # Disk Velocity (m/s)
        self.eta_disk = eta_disk  # Disk Efficiency
        self.eta_motor = eta_motor  # Motor Efficiency
        self.eta_prop = eta_prop  # Propeller Efficiency
        self.eta_total = self.eta_motor * self.eta_prop * self.eta_disk
        self.v_exhaust = 2 * self.v_disk - self.v_inlet  # Exhaust Velocity (m/s)

    def calculate_mass_flow_rate(self):
        """Calculate mass flow rate (mdot)."""
        mdot = self.rho * self.A_disk * self.v_disk
        return mdot

    def calculate_thrust(self, mdot=None):
        """Calculate thrust (T)."""
        if mdot is None:
            mdot = self.calculate_mass_flow_rate()
        T = mdot * (self.v_exhaust - self.v_inlet)
        return T

    def calculate_power_disk(self, thrust=None):
        """Calculate power required at the disk (P_disk)."""
        if thrust is None:
            thrust = self.calculate_thrust()
        P_disk = thrust * self.v_disk * 1e-3  # Convert to kW
        return P_disk

    def calculate_total_power(self, P_disk=None):
        """Calculate total electrical power required (P_total) after efficiency losses."""
        if P_disk is None:
            P_disk = self.calculate_power_disk()
        P_total = P_disk / self.eta_total
        return P_total

    def display_results(self):
        """Display results for mass flow rate, thrust, disk power, and total power."""
        mdot = self.calculate_mass_flow_rate()
        T = self.calculate_thrust(mdot)
        P_disk = self.calculate_power_disk(T)
        P_total = self.calculate_total_power(P_disk)
        return mdot, T, P_disk, P_total, self.A_disk


#---------------------------------------# 
# Section 7 - Visualization of the Engine Model #
#---------------------------------------# 
 
class EngineVisualization:
    def __init__(self, A_inlet, A_disk, A_exhaust, v_inlet, v_disk, v_exhaust, nac_length, inlet_radius, disk_radius, exhaust_radius, app=None):
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
        self.app = app  # Reference to the main app

        # Additional geometric parameters
        self.extra_length = 2
        self.l_intake = 1.5
        self.l_engine = 2.5
        self.l_exhaust = self.nac_length - (self.l_intake + self.l_engine)
        self.disk_location = self.l_intake + 0.5

    def calculate_geometry(self):
        # Create an array of x positions along the extended nacelle length.
        self.x = np.linspace(-self.extra_length, self.nac_length + self.extra_length, 700)

        # Define the outer radius along the nacelle using piecewise functions.
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

        # Define the velocities along the nacelle using piecewise functions.
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

    def plot_velocity_field(self, canvas_frame):
        self.calculate_geometry()
        x_grid = np.linspace(-5, self.nac_length + 5, 100)
        y_grid = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # Populate U and V (existing code)
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                x_p = X[j, i]
                y_p = Y[j, i]
                
                # Determine the local nacelle radius.
                if x_p < 0 or x_p > self.nac_length:
                    nacelle_radius_at_x = 0
                else:
                    idx = np.argmin(np.abs(self.x - x_p))
                    nacelle_radius_at_x = self.outer_radius[idx]
                
                # Mask the region inside the nacelle.
                if np.abs(y_p) <= nacelle_radius_at_x:
                    U[j, i] = np.nan
                    V[j, i] = np.nan
                    continue
                
                # Assign velocities based on x-position.
                if x_p < self.l_intake:
                    U[j, i] = self.v_inlet
                elif x_p < self.l_intake + self.l_engine:
                    U[j, i] = self.v_disk
                else:
                    U[j, i] = self.v_exhaust
                V[j, i] = 0
 
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.U_velocity = U
        self.V_velocity = V
        
        fig, ax = plt.subplots(figsize=(10, 6))
        strm = ax.streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), cmap='jet', density=1.5)
        ax.plot(self.x, self.outer_radius, 'k', linewidth=2)
        ax.plot(self.x, -self.outer_radius, 'k', linewidth=2)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Velocity Field Around the Nacelle")
        cbar = fig.colorbar(strm.lines, ax=ax)
        cbar.set_label("Velocity Magnitude (m/s)")
        ax.grid(True)

        # Clear previous contents of the canvas frame.
        for widget in canvas_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def plot_geometry(self, canvas_frame):
        # Calculate geometry data.
        self.calculate_geometry()

        # Dynamically set the figure dimensions.
        fig_width = max(16, self.nac_length / 2)
        fig_height = fig_width / 6

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        cmap = plt.cm.plasma
        norm = Normalize(vmin=min(self.velocities), vmax=max(self.velocities))
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Plot the nacelle geometry.
        ax.fill_between(self.x, -self.outer_radius, self.outer_radius, color="lightgray", alpha=0.5, label="Nacelle Geometry")
        ax.plot(self.x, self.outer_radius, color="darkred", linewidth=2)
        ax.plot(self.x, -self.outer_radius, color="darkred", linewidth=2)

        # Color-code the geometry based on velocity.
        for i in range(len(self.x) - 1):
            ax.fill_between(
                [self.x[i], self.x[i + 1]],
                [-self.outer_radius[i], -self.outer_radius[i + 1]],
                [self.outer_radius[i], self.outer_radius[i + 1]],
                color=cmap(norm(self.velocities[i])),
                edgecolor="none",
                alpha=0.7
            )

        # Add fan (disk) and exhaust boundaries.
        fan_x = self.disk_location
        ax.plot([fan_x, fan_x], [-self.disk_radius, self.disk_radius], color="black", linewidth=2, linestyle="--", label="Fan (Disk Location)")
        exhaust_start = self.l_intake + self.l_engine
        ax.plot([exhaust_start, exhaust_start], [-self.exhaust_radius, self.exhaust_radius], color="orange", linewidth=2, linestyle="--", label="Exhaust Boundary")
        ax.fill_between([exhaust_start, self.nac_length], [-self.exhaust_radius, -self.exhaust_radius], [self.exhaust_radius, self.exhaust_radius], color="orange", alpha=0.3, label="Exhaust Air")

        # Add text labels.
        ax.text(-self.extra_length / 2, self.inlet_radius, f"Inlet\nArea: {self.A_inlet:.2f} m²", color="black", fontsize=10, ha="center")
        ax.text(fan_x, self.disk_radius, f"Fan (Disk)\nArea: {self.A_disk:.2f} m²", color="black", fontsize=10, ha="center")
        ax.text((exhaust_start + self.nac_length) / 2, self.exhaust_radius, f"Exhaust\nArea: {self.A_exhaust:.2f} m²", color="black", fontsize=10, ha="center")

        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Velocity (m/s)', fontsize=12)

        ax.set_title("2D Engine Geometry with Velocity Representation", fontsize=16)
        ax.set_xlabel("Length (m)", fontsize=12)
        ax.set_ylabel("Radius (m)", fontsize=12)
        ax.legend()
        ax.grid()

        
        for widget in canvas_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.config(width=int(fig_width * 100), height=int(fig_height * 100))
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()
 
 
class BoundaryLayerIngestion:
    def __init__(self, root):
        self.root = root
        self._configure_root()
        self._initialize_variables()
        self._create_main_frame()
        self._create_io_section()
        self._create_notebook()

    # === GUI Setup Methods ===
    def _configure_root(self):
        self.root.title("Boundary Layer Ingestion Concept Data Screen")
        self.root.state('zoomed')

    def _initialize_variables(self):
        self.FL = None
        self.Mach = None
        self.A_inlet = None
        self.selected_data = {'x': None, 'y': None}

    def _create_main_frame(self):
        self.main_frame = tk.Frame(self.root, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def _create_io_section(self):
        # Left-side input/output section
        self.io_frame = tk.Frame(self.main_frame, padx=10, pady=10)
        self.io_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self._create_input_frame()
        self._create_output_frame()
        self._create_cursor_label()

    def _create_input_frame(self):
        input_frame = tk.LabelFrame(self.io_frame, text="Inputs", padx=10, pady=10)
        input_frame.pack(fill=tk.X, pady=5)

        tk.Label(input_frame, text="Enter the Flight Height (Feet):").pack(anchor='w')
        self.fl_entry = tk.Entry(input_frame)
        self.fl_entry.pack(fill=tk.X, pady=2)

        tk.Label(input_frame, text="Enter Mach Number:").pack(anchor='w')
        self.mach_entry = tk.Entry(input_frame)
        self.mach_entry.pack(fill=tk.X, pady=2)

        tk.Label(input_frame, text="Enter Inlet Area (m²):").pack(anchor='w')
        self.area_entry = tk.Entry(input_frame)
        self.area_entry.pack(fill=tk.X, pady=2)

        submit_btn = tk.Button(input_frame, text="Visualize", command=self.visualize)
        submit_btn.pack(pady=10)

    def _create_output_frame(self):
        output_frame = tk.LabelFrame(self.io_frame, text="Outputs", padx=10, pady=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, width=40, height=30)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def _create_cursor_label(self):
        self.cursor_label = tk.Label(self.io_frame, text="Cursor Data: x=None, y=None", font=("Arial", 10))
        self.cursor_label.pack(pady=5)

    def _create_notebook(self):
        # Create notebook tabs for different plots
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.engine_tab = tk.Frame(self.notebook)
        self.velocity_tab = tk.Frame(self.notebook)
        self.fuselage_tab = tk.Frame(self.notebook)
        self.pressure_tab = tk.Frame(self.notebook)
        self.boundary_layer_tab = tk.Frame(self.notebook)

        self.notebook.add(self.engine_tab, text="Engine Geometry")
        self.notebook.add(self.velocity_tab, text="Velocity Field")
        self.notebook.add(self.fuselage_tab, text="Source Strength Geometry")
        self.notebook.add(self.pressure_tab, text="Pressure Distribution")
        self.notebook.add(self.boundary_layer_tab, text="Boundary Layer")

    # === Utility Methods ===
    def update_cursor_data(self, x, y):
        """Update the cursor label with the current x and y values."""
        self.selected_data['x'] = x
        self.selected_data['y'] = y
        self.cursor_label.config(text=f"Cursor Data: x={x:.2f}, y={y:.2f}")

    def _embed_figure_in_tab(self, fig, tab, cursor_artists=None):
        """Clear the given tab and embed the matplotlib figure into it.
        
        If a list of artists is provided (for example, lines), then set up mplcursors
        to display interactive data.
        """
        # Clear the tab first
        for widget in tab.winfo_children():
            widget.destroy()

        # Create and pack the canvas
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Add the navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

        # If interactive cursors are needed
        if cursor_artists:
            cursor = mplcursors.cursor(cursor_artists, hover=True)
            @cursor.connect("add")
            def on_hover(sel):
                x, y = sel.target
                self.update_cursor_data(x, y)
                sel.annotation.set_text(f"x: {x:.2f}\ny: {y:.2f}")

        canvas.draw()

    # === Main Functionality ===
    def visualize(self):
        # Clear the previous output
        self.output_text.delete("1.0", tk.END)
        self._read_inputs()

        # Process the different chapters of the analysis
        flight_conditions, T, p, rho, v_inlet, v_freestream = self._process_flight_conditions()
        self.nacelle, results_nacelle, A_disk, v_disk, v_exhaust = self._process_nacelle_geometry(rho, p, v_inlet)
        mdot, T_thrust, P_disk, P_total, _ = self._process_actuator_disk_model(rho, A_disk, v_inlet, v_disk)
        Dzero, Cdzero, mu = self._process_drag_bli_engine(flight_conditions, self.nacelle)
        total_motor_mass = self._process_engine_mass_estimation(v_inlet, rho, p, self.nacelle)

        # Plot all visualizations in their respective tabs
        self._plot_visualizations(results_nacelle, v_inlet, v_disk, v_exhaust, v_freestream, rho, mu)

    def _read_inputs(self):
        """Read and convert input values from the entry widgets."""
        self.FL = float(self.fl_entry.get())
        self.Mach = float(self.mach_entry.get())
        self.A_inlet = float(self.area_entry.get())

    # === Processing Methods (Chapters) ===
    def _process_flight_conditions(self):
        flight_conditions = FlightConditions()
        T, p, rho = flight_conditions.calculate_atmospheric_properties(self.FL)
        v_inlet, v_freestream = flight_conditions.calculate_free_stream_velocity(self.Mach, self.FL)

        self.output_text.insert(tk.END, 'Chapter 1: Flight Conditions\n')
        self.output_text.insert(tk.END, f"Temperature: {T:.2f} K\n")
        self.output_text.insert(tk.END, f"Pressure: {p:.2f} Pa\n")
        self.output_text.insert(tk.END, f"Density: {rho:.6f} kg/m³\n")
        self.output_text.insert(tk.END, f"Free-stream velocity: {v_freestream:.2f} m/s\n")
        self.output_text.insert(tk.END, '-----------------------------\n')

        return flight_conditions, T, p, rho, v_inlet, v_freestream

    def _process_nacelle_geometry(self, rho, p, v_inlet):
        nacelle = NacelleParameters(v_inlet, self.A_inlet)
        results_nacelle = nacelle.variable_parameters(rho, p)
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
        self.output_text.insert(tk.END, f"Pressure Ratio: {results_nacelle[9]:.2f}\n")
        self.output_text.insert(tk.END, '--------------------------------\n')

        return nacelle, results_nacelle, A_disk, v_disk, v_exhaust

    def _process_actuator_disk_model(self, rho, A_disk, v_inlet, v_disk):
        actuator_model = ActuatorDiskModel(rho, A_disk, v_inlet, v_disk)
        mdot, T_thrust, P_disk, P_total, _ = actuator_model.display_results()

        self.output_text.insert(tk.END, 'Chapter 3: Basic Actuator Disk Model\n')
        self.output_text.insert(tk.END, f"Mass flow rate (mdot): {mdot:.2f} kg/s\n")
        self.output_text.insert(tk.END, f"Thrust (T): {T_thrust:.2f} N\n")
        self.output_text.insert(tk.END, f"Power required at the disk (P_disk): {P_disk:.2f} kW\n")
        self.output_text.insert(tk.END, f"Total efficiency (ηTotal): {self.nacelle.ηTotal:.2f}\n")
        self.output_text.insert(tk.END, f"Total electrical power required (P_total): {P_total:.2f} kW\n")
        self.output_text.insert(tk.END, '--------------------------------------\n')

        return mdot, T_thrust, P_disk, P_total, None

    def _process_drag_bli_engine(self, flight_conditions, nacelle):
        bli_engine = DragbyBLIEngine(flight_conditions, nacelle, self.FL, self.Mach)
        Dzero, Cdzero, mu = bli_engine.calculate_zero_lift_drag()

        self.output_text.insert(tk.END, 'Chapter 4: Drag Generated by BLI Engine\n')
        self.output_text.insert(tk.END, f"Zero Lift Drag (Dzero): {Dzero:.2f} N\n")
        self.output_text.insert(tk.END, '---------------------------------------\n')

        return Dzero, Cdzero, mu

    def _process_engine_mass_estimation(self, v_inlet, rho, p, nacelle):
        engine_mass = EngineMassEstimation(v_inlet, self.A_inlet, rho, p, nacelle.nac_length)
        total_motor_mass = engine_mass.calculate_total_motor_mass()

        self.output_text.insert(tk.END, 'Chapter 5: Engine Mass Estimation\n')
        self.output_text.insert(tk.END, f"Total Motor Mass: {total_motor_mass:.2f} kg\n")
        self.output_text.insert(tk.END, '--------------------------------------\n')

        return total_motor_mass

    # === Plotting Methods ===
        self.fuselage = Flow_around_fuselage(v_freestream, self.Mach, rho, mu, delta_p, A_inlet)
    def _plot_visualizations(self, results_nacelle, v_inlet, v_disk, v_exhaust, v_freestream, rho, mu):
        # Plot Engine Geometry
        self._plot_engine_geometry(results_nacelle, v_inlet, v_disk, v_exhaust)
        A_inlet = results_nacelle[0]    
        delta_p = results_nacelle[10]
        p = results_nacelle[-1]
        # Create a fuselage object (for velocity, pressure, etc.)
        self.fuselage = Flow_around_fuselage(v_freestream, self.Mach, rho, mu, delta_p, A_inlet,p)
        self.fuselage.solve_boundary_layer()
        self.fuselage.run_simulation()  # Ensure simulation results are computed
        # Plot Fuselage Geometry (Source Strength)
        self._plot_source_field()

        # Plot Velocity Field
        self._plot_velocity_field()
        # Plot Pressure Distribution
        self._plot_pressure_distribution()
        # Plot Boundary Layer Thickness
        self._plot_boundary_layer_thickness()

    def _plot_engine_geometry(self, results_nacelle, v_inlet, v_disk, v_exhaust):
        visualization = EngineVisualization(
            A_inlet=self.nacelle.A_inlet,
            A_disk=results_nacelle[1],
            A_exhaust=results_nacelle[2],
            v_inlet=v_inlet,
            v_disk=v_disk,
            v_exhaust=v_exhaust,
            nac_length=self.nacelle.nac_length,
            inlet_radius=results_nacelle[3],
            disk_radius=results_nacelle[8],
            exhaust_radius=results_nacelle[4]
        )
        # If the EngineVisualization class can return a figure (e.g. via get_geometry_figure),
        # then we embed it using our helper. Otherwise, fall back to its original method.
        if hasattr(visualization, 'get_geometry_figure'):
            fig = visualization.get_geometry_figure()
            self._embed_figure_in_tab(fig, self.engine_tab)
        else:
            visualization.plot_geometry(self.engine_tab)
            
    def _plot_source_field(self):
        self.fuselage.plot_fuselage_geometry(self.fuselage_tab)


    
    def _plot_velocity_field(self):
        # If your fuselage object has a method to return a figure for velocity, use it
        if hasattr(self.fuselage, 'get_velocity_figure'):
            fig = self.fuselage.get_velocity_figure()
            self._embed_figure_in_tab(fig, self.velocity_tab)
        else:
            # Otherwise, use the existing method to plot directly into the tab
            self.fuselage.plot_velocity_streamlines(self.velocity_tab)

    def _plot_pressure_distribution(self):
        if hasattr(self.fuselage, 'get_pressure_figure'):
            fig = self.fuselage.get_pressure_figure()
            self._embed_figure_in_tab(fig, self.pressure_tab)
        else:
            self.fuselage.plot_pressure_distribution(self.pressure_tab)

    def _plot_boundary_layer_thickness(self):
        if hasattr(self.fuselage, 'get_boundary_layer_figure'):
            fig = self.fuselage.get_boundary_layer_figure()
            self._embed_figure_in_tab(fig, self.boundary_layer_tab)
        else:
            self.fuselage.plot_boundary_layer_thickness(self.boundary_layer_tab)

# === Main Program ===
if __name__ == "__main__":
    root = tk.Tk()
    app = BoundaryLayerIngestion(root)
    root.mainloop()