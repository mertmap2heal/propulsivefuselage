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
import matplotlib.colors as mcolors  


#---------------------------------------# 
# Section 1 - Flight Conditions #
#---------------------------------------# 
class FlightConditions:
    """
    Calculates atmospheric properties at different flight levels using the International Standard Atmosphere (ISA) model
    Theory: Divided into three atmospheric layers (Troposphere, Lower Stratosphere, Upper Stratosphere) with different
            temperature gradients and thermodynamic relationships. Uses hydrostatic equations and ideal gas law.
    """

    def calculate_atmospheric_properties(self, FL):
        # 1.1 Constants initialization
        g_s = 9.80665  # Gravitational acceleration at sea level (m/s²)
        R = 287.05  # Specific gas constant for dry air (J/(kg·K))
        
        # 1.2 Reference values at Mean Sea Level (MSL)
        T_MSL = 288.15  # Temperature at MSL (K)
        p_MSL = 101325  # Pressure at MSL (Pa)
        rho_MSL = 1.225  # Density at MSL (kg/m³)

        # 1.3 Layer-specific thermodynamic constants
        n_trop = 1.235  # Polytropic index for Troposphere
        n_uStr = 0.001  # Polytropic index for Upper Stratosphere
        
        # 1.4 Atmospheric layer boundary heights (m)
        H_G11 = 11000  # Tropopause height
        H_G20 = 20000  # Upper stratosphere base height

        # 1.5 Standard values at layer boundaries
        T_11 = 216.65  # Temperature at tropopause (K)
        p_11 = 22632  # Pressure at tropopause (Pa)
        rho_11 = 0.364  # Density at tropopause (kg/m³)

        T_20 = 216.65  # Temperature at stratosphere base (K)
        p_20 = 5474.88  # Pressure at stratosphere base (Pa)
        rho_20 = 0.088  # Density at stratosphere base (kg/m³)

        # 1.6 Temperature gradients (K/m)
        gamma_Tropo = -0.0065  # Tropospheric lapse rate
        gamma_UpperStr = 0.001  # Upper stratosphere lapse rate

        H_G = FL * 0.3048  # Convert flight level (feet) to geopotential height (meters)

        # Atmospheric layer calculations
        if H_G <= H_G11:  # Troposphere (0-11 km)
            # Theory: Temperature decreases linearly with height, use polytropic process equations
            T = T_MSL * (1 + (gamma_Tropo / T_MSL) * H_G)
            p = p_MSL * (1 + (gamma_Tropo / T_MSL) * H_G) ** (n_trop / (n_trop - 1))
            rho = rho_MSL * (1 + (gamma_Tropo / T_MSL) * H_G) ** (1 / (n_trop - 1))

        elif H_G <= H_G20:  # Lower Stratosphere (11-20 km)
            # Theory: Isothermal layer, pressure/density follow exponential decay (hydrostatic equilibrium)
            T = T_11
            p = p_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))
            rho = rho_11 * math.exp(-g_s / (R * T_11) * (H_G - H_G11))

        else:  # Upper Stratosphere (>20 km)
            # Theory: Temperature increases slightly, modified polytropic relationships
            T = T_20 * (1 + (gamma_UpperStr / T_20) * (H_G - H_G20))
            p = p_20 * (1 + (gamma_UpperStr / T_20) * (H_G - H_G20)) ** (n_uStr / (n_uStr - 1))
            rho = rho_20 * (1 - ((n_uStr - 1) / n_uStr) * (g_s / (R * T_20)) * (H_G - H_G20)) ** (1 / (n_uStr - 1))

        # 1.7 Return atmospheric properties tuple (Temperature, Pressure, Density)
        return T, p, rho
    
    def calculate_free_stream_velocity(self, Mach, FL):
        # 1.8 Gas property constants
        kappa = 1.4  # Adiabatic index for air (ratio of specific heats)
        R = 287.05  # Specific gas constant for dry air (J/(kg·K))

        # 1.9 Get atmospheric temperature
        # Theory: Uses previous atmospheric model calculation, discards pressure/density
        T, _, _ = self.calculate_atmospheric_properties(FL)

        # 1.10 Calculate speed of sound
        # Theory: a = √(γ*R*T) from ideal gas law for adiabatic processes
        a = math.sqrt(kappa * R * T)

        # 1.11 Compute velocities
        # Theory: v = M*a (Mach number definition), v_inlet assumes no upstream losses
        v_freestream = Mach * a
        v_inlet = v_freestream  # Simplified assumption (inlet = freestream velocity)

        return v_inlet, v_freestream
#---------------------------------------# 
# Section 2 - Nacelle Parameters #
#---------------------------------------# 
# Global efficiency defaults for propulsion system components
_EFFICIENCY_DEFAULTS = {
    "eta_disk": 0.95,   # Default actuator disk efficiency
    "eta_motor": 0.95,  # Default electric motor efficiency
    "eta_prop": 0.97    # Default propeller efficiency
}

class NacelleParameters:
    """
    Models nacelle aerodynamics and propulsion system performance
    Theory: Uses actuator disk theory for momentum analysis, incorporates component efficiencies
            to calculate total system performance. Manages mass flow continuity and pressure changes.
    """
    
    def __init__(self, v_inlet, A_inlet, nacelle_length=None, 
                 eta_disk=None, eta_motor=None, eta_prop=None):
        # 2.1 Initialize core geometric parameters
        self.A_inlet = A_inlet  # Inlet capture area (m²)
        self.v_inlet = v_inlet  # Inlet velocity (m/s)
        
        # 2.2 Set nacelle geometry with defaults
        self.nac_length = nacelle_length if nacelle_length is not None else 5  # Default length (m)
        
        # 2.3 Configure efficiency parameters using defaults if not provided
        self.ηdisk = eta_disk or _EFFICIENCY_DEFAULTS["eta_disk"]
        self.ηmotor = eta_motor or _EFFICIENCY_DEFAULTS["eta_motor"]
        self.ηprop = eta_prop or _EFFICIENCY_DEFAULTS["eta_prop"]
        
        # 2.4 Calculate total system efficiency
        # Theory: Overall efficiency = product of component efficiencies
        self.ηTotal = self.ηmotor * self.ηprop * self.ηdisk
        
        # 2.5 Initialize flow properties
        self.v_disk = None  # Actuator disk velocity (m/s)
        self.effective_A_disk = None  # Effective disk area (m²)

    def variable_parameters(self, rho, p):
        # 2.6 Calculate effective actuator disk area
        # Theory: Accounts for flow contraction (90% of inlet area as empirical factor)
        self.effective_A_disk = self.A_inlet * 0.9
        
        # 2.7 Calculate geometric parameters
        Inlet_radius = math.sqrt(self.A_inlet / math.pi)  # Inlet radius from area
        Disk_radius = math.sqrt(self.effective_A_disk / math.pi)  # Disk radius
        
        # 2.8 Calculate disk velocity using continuity equation
        # Theory: A1*v1 = A2*v2 for incompressible flow
        self.v_disk = (self.A_inlet * self.v_inlet) / self.effective_A_disk
        
        # 2.9 Calculate exhaust flow parameters
        # Theory: Momentum theory for ideal propulsive flow
        v_exhaust = 2 * self.v_disk - self.v_inlet  # Jet velocity after disk
        A_exhaust = self.effective_A_disk * (self.v_disk / v_exhaust)  # Exhaust area
        Exhaust_radius = math.sqrt(A_exhaust / math.pi)  # Exhaust radius
        
        # 2.10 Calculate pressure changes
        # Theory: Bernoulli's equation with momentum addition
        delta_p = 0.5 * rho * (v_exhaust**2 - self.v_inlet**2)  # Pressure difference
        P2 = p + delta_p  # Static pressure at disk
        Pressure_ratio = (P2 + 0.5 * rho * v_exhaust**2) / (p + 0.5 * rho * self.v_inlet**2)

        # 2.11 Return comprehensive parameter tuple
        return (
            self.A_inlet, self.effective_A_disk, A_exhaust,
            Inlet_radius, Exhaust_radius, self.v_disk,
            v_exhaust, self.v_inlet, Disk_radius,
            Pressure_ratio, delta_p, p
        )
#---------------------------------------# 
# Section 3 - Drag Generation by BLI Engine #
#---------------------------------------# 
 
class DragbyBLIEngine:
    """
    Calculates BLI-induced drag components using boundary layer theory
    Theory: Computes viscous drag contributions from nacelle surfaces considering:
            - Skin friction using Reynolds number correlations
            - Surface roughness effects
            - Form factor adjustments for slender bodies
    """
    
    def __init__(self, flight_conditions, nacelle_params, FL, Mach):
        # 4.1 Get atmospheric properties from flight conditions
        self.T, self.p, self.rho = flight_conditions.calculate_atmospheric_properties(FL)
        
        # 4.2 Calculate free-stream flow parameters
        self.v_freestream, _ = flight_conditions.calculate_free_stream_velocity(Mach, FL)
        
        # 4.3 Configure nacelle geometry
        self.nac_length = nacelle_params.nac_length  # Nacelle length (m)
        self.A_inlet = nacelle_params.A_inlet  # Inlet capture area (m²)
        self.inlet_radius = math.sqrt(self.A_inlet / math.pi)  # Equivalent circular radius (m)

    def calculate_zero_lift_drag(self):
        # 4.4 Calculate dynamic viscosity using Sutherland's formula
        # Theory: μ = μ_ref*(T/T_ref)^1.5*(T_ref+S)/(T+S)
        mu = (18.27e-6) * (411.15/(self.T + 120)) * (self.T/291.15)**1.5  # (Pa·s)
        
        # 4.5 Surface roughness parameters
        k = 10e-6  # Equivalent sand roughness for aluminum (m)
        
        # 4.6 Calculate Reynolds number with roughness limitation
        # Theory: Re = ρ*v*L/μ, Re_cutoff prevents overestimation in fully rough regime
        Re = (self.rho * self.v_freestream * self.nac_length) / mu
        Re0 = 38 * (self.nac_length / k) ** 1.053  # Critical Reynolds number
        Re = min(Re, Re0)  # Apply cutoff for fully rough flow
        
        # 4.7 Calculate skin friction coefficient
        # Theory: Laminar flat plate cf (conservative assumption for BLI flow)
        cf = 1.328 / math.sqrt(Re)
        
        # 4.8 Calculate form factor adjustment
        f = self.nac_length / (2 * self.inlet_radius)  # Fineness ratio
        Fnac = 1 + (0.35 / f)  # Form factor (1.0 = ideal streamlined body)
        
        # 4.9 Calculate zero-lift drag coefficient
        Cdzero = cf * Fnac
        
        # 4.10 Compute wetted area
        # Theory: Cylindrical surface area (πDL) for simplified nacelle
        Snacwet = math.pi * 2 * self.inlet_radius * self.nac_length  # (m²)
        
        # 4.11 Calculate total parasite drag force
        # Theory: D = 0.5*ρ*v²*C_d*A
        fnacparasite = cf * Fnac * Snacwet
        Dzero = 0.5 * self.rho * self.v_freestream**2 * fnacparasite  # (N)
        
        return Dzero, Cdzero, mu
    
 
#---------------------------------------# 
# Section 5 - Engine Mass Estimation #
#---------------------------------------# 
class EngineMassEstimation:
    """
    Estimates propulsion system mass using empirical relationships
    Theory: NASA-derived scaling laws for electric propulsion systems considering:
            - Motor power density
            - Disk loading effects
            - Structural scaling
            - Cooling requirements
    """
    
    def __init__(self, actuator_disk):
        # 5.1 Initialize with actuator disk model reference
        self.disk = actuator_disk  # Reference to ActuatorDiskModel instance
        self._calculate_derived_params()  # Calculate derived parameters immediately

    def _calculate_derived_params(self):
        """Calculate key performance parameters for mass estimation"""
        # 5.2 Extract core performance parameters
        self.power_total = self.disk.calculate_total_power()  # Total electrical power [kW]
        self.A_eff = self.disk.A_effective  # Effective actuator disk area [m²]
        self.m_dot = self.disk.calculate_mass_flow_rate()  # Mass flow rate [kg/s]

        # 5.3 Calculate propulsion system loadings
        # Theory: Key metrics for component mass scaling
        self.disk_loading = (self.disk.calculate_thrust() / self.A_eff)  # Thrust/area [N/m²]
        self.power_loading = (self.disk.calculate_thrust() * 1e-3) / self.power_total  # Thrust/power [kN/kW]

    def calculate_total_motor_mass(self):
        """Calculate total propulsion system mass with contingency"""
        # 5.4 Electric motor mass correlation
        # Theory: NASA N3-X scaling: mass ~ power^0.72
        motor_mass = 1.8 * self.power_total**0.72  # [kg]
        
        # 5.5 Rotor/propeller mass estimation
        # Theory: Composite blade scaling with disk loading
        rotor_mass = 0.15 * self.A_eff * (self.disk_loading/1000)**0.6  # [kg]
        
        # 5.6 Structural mass components
        # Theory: Nacelle scaling with area + motor support structure
        struct_mass = 4.2 * (self.A_eff**0.65) + 0.12 * motor_mass  # [kg]
        
        # 5.7 Cooling system requirements
        # Theory: Liquid cooling needed above 200kW threshold
        cooling_mass = 0.07 * self.power_total if self.power_total > 200 else 0  # [kg]
        
        # 5.8 Total mass with design contingency
        # Theory: 12% margin for brackets, connectors, and unaccounted components
        return (motor_mass + rotor_mass + struct_mass + cooling_mass) * 1.12  # [kg]
    
#---------------------------------------# 
# Section 6 - Fuselage Flow Analysis (Continued) #
#---------------------------------------# 
class Flow_around_fuselage:
    def __init__(self, v_freestream, Mach, rho, mu, delta_p, A_inlet, p,
                propulsor_position=33.0, nacelle_length=None,  
                eta_disk=None, eta_motor=None, eta_prop=None):
        # 6.1 Initialize core flow parameters
        self.A_inlet = A_inlet  # Propulsor capture area [m²]
        self.fuselage_length = 38  # Total fuselage length [m]
        self.fuselage_radius = 2.0  # Maximum fuselage radius [m]
        self.free_stream_velocity = v_freestream  # Freestream velocity [m/s]
        self.rho = rho  # Air density [kg/m³]
        # 6.2 Set component efficiencies with defaults
        self.ηdisk = eta_disk or _EFFICIENCY_DEFAULTS["eta_disk"]  # Disk efficiency
        self.ηmotor = eta_motor or _EFFICIENCY_DEFAULTS["eta_motor"]  # Motor efficiency
        self.ηprop = eta_prop or _EFFICIENCY_DEFAULTS["eta_prop"]  # Propeller efficiency
        
        # 6.3 Calculate BLI parameters
        # Theory: Empirical scaling for maximum boundary layer thickness (Kaiser et al.)
        self.delta_99_max = 0.18 * self.fuselage_length * (A_inlet/12.0)**0.25
        
        # 6.4 Propulsor disk geometry
        # Theory: 10% area contraction for flow acceleration
        A_disk = A_inlet * 0.9  # Effective disk area [m²]
        self.disk_radius = math.sqrt(A_disk / math.pi)  # Disk radius [m]
        disk_diameter = 2 * self.disk_radius  # Propulsor diameter [m]
        
        # 6.5 Suction parameters initialization
        # Theory: Base suction strength proportional to capture area
        self.suction_strength = 0.06 * A_inlet  # Initial suction parameter [m²/s]
        
        # 6.6 Mass flow ratio calculation
        # Theory: Engine mass flow vs boundary layer mass flow ratio
        mdot_engine = rho * v_freestream * A_inlet  # Engine mass flow [kg/s]
        avg_bl_velocity = 0.7 * v_freestream  # Average BL velocity [m/s]
        bl_area = self.delta_99_max * self.fuselage_radius  # BL cross-section [m²]
        mdot_bl = rho * avg_bl_velocity * bl_area  # BL mass flow [kg/s]
        flow_ratio = mdot_engine / (mdot_bl + 1e-6)  # Avoid division by zero
        self.suction_strength *= np.clip(flow_ratio, 0.5, 2.0)  # Scale suction strength

        # 6.7 Define influence regions
        # Theory: Empirical scaling for suction/recovery zone widths
        self.suction_width = 3.0 * disk_diameter  # Upstream influence region [m]
        self.recovery_width = 0.00001 * disk_diameter  # Downstream recovery zone [m]

        # 6.8 Initialize remaining parameters
        self.nacelle_length = nacelle_length  # Nacelle length [m]
        self.delta_p = delta_p  # Pressure difference [Pa]
        self.nose_length = 3.0  # Nose section length [m]
        self.tail_length = 10.0  # Tail section length [m]
        self.Mach = Mach  # Flight Mach number
        self.p = p  # Static pressure [Pa]
        self.mu = mu  # Dynamic viscosity [Pa·s]
        self.propulsor_position = propulsor_position  # Axial position [m]
        
        # 6.9 Initialize force metrics
        self.T_net = 0.0  # Net thrust [N]
        self.D_red = 0.0  # Drag reduction [N]
        self.PSC = 0.0  # Power Saving Coefficient
        self.disk_active = False  # Propulsor state flag

        # 6.10 Validate Mach number range
        if self.Mach >= 1:
            raise ValueError("The model is invalid for Mach ≥ 1")
        if self.Mach < 0.3:  
            raise ValueError("The model is invalid for Mach < 0.3")

        # 6.11 Create fuselage geometry
        # Theory: Piecewise construction (nose cone + cylinder + tail cone)
        N = 1000  # Number of stations
        self.x = np.linspace(0, self.fuselage_length, N)
        self.Re_x = self.rho * self.free_stream_velocity * self.x / self.mu
        self.y_upper = np.zeros(N)
        self.y_lower = np.zeros(N)

        # 6.12 Build fuselage cross-section
        for i, xi in enumerate(self.x):
            if xi <= self.nose_length:
                # Nose cone (parabolic profile)
                y = self.fuselage_radius * (1 - ((xi - self.nose_length)/self.nose_length)**2)
                self.y_upper[i] = y
                self.y_lower[i] = -y
            elif xi >= self.fuselage_length - self.tail_length:
                # Tail cone (parabolic profile)
                x_tail = xi - (self.fuselage_length - self.tail_length)
                y = self.fuselage_radius * (1 - (x_tail/self.tail_length)**2)
                self.y_upper[i] = y
                self.y_lower[i] = -y
            else:
                # Cylindrical section
                self.y_upper[i] = self.fuselage_radius
                self.y_lower[i] = -self.fuselage_radius

        # 6.13 Calculate geometric derivatives
        self.R = np.abs(self.y_upper)  # Local radius array [m]
        self.dR_dx = np.gradient(self.R, self.x)  # Radius gradient [m/m]

        # 6.14 Validate propulsor installation
        idx = np.argmin(np.abs(self.x - propulsor_position))
        R_prop = self.y_upper[idx]
        self.effective_A_disk = np.pi * (self.disk_radius**2 - R_prop**2)
        if self.effective_A_disk <= 0:
            raise ValueError("Disk radius must exceed fuselage radius at propulsor")

        # 6.15 Initialize nacelle parameters
        self.nacelle = NacelleParameters(
            v_inlet=self.free_stream_velocity,
            A_inlet=self.effective_A_disk,
            nacelle_length=nacelle_length,   
            eta_disk=eta_disk,
            eta_motor=eta_motor,
            eta_prop=eta_prop
        )
        
        # 6.16 Extract nacelle performance parameters
        # Preserve original index-based parameter extraction
        nacelle_params = self.nacelle.variable_parameters(rho=self.rho, p=self.p)
        self.delta_p = nacelle_params[10]  # Pressure difference [Pa] (original index)
        v_disk = nacelle_params[5]  # Disk velocity [m/s] (original index)

        # 6.17 Store efficiency parameters
        self.eta_disk = self.nacelle.ηdisk
        self.eta_motor = self.nacelle.ηmotor
        self.eta_prop = self.nacelle.ηprop
        self.eta_total = self.nacelle.ηTotal

        # 6.18 Initialize actuator disk model
        # Theory: Uses nacelle-derived parameters for propulsion modeling
        self.actuator_disk = ActuatorDiskModel(
            rho=self.rho,
            A_effective=self.effective_A_disk,
            v_inlet=self.free_stream_velocity,  
            v_disk=v_disk,  # From nacelle_params[5]
            eta_disk=self.eta_disk,
            eta_motor=self.eta_motor,
            eta_prop=self.eta_prop
        )

        # 6.19 Initialize spatial discretization
        self.dx = self.x[1] - self.x[0]  # Spatial step [m]
        self.source_strength = self.source_strength_thin_body()  # Source distribution

        # 6.20 Initialize boundary layer arrays
        self.delta_99 = np.zeros_like(self.x)  # 99% BL thickness [m]
        self.delta_star = np.zeros_like(self.x)  # Displacement thickness [m]
        self.theta = np.zeros_like(self.x)  # Momentum thickness [m]
        self.tau_wall = np.zeros_like(self.x)  # Wall shear stress [Pa]
        self.nu_t = np.zeros_like(self.x)  # Turbulent viscosity [m²/s]

        # 6.21 Initialize result storage
        self.results_with_propulsor = None
        self.results_without_propulsor = None 

    def source_strength_thin_body(self):
        """Calculate source strength distribution for thin body approximation."""
        # 6.22 Initialize derivative array
        dr2_dx = np.zeros_like(self.x)  # d(r²)/dx array initialization

        # 6.23 Apply Prandtl-Glauert compressibility correction - It s not used anymore 
 
        
        # 6.24 Calculate source distribution along fuselage
        for i, xi in enumerate(self.x):
            if xi <= self.nose_length:
                # Nose section source strength
                # Theory: Parabolic nose shape derivative from Kaiser's method
                # Ref: QUASI ANALYTICAL AERODYNAMIC METHODS FOR PROPULSIVE FUSELAGE CONCEPTS
                term = (xi - self.nose_length)/self.nose_length
                dr2_dx[i] = 2 * (self.fuselage_radius)**2 * (1 - term**2) * (-2 * term/self.nose_length)
                
            elif xi >= self.fuselage_length - self.tail_length:
                # 6.25 Tail section source strength
                # Theory: Parabolic tail shape derivative with PG correction
                x_tail = xi - (self.fuselage_length - self.tail_length)
                term = x_tail/self.tail_length
                dr2_dx[i] = 2 * (self.fuselage_radius)**2 * (1 - term**2) * (-2 * term/self.tail_length)
                
            else:
                # 6.26 Cylindrical section handling
                # Theory: No source strength in constant cross-section region
                dr2_dx[i] = 0.0

        # 6.27 Return scaled source strength
        # Theory: Q(x) = V_∞ * π * d(r²)/dx 
        # Ref: https://www.bauhaus-luftfahrt.net/fileadmin/.../Propulsive_Fuselage_Concepts_X.pdf
        return self.free_stream_velocity * np.pi * dr2_dx
    
    def plot_source_strength(self, canvas_frame):
        """Visualize source strength distribution along fuselage with interactive features"""
        # 6.28 Clear previous visualization
        for widget in canvas_frame.winfo_children():
            widget.destroy()

        # 6.29 Create figure and axis setup
        # Theory: Dual-axis plot for geometry + source strength visualization
        plt.rcParams.update({'font.size': 12})  # Base font size for all elements
        fig, ax1 = plt.subplots(figsize=(12, 4))
        
        # 6.30 Plot fuselage geometry
        # Theory: Upper/lower surfaces with filled area for visual clarity
        line_upper, = ax1.plot(self.x, self.y_upper, 'k', linewidth=2, label='Fuselage')
        line_lower, = ax1.plot(self.x, self.y_lower, 'k', linewidth=2)
        ax1.fill_between(self.x, self.y_upper, self.y_lower, color='lightgray', alpha=0.5)
        
        # 6.31 Configure geometry axis
        ax1.set_aspect(0.5)  # Maintain proportional scaling
        y_max = max(self.y_upper) * 1.1  # 10% margin above fuselage
        ax1.set_ylim(-y_max, y_max)
        ax1.set_xlabel('Axial Position [m]', fontsize=14)
        ax1.set_ylabel('Vertical Position [m]', fontsize=14)
        ax1.grid(True)
        ax1.set_title('Source Strength Distribution', fontsize=16, pad=20)

        # 6.32 Create source strength axis
        ax2 = ax1.twinx()
        line_source, = ax2.plot(self.x, self.source_strength, 'r', label='Source Strength Q(x) [m²/s]')
        
        # 6.33 Configure source strength axis
        q_max = max(abs(self.source_strength)) * 1.1  # Symmetric limits
        ax2.set_ylim(-q_max, q_max)
        ax2.set_ylabel('Source Strength Q(x) [m²/s]', fontsize=14)
        
        # 6.34 Create unified legend
        lines = [line_upper, line_source]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', 
                 bbox_to_anchor=(0.5, -0.15), ncol=2,
                 fontsize=16, framealpha=0.95)

        # 6.35 Add interactive hover tool
        def on_hover(sel):
            """Display real-time data values on cursor hover"""
            x_val = sel.target[0]
            idx = np.argmin(np.abs(self.x - x_val))
            sel.annotation.set_text(
                f"x: {x_val:.2f}m\n"
                f"Q(x): {self.source_strength[idx]:.2f} m²/s\n"
                f"Radius: {self.y_upper[idx]:.2f}m"
            )
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
            sel.annotation.set_fontsize(16)
        
        cursor = mplcursors.cursor(line_source, hover=True)
        cursor.connect("add", on_hover)

        # 6.36 Finalize layout
        fig.tight_layout()
        
        # 6.37 Embed in Tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        
        # 6.38 Add navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def velocity_components_around_fuselage(self, X, Y, apply_mask=True):
        """
        Calculate velocity field around fuselage using source superposition
        Theory: Potential flow solution from distributed sources along fuselage centerline
                Ref: MIT OCW 2.016 Hydrodynamics, Lecture 4 - Source Panel Method
                Ref: https://ocw.mit.edu/courses/2-016-hydrodynamics-13-012-fall-2005/c472432debcf6ee250209b68cf18cc12_2005reading4.pdf
        """
        # 6.39 Initialize velocity arrays
        # Theory: Start with free-stream values, add source-induced velocities
        U = np.full(X.shape, self.free_stream_velocity, dtype=np.float64)  # X-velocity component [m/s]
        V = np.zeros(Y.shape, dtype=np.float64)  # Y-velocity component [m/s]

        # 6.40 Superpose source contributions
        # Theory: Each source element contributes velocity per potential flow equations
        for i in range(len(self.x)):
            if self.source_strength[i] == 0:
                continue  # Skip inactive sources
            
            # 6.41 Calculate spatial relationships
            dx = X - self.x[i]  # X-distance from source [m]
            dy = Y  # Y-distance from source [m]
            r_sq = dx**2 + dy**2 + 1e-6  # Squared distance + epsilon for numerical stability

            # 6.42 Calculate source-induced velocities
            # Theory: u = (Q/2π) * (Δx/r²), v = (Q/2π) * (Δy/r²)
            source_effect = (self.source_strength[i] * self.dx) / (2 * np.pi)
            U += source_effect * (dx / r_sq)  # Add x-component
            V += source_effect * (dy / r_sq)  # Add y-component

        # 6.43 Apply fuselage masking
        if apply_mask:
            epsilon = 1e-6  # Tolerance for geometric masking
            for i in range(len(self.x)):
                # 6.44 Create fuselage exclusion zone
                # Theory: Mask velocities inside physical fuselage boundaries
                x_mask = (X >= self.x[i] - self.dx/2) & (X <= self.x[i] + self.dx/2)
                y_mask = (Y > -self.R[i] - epsilon) & (Y < self.R[i] + epsilon)
                
                # 6.45 Nullify velocities within fuselage
                U[x_mask & y_mask] = np.nan
                V[x_mask & y_mask] = np.nan

        return U, V
    
    def net_velocity_calculation(self, x_pos):
        """
        Calculate resultant velocity magnitude at specified axial position
        Theory: Computes boundary layer edge velocity for integral calculations
                using potential flow solution at fuselage surface
        """
        # 6.46 Find nearest fuselage station index
        idx = np.argmin(np.abs(self.x - x_pos))  # Closest grid point search
        
        # 6.47 Calculate velocity components at surface point
        # Note: Mask disabled to get velocity at fuselage surface (y = R)
        U, V = self.velocity_components_around_fuselage(
            self.x[idx], 
            self.y_upper[idx],  # Surface y-coordinate
            apply_mask=False  # Essential for boundary layer calculations
        )
        
        # 6.48 Compute velocity magnitude
        # Theory: V = √(U² + V²) from potential flow solution
        return np.sqrt(U**2 + V**2)
    
    def plot_velocity_streamlines(self, canvas_frame):
        """
        Visualize flow field with streamlines and velocity magnitude coloring
        Theory: Potential flow solution visualization using streamlines
                Enhanced with power-law normalized color mapping for better
                low-velocity resolution
        """
        # 6.49 Clear previous visualization
        for widget in canvas_frame.winfo_children():
            widget.destroy()

        # 6.50 Create computational grid
        x = np.linspace(-10, self.fuselage_length + 10, 100)  # X-grid with 10m margins
        y = np.linspace(-10, 10, 100)  # Y-grid symmetric about fuselage
        X, Y = np.meshgrid(x, y)
        
        # 6.51 Calculate velocity field
        U, V = self.velocity_components_around_fuselage(X, Y)
        
        # 6.52 Clean and process velocity data
        U = np.nan_to_num(U, nan=0.0)  # Handle masked fuselage regions
        V = np.nan_to_num(V, nan=0.0)
        vel_magnitude = np.sqrt(U**2 + V**2)  # Velocity magnitude [m/s]

        # 6.53 Create figure and streamplot
        fig, ax = plt.subplots(figsize=(10, 6))
        norm = mcolors.PowerNorm(gamma=2,  # Enhanced low-velocity visibility
                               vmin=vel_magnitude.min(), 
                               vmax=vel_magnitude.max())
        
        # 6.54 Generate streamline plot
        strm = ax.streamplot(
            X, Y, U, V, 
            color=vel_magnitude,
            cmap='jet',
            norm=norm,
            linewidth=1,  # Balance between visibility and clutter
            density=2,    # Streamline spacing
            arrowsize=1   # Direction indicator size
        )
        
        # 6.55 Add velocity magnitude colorbar
        cbar = fig.colorbar(strm.lines, ax=ax, label='Velocity Magnitude (m/s)')
        
        # 6.56 Plot fuselage geometry
        ax.plot(self.x, self.y_upper, 'k', linewidth=2)
        ax.plot(self.x, self.y_lower, 'k', linewidth=2)
        ax.fill_between(self.x, self.y_upper, self.y_lower, 
                       color='lightgray', alpha=0.5)
        
        # 6.57 Configure plot axes
        ax.set(
            xlabel='Axial Position [m]',
            ylabel='Vertical Position [m]',
            title='2D Velocity Field Around Fuselage',
            aspect='equal'  # Maintain proper geometric proportions
        )
        ax.grid(True)

        # 6.58 Implement hover tooltip
        def on_hover(sel):
            """Display real-time flow parameters at cursor position"""
            try:
                x_pos, y_pos = sel.target[0], sel.target[1]
                x_idx = np.argmin(np.abs(X[0,:] - x_pos))
                y_idx = np.argmin(np.abs(Y[:,0] - y_pos))
                
                sel.annotation.set_text(
                    f"Position: ({x_pos:.2f}, {y_pos:.2f}) m\n"
                    f"Velocity: {vel_magnitude[y_idx, x_idx]:.2f} m/s\n"
                    f"Components: U={U[y_idx, x_idx]:.2f}, V={V[y_idx, x_idx]:.2f}"
                )
            except IndexError:
                sel.annotation.set_text("Position out of data range")

        cursor = mplcursors.cursor(strm.lines, hover=True)
        cursor.connect("add", on_hover)

        # 6.59 Embed visualization in GUI
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        
        # 6.60 Add interactive navigation tools
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def pressure_distribution(self):
        """
        Computes inviscid pressure distribution without BLI effects.
        Theory: Uses potential flow solution with compressibility correction
                - Bernoulli's equation for incompressible Cp
                - Prandtl-Glauert transformation for subsonic compressibility
        """
        # 6.61 Get velocity components at fuselage surface
        # Theory: Potential flow solution along upper fuselage contour
        U, V = self.velocity_components_around_fuselage(
            self.x, 
            self.y_upper,
            apply_mask=False  # Essential for surface pressure calculation
        )
        
        # 6.62 Calculate velocity magnitude ratio
        # Theory: V/V∞ ratio for Bernoulli equation application
        velocity_ratio = np.sqrt(U**2 + V**2) / self.free_stream_velocity
        
        # 6.63 Compute incompressible pressure coefficient
        # Theory: Cp = 1 - (V/V∞)^2 (Bernoulli's equation)
        # Ref: https://en.wikipedia.org/wiki/Pressure_coefficient
        Cp_incompressible = 1 - velocity_ratio**2
        
        # 6.64 Apply compressibility correction
        # Theory: Prandtl-Glauert transformation for Mach < 1
        # Ref: https://en.wikipedia.org/wiki/Prandtl-Glauert_transformation
        Cp_compressible = Cp_incompressible / np.sqrt(1 - self.Mach**2)
        
        return Cp_incompressible, Cp_compressible
    
    def compute_pressure_gradient(self):
        """
        Compute pressure gradient using inviscid Euler equation.
        Theory: Derived from momentum equation neglecting viscous terms,
                crucial for boundary layer analysis. Used to drive
                boundary layer development through pressure gradient term.
        Ref: NASA Beginner's Guide to Aeronautics - Equation 11
             https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/conmo508.html
        """
        # 6.65 Calculate edge velocity profile
        # Theory: Boundary layer edge velocity from potential flow solution
        U_e = np.array([self.net_velocity_calculation(xi) for xi in self.x])
        
        # 6.66 Compute velocity gradient
        # Theory: Central difference scheme (2nd order accurate)
        # Ref: https://aerodynamics4students.com/subsonic-aerofoil-and-wing-theory/2d-boundary-layer-modelling.php
        dU_e_dx = np.gradient(U_e, self.x)  # ∂U_e/∂x [1/s]
        
        # 6.67 Apply Euler equation for pressure gradient
        # Theory: dp/dx = -ρU_e(dU_e/dx) from inviscid momentum equation
        dp_dx = -self.rho * U_e * dU_e_dx  # Pressure gradient [Pa/m]
        
        return dp_dx

    def plot_pressure_distribution(self, canvas_frame):
        """
        Visualize pressure distribution with interactive comparison of
        incompressible/compressible solutions and fuselage-wing geometry
        """
        # 6.68 Clear previous visualization
        for widget in canvas_frame.winfo_children():
            widget.destroy()

        # 6.69 Create figure and axis setup
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 6.70 Calculate pressure coefficients
        Cp_incompressible, Cp_compressible = self.pressure_distribution()
        
        # 6.71 Plot incompressible solution
        line_incomp, = ax1.plot(self.x, Cp_incompressible, 'b', label='Incompressible Cp')
        # 6.72 Plot compressible solution
        line_comp, = ax1.plot(self.x, Cp_compressible, 'r', label='Compressible Cp')
        
        # Configure primary axis
        ax1.set_xlabel('Axial Position [m]')
        ax1.set_ylabel('Pressure Coefficient (Cp)')
        ax1.grid(True)

        # 6.73 Create geometry visualization axis
        ax2 = ax1.twinx()
        line_upper, = ax2.plot(self.x, self.y_upper, 'k', linewidth=2, label='Fuselage')
        line_lower, = ax2.plot(self.x, self.y_lower, 'k', linewidth=2)
        ax2.fill_between(self.x, self.y_upper, self.y_lower, color='lightgray', alpha=0.5)

        # 6.74 Add wing profile (NACA 0012)
 

        # 6.77 Configure geometry axis
        ax2.set_aspect(0.5)  # Maintain fuselage proportions
        y_max = max(self.y_upper) * 1.1
        ax2.set_ylim(-y_max, y_max)
        ax2.set_ylabel('Vertical Position [m]')

        # 6.78 Create unified legend
        lines = [line_incomp, line_comp, line_upper]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', 
                bbox_to_anchor=(0.5, -0.15), ncol=4)

        # 6.79 Implement hover functionality
        def on_hover(sel):
            """Display context-sensitive data on hover"""
            try:
                artist = sel.artist
                x_val = sel.target[0]
                idx = np.argmin(np.abs(self.x - x_val))
                
                if artist == line_incomp or artist == line_comp:
                    text = (f"x: {x_val:.2f}m\n"
                            f"Incomp Cp: {Cp_incompressible[idx]:.2f}\n"
                            f"Comp Cp: {Cp_compressible[idx]:.2f}")
                elif artist == line_upper:
                    text = (f"x: {x_val:.2f}m\n"
                            f"Fuselage Y: {self.y_upper[idx]:.2f}m")
                else:
                    text = f"x: {x_val:.2f}m"
                
                sel.annotation.set_text(text)
            except Exception as e:
                sel.annotation.set_text("Data not available")

        cursor = mplcursors.cursor([line_incomp, line_comp, line_upper], hover=True)
        cursor.connect("add", on_hover)

        # 6.80 Finalize layout
        fig.tight_layout()

        # 6.81 Embed in GUI
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        
        # 6.82 Add navigation tools
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def solve_boundary_layer(self):
        """
        Solves boundary layer development using integral methods
        Theory: Integrates momentum and displacement thickness equations
                with pressure gradient effects. Uses:
                - Blasius solution for laminar initial conditions
                - 1/7th power law for turbulent flow
                - LSODA adaptive ODE solver for stability
        """
        # 6.83 Calculate pressure gradient and kinematic viscosity
        dp_dx = self.compute_pressure_gradient()  # Pressure gradient [Pa/m]
        nu = self.mu / self.rho  # Kinematic viscosity [m²/s]

        # 6.84 Determine initial conditions
        x_initial = 0.01  # Starting point to avoid singularity [m]
        Re_x_initial = self.rho * self.free_stream_velocity * x_initial / self.mu

        # 6.85 Set initial BL parameters based on flow regime
        if Re_x_initial < 5e5:  # Laminar flow (Blasius solution)
            # Theory: θ = 0.664√(νx/U_∞) (An Introduction to the Mechanics of Incompressible Fluids, Eq. 7.58 and 7.65)
            theta_initial = 0.664 * np.sqrt(nu * x_initial / self.free_stream_velocity)
            delta_99_initial = 5.0 * np.sqrt(nu * x_initial / self.free_stream_velocity)
        else:  # Turbulent flow (1/7th power law)
            # Theory: θ ≈ 0.016x/Re_x^{1/7} (Schlichting Boundary Layer Theory)
            theta_initial = 0.016 * x_initial / (Re_x_initial ** (1/7))
            delta_99_initial = 0.16 * x_initial / (Re_x_initial ** (1/7))

        # 6.86 Solve boundary layer ODE system
        sol = solve_ivp(
            self._boundary_layer_ode_system,
            [x_initial, self.x[-1]],  # Integration bounds
            [theta_initial, delta_99_initial],  # Initial conditions
            args=(dp_dx,),  # Pressure gradient array
            t_eval=self.x[self.x >= x_initial],  # Output points
            method='LSODA',  # Adaptive stiff/non-stiff solver
            atol=1e-6,  # Absolute error tolerance
            rtol=1e-6  # Relative error tolerance
        )
        
        # 6.87 Interpolate and post-process results
        self.delta_99 = np.interp(self.x, sol.t, sol.y[1])  # 99% thickness
        self.theta = np.interp(self.x, sol.t, sol.y[0])  # Momentum thickness
        
        # 6.88 Apply physical constraints
        self.delta_99 = np.clip(self.delta_99, 0, self.delta_99_max)  # Prevent overshoot
        
        # 6.89 Calculate derived BL parameters
        self.delta_star = self._compute_displacement_thickness()  # Displacement thickness
        self.tau_wall = self._compute_wall_shear_stress()  # Wall shear stress

    def _boundary_layer_ode_system(self, x, y, dp_dx):
        theta, delta_99 = y
        eps = 1e-10
        
        # Get local parameters
        idx = np.clip(np.searchsorted(self.x, x), 0, len(self.x)-1)
        U_e = self.net_velocity_calculation(x)
        Re_x = self.rho * U_e * x / (self.mu + eps)
        laminar = Re_x < 5e5

        if laminar:
            # Blasius solution derivatives
            dtheta_dx = 0.664 * np.sqrt(self.mu/(self.rho * U_e * x)) * (0.5/x)
            ddelta99_dx = 5.0 * dtheta_dx
        else:
            # 1/7-power-law derivatives
            dtheta_dx = 0.016 * (1 - 1/7) * x**(-1/7) * (self.rho * U_e/self.mu)**(-1/7)
            ddelta99_dx = 0.16 * (1 - 1/7) * x**(-1/7) * (self.rho * U_e/self.mu)**(-1/7)

        if self.disk_active:
            suction_start = self.propulsor_position - self.suction_width
            if suction_start <= x <= self.propulsor_position:
                suction_factor = self.suction_strength * (x - suction_start)/self.suction_width
                dtheta_dx -= suction_factor * (theta/self.suction_width)
                ddelta99_dx -= suction_factor * (delta_99/self.suction_width)

        return [dtheta_dx, ddelta99_dx]
    
    def plot_boundary_layer_thickness(self, canvas_frame):
        """
        Visualize boundary layer development comparison with/without BLI
        Theory: Shows both exaggerated visual thickness and absolute values
                using dual-axis plot. Incorporates performance metrics.
        """
        # 6.97 Clear previous visualization
        for widget in canvas_frame.winfo_children():
            widget.destroy()

        # 6.98 Create figure and axis setup
        fig, ax = plt.subplots(figsize=(12, 6))

        # 6.99 Plot fuselage geometry baseline
        ax.plot(self.x, self.y_upper, 'k', linewidth=2, label='Fuselage')
        ax.plot(self.x, self.y_lower, 'k', linewidth=2)
        ax.fill_between(self.x, self.y_upper, self.y_lower, 
                       color='lightgray', alpha=0.5)

        # 6.100 Configure geometry axis
        ax.set_aspect(0.5)  # Maintain fuselage proportions
        y_max = max(self.y_upper) * 1.1  # 10% vertical margin
        ax.set_ylim(-y_max, y_max)

        # 6.101 Get propulsor installation parameters
        idx = np.argmin(np.abs(self.x - self.propulsor_position))
        prop_radius = self.y_upper[idx]  # Fuselage radius at propulsor [m]
        disk_radius = self.disk_radius  # Propulsor outer radius [m]

        # 6.102 Prepare boundary layer visualization parameters
        dy_dx = np.gradient(self.y_upper, self.x)  # Surface slope
        theta = np.arctan(dy_dx)  # Surface angle [rad]
        
        # 6.103 Calculate exaggerated BL thickness for visualization
        exaggeration_factor = 1  # Amplification factor for visible BL
        delta_normal_with = (self.results_with_propulsor["delta_99"] 
                            * np.cos(theta) * exaggeration_factor)
        delta_normal_without = (self.results_without_propulsor["delta_99"]
                               * np.cos(theta) * exaggeration_factor)

        # 6.104 Plot BL thickness with propulsor
        ax.fill_between(
            self.x, 
            self.y_upper + delta_normal_with, 
            self.y_upper, 
            color='red', alpha=0.5, 
            label='BL With Propulsor'
        )
        
        # 6.105 Plot BL thickness without propulsor
        ax.fill_between(
            self.x, 
            self.y_upper + delta_normal_without, 
            self.y_upper, 
            color='blue', alpha=0.3, 
            label='BL Without Propulsor'
        )

        # 6.106 Create absolute thickness axis
        ax2 = ax.twinx()
        
        # 6.107 Plot absolute BL thickness values
        line_without, = ax2.plot(
            self.results_without_propulsor["x"], 
            self.results_without_propulsor["delta_99"], 
            'b--', 
            label='δ99 Without Propulsor'
        )
        line_with, = ax2.plot(
            self.results_with_propulsor["x"], 
            self.results_with_propulsor["delta_99"], 
            'r--', 
            label='δ99 With Propulsor'
        )

        # 6.108 Mark propulsor position
        ax2.axvline(
            x=self.propulsor_position, 
            color='black', 
            linestyle='--', 
            linewidth=1.5, 
            label='Propulsor Position'
        )

        # 6.109 Configure axis labels and grid
        ax2.set_ylim(ax.get_ylim())  # Align y-limits for visual coherence
        ax2.set_ylabel('Absolute Boundary Layer Thickness [m]', color='k')
        ax.set_xlabel('Axial Position [m]')
        ax.set_ylabel('Vertical Position [m]')
        ax.set_title('Boundary Layer Development Comparison')
        ax.grid(True)

        # 6.110 Create unified legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2, 
            labels1 + labels2, 
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.15), 
            ncol=3
        )

        # 6.111 Implement hover functionality
        def on_hover(sel):
            """Display BL thickness values at cursor position"""
            x_val = sel.target[0]
            idx_without = np.argmin(np.abs(self.results_without_propulsor["x"] - x_val))
            idx_with = np.argmin(np.abs(self.results_with_propulsor["x"] - x_val))
            sel.annotation.set_text(
                f"x: {x_val:.2f}m\n"
                f"Without: {self.results_without_propulsor['delta_99'][idx_without]:.3f}m\n"
                f"With: {self.results_with_propulsor['delta_99'][idx_with]:.3f}m"
            )

        cursor = mplcursors.cursor([line_without, line_with], hover=True)
        cursor.connect("add", on_hover)

        # 6.112 Display performance metrics
        metrics_text = (
            f"Net Thrust: {self.T_net:.2f} N\n"
            f"Drag Reduction: {self.D_red:.2f} N\n"
            f"PSC: {self.PSC:.1%}"
        )
        ax.text(
            0.02, 0.90, metrics_text, 
            transform=ax.transAxes,
            fontsize=12, 
            bbox=dict(facecolor='wheat', alpha=0.5)
        )

        # 6.113 Finalize layout
        fig.tight_layout()

        # 6.114 Embed in GUI with navigation tools
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        NavigationToolbar2Tk(canvas, canvas_frame).pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def boundary_layer_velocity_profile(self, x_pos, y):
        """
        Compute velocity at (x, y) considering boundary layer effects
        Theory: Uses empirical velocity profiles (laminar/turbulent) with
                physical safeguards for numerical stability
        Ref: White, F.M., Viscous Fluid Flow, McGraw-Hill, 2006
        """
        # 6.115 Get boundary layer edge velocity
        # Theory: Potential flow solution at surface
        U_e = self.net_velocity_calculation(x_pos)  # Edge velocity [m/s]
        
        # 6.116 Retrieve BL parameters at position
        idx = np.argmin(np.abs(self.x - x_pos))  # Nearest grid index
        delta_99 = max(self.delta_99[idx], 1e-6)  # Clamped 99% thickness [m]
        R_fuselage = self.R[idx]  # Local fuselage radius [m]

        # 6.117 Calculate wall-normal distance
        # Safeguards: Clamp y between fuselage surface and BL edge
        y_wall = y - R_fuselage  # Distance from surface [m]
        y_wall = np.clip(y_wall, 0.0, delta_99)  # Prevent negative/overshoot values

        # 6.118 Handle invalid flow regions
        if delta_99 < 1e-6 or y_wall < 0:
            return 0.0  # No flow inside structure or invalid BL

        # 6.119 Determine flow regime
        Re_local = self.Re_x[idx]  # Local Reynolds number

        # 6.120 Laminar velocity profile (Blasius solution)
        if Re_local < 5e5:
            # Theory: Parabolic profile u/U_e = 2(y/δ) - (y/δ)^2
            return U_e * (2*(y_wall/delta_99) - (y_wall/delta_99)**2)
        
        # 6.121 Turbulent velocity profile (1/7th power law)
        else:
            # Theory: Empirical power law u/U_e = (y/δ)^{1/7}
            return U_e * (y_wall/delta_99)**(1/7)
        
    def get_local_velocity_at_propulsor(self):
        """
        Compute mass-averaged velocity over actuator disk using Kaiser's method
        Theory: Integrates BL-modified velocity profile across disk area
                - Combines boundary layer and freestream contributions
                - Accounts for shape factor effects in turbulent flow
        Ref: Kaiser, "Quasi-Analytical Methods...", ICAS 2014
        """
        # 6.122 Get propulsor installation parameters
        idx = np.argmin(np.abs(self.x - self.propulsor_position))
        R_prop = self.y_upper[idx]  # Fuselage radius at propulsor [m]
        R_disk = self.disk_radius   # Propulsor outer radius [m]
        delta_99 = self.delta_99[idx]  # BL thickness [m]
        x_disk = self.propulsor_position  # Axial position [m]

        # 6.123 Numerical stability parameters
        epsilon = 1e-10  # Prevent division by zero
        delta_99_min = 1e-6  # Minimum BL thickness [m]

        # 6.124 Define velocity profile function (Kaiser Eq. 10)
        def boundary_layer_velocity(x, r):
            """Velocity profile with shape factor correction"""
            Re_theta = (self.rho * self.net_velocity_calculation(x) 
                       * self.theta[idx]) / (self.mu + epsilon)
            
            # 6.125 Calculate shape factor with clamping
            theta_clamped = self.theta[idx] + epsilon
            H = self.delta_star[idx] / theta_clamped  # Shape factor [-]

            if Re_theta > 2000:  # Turbulent flow
                # Theory: 1/7th power law with H correction (Kaiser Eq.10a)
                return (self.free_stream_velocity 
                       * (1 - (r - R_prop)/delta_99)**(1/7) 
                       * (1 - 0.2*(H - 1.3)))
            else:  # Laminar flow
                # Theory: Quadratic profile (Kaiser Eq.10b)
                delta_99_clamped = max(delta_99, delta_99_min)
                term = (r - R_prop)/delta_99_clamped
                return self.free_stream_velocity * (2*term - term**2)

        # 6.126 Sample velocity across disk radius
        if (R_prop + delta_99) >= R_disk:  # Entire disk in BL
            r = np.linspace(R_prop, R_disk, 100)  # Radial points [m]
            velocities = [boundary_layer_velocity(x_disk, y) for y in r]
        else:  # Combined BL + freestream
            # 6.127 Split into BL and freestream regions
            r_bl = np.linspace(R_prop, R_prop + delta_99, 50)
            v_bl = [boundary_layer_velocity(x_disk, y) for y in r_bl]
            r_fs = np.linspace(R_prop + delta_99 + 1e-6, R_disk, 50)
            v_fs = [self.free_stream_velocity] * len(r_fs)
            r = np.concatenate([r_bl, r_fs])
            velocities = np.concatenate([v_bl, v_fs])

        # 6.128 Compute mass-averaged velocity (Kaiser Eq.11)
        # Theory: V_avg = (2∫u(r)·r dr) / (R_disk² - R_prop²)
        numerator = np.trapz([u*r_i for u, r_i in zip(velocities, r)], r)
        denominator = 0.5 * (R_disk**2 - R_prop**2)
        return (2 * numerator) / (denominator + epsilon)  # [m/s]
    
    def compute_skin_friction(self, i):
        """
        Calculate skin friction coefficient using empirical correlations
        Theory: 
        - Laminar: Blasius solution (Cf = 0.664/√Re_x)
        - Turbulent: 1/7th power law (Cf = 0.027/Re_x^{1/7})
        Ref: White, F.M., Viscous Fluid Flow, 3rd Ed., McGraw-Hill 2006
        """
        # 6.129 Get local Reynolds number
        Re_x = self.Re_x[i]  # Reynolds number at station i
        
        # 6.130 Laminar flow correlation
        if Re_x < 5e5:  
            # Theory: Blasius flat plate solution (Chapter 4)
            Cf = 0.664 / np.sqrt(Re_x + 1e-10)  # Prevent division by zero
        
        # 6.131 Turbulent flow correlation  
        else:  
            # Theory: Empirical power law (Chapter 6)
            Cf = 0.027 / (Re_x ** (1/7))   
        
        return Cf

    def _compute_displacement_thickness(self):
        """
        Calculate displacement thickness (δ*) using shape factor
        Theory: δ* = H·θ where H is:
        - 2.59 for laminar flow
        - 1.29 for turbulent flow
        """
        # 6.132 Determine shape factor
        H = np.where(self.Re_x < 5e5, 2.59, 1.29)  # Laminar/turbulent selector
        
        # 6.133 Compute displacement thickness
        return H * self.theta  # δ* = H·θ [m]

    def _compute_wall_shear_stress(self):
        """
        Calculate wall shear stress distribution
        Theory: τ_w = 1/2 ρ U_∞² C_f
        Ref: Young, D.F., et al., A Brief Introduction to Fluid Mechanics, 5th Ed.
        """
        # 6.134 Initialize array
        tau_wall = np.zeros_like(self.x)  # Wall shear stress [Pa]
        
        # 6.135 Calculate at each station
        for i in range(len(self.x)):
            Cf = self.compute_skin_friction(i)
            # 6.136 Shear stress formula
            tau_wall[i] = Cf * 0.5 * self.rho * self.free_stream_velocity**2  
        
        return tau_wall

    def run_simulation(self):
        """
        Execute full simulation sequence with/without BLI effects
        Theory: Performs comparative analysis of:
                1. Baseline case (no propulsor)
                2. BLI-affected case
                Calculates net performance metrics from both cases
        """
        # 6.137 Run baseline case (no propulsor)
        self.disk_active = False  # Deactivate suction effects
        self._reset_boundary_layer()  # Clear previous BL data
        self.solve_boundary_layer()  # Solve BL equations
        
        # 6.138 Store baseline results
        self.results_without_propulsor = {
            "x": self.x.copy(),
            "delta_99": self.delta_99.copy(),
            "theta": self.theta.copy()
        }

        # 6.139 Run BLI-affected case
        self.disk_active = True  # Activate suction effects
        self._reset_boundary_layer()  # Reset BL parameters
        
        # 6.140 Update propulsion parameters with BLI effects
        v_local = self.get_local_velocity_at_propulsor()  # Mass-averaged velocity
        self.nacelle.v_inlet = v_local  # Update nacelle inflow
        
        # 6.141 Get updated nacelle performance
        nacelle_params = self.nacelle.variable_parameters(rho=self.rho, p=self.p)
        self.delta_p = nacelle_params[10]  # Pressure recovery
        v_disk = nacelle_params[5]  # Disk velocity
        
        # 6.142 Reinitialize actuator disk with BLI parameters
        self.actuator_disk = ActuatorDiskModel(
            rho=self.rho,
            A_effective=self.effective_A_disk,
            v_inlet=v_local,
            v_disk=v_disk,
            eta_disk=self.eta_disk,
            eta_motor=self.eta_motor,
            eta_prop=self.eta_prop
        )
        
        # 6.143 Solve BL with active propulsor
        self.solve_boundary_layer()
        self.results_with_propulsor = {
            "x": self.x.copy(),
            "delta_99": self.delta_99.copy(),
            "theta": self.theta.copy()
        }

        # 6.144 Calculate performance metrics
        self.T_net = self.compute_net_thrust()  # Propulsion benefit
        self.D_red = self.compute_drag_reduction()  # Drag reduction
        self.PSC = self.compute_PSC()  # Power Saving Coefficient

    def _reset_boundary_layer(self):
        """Reset boundary layer arrays to initial state"""
        # 6.145 Initialize BL parameter arrays
        self.delta_99 = np.zeros_like(self.x)  # 99% thickness [m]
        self.theta = np.zeros_like(self.x)     # Momentum thickness [m]
        self.delta_star = np.zeros_like(self.x) # Displacement thickness [m]
        self.tau_wall = np.zeros_like(self.x)   # Wall shear stress [Pa]

    def activate_propulsor(self):
        """Activate the propulsor and update the boundary layer solution."""
        # 6.146 Enable BLI effects
        self.disk_active = True  # Activate suction terms in ODE
        # 6.147 Solve updated BL equations
        self.solve_boundary_layer()

    def deactivate_propulsor(self):
        """Deactivate the propulsor and update the boundary layer solution."""
        # 6.148 Disable BLI effects
        self.disk_active = False  # Deactivate suction terms
        # 6.149 Solve baseline BL equations
        self.solve_boundary_layer()

    def compute_net_thrust(self):
        """
        Calculate net propulsive benefit
        Theory: Net thrust = Gross thrust - (Intake drag + Nacelle drag)
                Ref: Airbus A380 Aerodynamic Design and Fuel Efficiency
        """
        if not self.disk_active:
            return 0.0  # No thrust when propulsor off
        
        # 6.150 Calculate gross thrust
        # Theory: Actuator disk momentum theory
        T_gross = self.actuator_disk.calculate_thrust()  # [N]
        
        # 6.151 Calculate intake drag
        # Theory: Momentum deficit from flow contraction
        q_inlet = 0.5 * self.rho * self.nacelle.v_inlet**2  # Dynamic pressure [Pa]
        intake_drag = q_inlet * (self.nacelle.A_inlet - self.effective_A_disk)  # [N]
        
        # 6.152 Calculate nacelle drag components
        inlet_radius = math.sqrt(self.nacelle.A_inlet / math.pi)  # [m]
        wetted_area = 2 * math.pi * inlet_radius * self.nacelle.nac_length  # [m²]

        # 6.153 Empirical drag coefficients
        C_d_form = 0.01  # Form drag coefficient (streamlined shape)
        C_f = 0.004      # Skin friction coefficient (polished surface)
        
        # 6.154 Compute drag components
        form_drag = 0.5 * self.rho * self.free_stream_velocity**2 * self.nacelle.A_inlet * C_d_form  # [N]
        skin_friction = 0.5 * self.rho * self.free_stream_velocity**2 * wetted_area * C_f  # [N]
        
        nacelle_drag = form_drag + skin_friction  # Total nacelle drag [N]

        # 6.155 Calculate net thrust
        return T_gross - (intake_drag + nacelle_drag)  # [N]
    
     # Drag per unit span: D' = ρ U_e² θ  / https://www.youtube.com/watch?v=olqoce8ui5s&t=1447s re check this part
    
    # Revised drag reduction calculation
    def compute_drag_reduction(self):
        """
        Calculate net drag reduction from BLI effects
        Theory: Compares fuselage skin friction drag with/without propulsor
                and subtracts nacelle installation drag
        """
        # 6.156 Calculate baseline fuselage drag (no BLI)
        # Theory: D = ∫ρU²θ·2πR dx (Momentum thickness integration)
        D_NoBLI = np.trapz(
            self.rho * self.free_stream_velocity**2 * 
            self.results_without_propulsor["theta"] * 2 * np.pi * self.R,
            self.x
        )
        
        # 6.157 Calculate BLI-affected fuselage drag
        D_BLI = np.trapz(
            self.rho * self.free_stream_velocity**2 * 
            self.results_with_propulsor["theta"] * 2 * np.pi * self.R,
            self.x
        )

        # 6.158 Calculate nacelle installation drag
        C_d_nacelle = 0.008  # Form drag coefficient for nacelle
        Re = self.rho * self.free_stream_velocity * self.nacelle.nac_length / self.mu
        
        # 6.159 Turbulent skin friction coefficient (Prandtl-Schlichting)
        C_f = 0.455 / (np.log(0.06*Re))**2  
        
        # 6.160 Compute total nacelle drag components
        nacelle_drag = 0.5 * self.rho * self.free_stream_velocity**2 * (
            self.nacelle.A_inlet * C_d_nacelle +  # Form drag
            2 * np.pi * self.disk_radius * self.nacelle.nac_length * C_f  # Skin friction
        )

        return D_NoBLI - D_BLI - nacelle_drag

    def compute_PSC(self):  
        """
        Calculate Power Saving Coefficient (PSC)
        Theory: PSC = (D_baseline - D_BLI - D_nacelle)/D_baseline
                Represents net propulsive efficiency gain
        """
        Vinf = self.free_stream_velocity
        
        # 6.161 Calculate baseline drag
        D_NoBLI = np.trapz(
            self.rho * Vinf**2 * 
            self.results_without_propulsor["theta"] * 2 * np.pi * self.R,
            self.x
        )
        
        # 6.162 Calculate BLI-affected drag
        D_BLI = np.trapz(
            self.rho * Vinf**2 * 
            self.results_with_propulsor["theta"] * 2 * np.pi * self.R,
            self.x
        )

        # 6.163 Calculate nacelle drag components
        C_d_nacelle = 0.008  # Nacelle form drag coefficient
        Re = self.rho * Vinf * self.nacelle.nac_length / self.mu
        C_f = 0.455 / (np.log(0.06*Re))**2  # Turbulent skin friction
        
        # 6.164 Compute installation drag
        nacelle_drag = 0.5 * self.rho * Vinf**2 * (
            self.nacelle.A_inlet * C_d_nacelle + 
            2 * np.pi * self.disk_radius * self.nacelle.nac_length * C_f
        )

        # 6.165 Calculate PSC with numerical safety
        return (D_NoBLI - D_BLI - nacelle_drag) / (D_NoBLI + 1e-9)
#---------------------------------------# 
# Section 7 - Actuator Disk Model #
#---------------------------------------# 
class ActuatorDiskModel:
    """
    Models propulsion system performance using actuator disk theory
    Theory: Momentum exchange approach for propulsor analysis
            - Mass flow through effective disk area
            - Accounts for efficiency losses in components
            - Based on classical momentum theory (Rankine-Froude)
    """
    
    def __init__(self, rho, A_effective, v_inlet, v_disk, 
               eta_disk=None, eta_motor=None, eta_prop=None):
        # 7.1 Initialize fluid properties
        self.rho = rho  # Air density [kg/m³]
        self.A_effective = A_effective  # Effective disk area [m²]
        self.v_inlet = v_inlet  # Velocity at disk inlet [m/s]
        self.v_disk = v_disk  # Velocity at actuator disk [m/s]
        
        # 7.2 Set component efficiencies with defaults
        self.ηdisk = eta_disk or _EFFICIENCY_DEFAULTS["eta_disk"]  # Disk efficiency
        self.ηmotor = eta_motor or _EFFICIENCY_DEFAULTS["eta_motor"]  # Motor efficiency
        self.ηprop = eta_prop or _EFFICIENCY_DEFAULTS["eta_prop"]  # Propeller efficiency
        
        # 7.3 Calculate total system efficiency
        self.eta_total = self.ηdisk * self.ηmotor * self.ηprop
        
        # 7.4 Compute exhaust velocity using momentum theory
        # Theory: v_exit = 2v_disk - v_inlet (from conservation laws)
        self.v_exhaust = 2 * self.v_disk - self.v_inlet

    def calculate_mass_flow_rate(self):
        """Calculate mass flow rate (mdot)."""
        # 7.5 Apply continuity equation
        # Theory: mdot = ρ*A_effective*v_disk
        mdot = self.rho * self.A_effective * self.v_disk
        return mdot

    def calculate_thrust(self, mdot=None):
        """Calculate thrust (T)."""
        # 7.6 Use momentum theorem
        # Theory: T = mdot(v_exhaust - v_inlet)
        if mdot is None:
            mdot = self.calculate_mass_flow_rate()
        T = mdot * (self.v_exhaust - self.v_inlet)
        return T

    def calculate_power_disk(self, thrust=None):
        """Calculate power required at the disk (P_disk)."""
        # 7.7 Compute propulsive power
        # Theory: P_disk = T*v_disk (work done on fluid)
        if thrust is None:
            thrust = self.calculate_thrust()
        P_disk = thrust * self.v_disk * 1e-3  # Convert to kW
        return P_disk

    def calculate_total_power(self, P_disk=None):
        """Calculate total electrical power required (P_total) after efficiency losses."""
        # 7.8 Account for system inefficiencies
        # Theory: P_total = P_disk / η_total
        if P_disk is None:
            P_disk = self.calculate_power_disk()
        P_total = P_disk / self.eta_total
        return P_total

    def display_results(self):
        """Display results for mass flow rate, thrust, disk power, and total power."""
        # 7.9 Comprehensive performance report
        mdot = self.calculate_mass_flow_rate()
        T = self.calculate_thrust(mdot)
        P_disk = self.calculate_power_disk(T)
        P_total = self.calculate_total_power(P_disk)
        return mdot, T, P_disk, P_total, self.A_effective

 
#---------------------------------------# 
# Section 8 - Engine Visualization #
#---------------------------------------# 
class EngineVisualization:
    """
    Creates 2D visualizations of engine geometry and flow fields
    Theory: Combines aerodynamic performance data with geometric modeling
            using piecewise functions for nacelle shaping and velocity profiles
    """
    
    def __init__(self, A_inlet, A_disk, A_exhaust, v_inlet, v_disk, v_exhaust, 
                 nac_length, inlet_radius, disk_radius, exhaust_radius, app=None):
        # 8.1 Initialize core geometric parameters
        self.A_inlet = A_inlet  # Inlet capture area [m²]
        self.A_disk = A_disk    # Actuator disk area [m²]
        self.A_exhaust = A_exhaust  # Exhaust area [m²]
        self.v_inlet = v_inlet  # Inlet velocity [m/s]
        self.v_disk = v_disk    # Disk velocity [m/s]
        self.v_exhaust = v_exhaust  # Exhaust velocity [m/s]
        self.nac_length = nac_length  # Total nacelle length [m]
        self.inlet_radius = inlet_radius  # Inlet radius [m]
        self.disk_radius = disk_radius    # Disk radius [m]
        self.exhaust_radius = exhaust_radius  # Exhaust radius [m]
        self.app = app  # GUI application reference

        # 8.2 Define geometric segmentation
        self.extra_length = 2  # Visualization padding [m]
        self.l_intake = 1.5    # Intake section length [m]
        self.l_engine = 2.5    # Core engine length [m]
        self.l_exhaust = self.nac_length - (self.l_intake + self.l_engine)  # Exhaust length [m]
        self.disk_location = self.l_intake + 0.5  # Disk axial position [m]

    def calculate_geometry(self):
        """Calculate nacelle geometry and velocity distribution"""
        # 8.3 Create axial coordinate array
        self.x = np.linspace(-self.extra_length, self.nac_length + self.extra_length, 700)
        
        # 8.4 Define piecewise outer radius function
        # Theory: Parametric nacelle shaping for aerodynamic efficiency
        self.outer_radius = np.piecewise(
            self.x,
            [
                self.x < 0,  # Pre-inlet
                (self.x >= 0) & (self.x < self.l_intake),  # Intake
                (self.x >= self.l_intake) & (self.x < self.disk_location),  # Pre-disk
                (self.x >= self.disk_location) & (self.x < self.l_intake + self.l_engine),  # Engine core
                (self.x >= self.l_intake + self.l_engine) & (self.x <= self.nac_length),  # Exhaust
                self.x > self.nac_length  # Post-exhaust
            ],
            [
                lambda x: self.inlet_radius,  # Constant pre-inlet
                lambda x: self.inlet_radius + (self.disk_radius - self.inlet_radius) * (x / self.l_intake),  # Linear expansion
                lambda x: self.disk_radius,  # Constant disk area
                lambda x: self.disk_radius - (self.disk_radius - self.exhaust_radius) * ((x - self.disk_location) / (self.l_engine - (self.disk_location - self.l_intake))),  # Linear contraction
                lambda x: self.exhaust_radius,  # Constant exhaust
                lambda x: self.exhaust_radius  # Post-exhaust
            ]
        )

        # 8.5 Define piecewise velocity distribution
        # Theory: Simplified velocity profile for visualization
        self.velocities = np.piecewise(
            self.x,
            [
                self.x < 0,  # Freestream
                (self.x >= 0) & (self.x < self.l_intake),  # Intake deceleration
                (self.x >= self.l_intake) & (self.x < self.disk_location),  # Pre-disk flow
                (self.x >= self.disk_location) & (self.x < self.l_intake + self.l_engine),  # Disk acceleration
                (self.x >= self.l_intake + self.l_engine) & (self.x <= self.nac_length),  # Exhaust
                self.x > self.nac_length  # Post-exhaust
            ],
            [
                lambda x: self.v_inlet,    # Freestream velocity
                lambda x: self.v_inlet - 10,  # Intake deceleration
                lambda x: self.v_disk - 20,   # Pre-disk flow
                lambda x: self.v_disk + 10,   # Disk acceleration
                lambda x: self.v_exhaust,  # Exhaust velocity
                lambda x: self.v_exhaust   # Post-exhaust
            ]
        )

    def plot_velocity_field(self, canvas_frame):
        """Visualize velocity field around nacelle using streamlines"""
        # 8.6 Calculate geometry data
        self.calculate_geometry()
        
        # 8.7 Create computational grid
        x_grid = np.linspace(-5, self.nac_length + 5, 100)
        y_grid = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        # 8.8 Populate velocity field
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                x_p = X[j, i]
                y_p = Y[j, i]
                
                # 8.9 Determine nacelle boundary
                if x_p < 0 or x_p > self.nac_length:
                    nacelle_radius_at_x = 0
                else:
                    idx = np.argmin(np.abs(self.x - x_p))
                    nacelle_radius_at_x = self.outer_radius[idx]
                
                # 8.10 Mask internal nacelle region
                if np.abs(y_p) <= nacelle_radius_at_x:
                    U[j, i] = np.nan
                    V[j, i] = np.nan
                    continue
                
                # 8.11 Assign velocity components
                if x_p < self.l_intake:
                    U[j, i] = self.v_inlet    # Intake region
                elif x_p < self.l_intake + self.l_engine:
                    U[j, i] = self.v_disk     # Engine core
                else:
                    U[j, i] = self.v_exhaust  # Exhaust
                V[j, i] = 0  # Simplified 2D flow

        # 8.12 Create streamline plot
        fig, ax = plt.subplots(figsize=(10, 6))
        strm = ax.streamplot(X, Y, U, V, color=np.sqrt(U**2 + V**2), 
                           cmap='jet', density=1.5)
        
        # 8.13 Plot nacelle geometry
        ax.plot(self.x, self.outer_radius, 'k', linewidth=2)
        ax.plot(self.x, -self.outer_radius, 'k', linewidth=2)
        
        # 8.14 Configure plot elements
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Velocity Field Around the Nacelle")
        cbar = fig.colorbar(strm.lines, ax=ax)
        cbar.set_label("Velocity Magnitude (m/s)")
        ax.grid(True)

        # 8.15 Embed visualization in GUI
        for widget in canvas_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def plot_geometry(self, canvas_frame):
        """Visualize nacelle geometry with velocity color mapping"""
        # 8.16 Clear previous visualization
        for widget in canvas_frame.winfo_children():
            widget.destroy()

        self.calculate_geometry()

        # 8.17 Create adaptive figure size
        fig_width = max(16, self.nac_length / 2)
        fig_height = fig_width / 6
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # 8.18 Configure color mapping
        cmap = plt.cm.plasma
        norm = mcolors.Normalize(vmin=min(self.velocities), vmax=max(self.velocities))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # 8.19 Plot nacelle geometry
        ax.fill_between(self.x, -self.outer_radius, self.outer_radius, 
                      color="lightgray", alpha=0.5, label="Nacelle Geometry")
        main_line, = ax.plot(self.x, self.outer_radius, color="darkred", linewidth=2)
        ax.plot(self.x, -self.outer_radius, color="darkred", linewidth=2)

        # 8.20 Create velocity-colored segments
        patches = []
        for i in range(len(self.x) - 1):
            patch = ax.fill_between(
                [self.x[i], self.x[i + 1]],
                [-self.outer_radius[i], -self.outer_radius[i + 1]],
                [self.outer_radius[i], self.outer_radius[i + 1]],
                color=cmap(norm(self.velocities[i])),
                edgecolor="none",
                alpha=0.7
            )
            patches.append(patch)

        # 8.21 Add key engine markers
        fan_x = self.disk_location
        fan_line = ax.plot([fan_x, fan_x], [-self.disk_radius, self.disk_radius], 
                         color="black", linewidth=2, linestyle="--", 
                         label="Fan (Disk Location)")[0]
        
        exhaust_start = self.l_intake + self.l_engine
        exhaust_line = ax.plot([exhaust_start, exhaust_start], [-self.exhaust_radius, self.exhaust_radius], 
                             color="orange", linewidth=2, linestyle="--", 
                             label="Exhaust Boundary")[0]
        
        # 8.22 Add annotations
        text_elements = [
            ax.text(-self.extra_length/2, self.inlet_radius, 
                   f"Inlet\nArea: {self.A_inlet:.2f} m²", ha="center"),
            ax.text(fan_x, self.disk_radius, 
                   f"Fan (Disk)\nArea: {self.A_disk:.2f} m²", ha="center"),
            ax.text((exhaust_start + self.nac_length)/2, self.exhaust_radius,
                   f"Exhaust\nArea: {self.A_exhaust:.2f} m²", ha="center")
        ]

        # 8.23 Add colorbar and final touches
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Velocity (m/s)', fontsize=12)
        ax.set_title("2D Engine Geometry with Velocity Representation", fontsize=16)
        ax.set_xlabel("Length (m)", fontsize=12)
        ax.set_ylabel("Radius (m)", fontsize=12)
        ax.legend()
        ax.grid(True)

        # 8.24 Implement hover functionality
        def on_hover(sel):
            """Interactive data display on hover"""
            try:
                if sel.artist in patches:
                    idx = patches.index(sel.artist)
                    x_val = (self.x[idx] + self.x[idx+1])/2
                    y_val = (self.outer_radius[idx] + self.outer_radius[idx+1])/2
                    sel.annotation.set_text(
                        f"Position: {x_val:.2f}m\n"
                        f"Velocity: {self.velocities[idx]:.2f} m/s\n"
                        f"Radius: {y_val:.2f}m"
                    )
                elif sel.artist == fan_line:
                    sel.annotation.set_text(f"Fan Location\nx: {fan_x:.2f}m\nRadius: {self.disk_radius:.2f}m")
                elif sel.artist == exhaust_line:
                    sel.annotation.set_text(f"Exhaust Start\nx: {exhaust_start:.2f}m\nRadius: {self.exhaust_radius:.2f}m")
            except Exception as e:
                sel.annotation.set_text("Data not available")

        cursor = mplcursors.cursor(patches + [fan_line, exhaust_line], hover=True)
        cursor.connect("add", on_hover)

        # 8.25 Embed in GUI with navigation
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

#---------------------------------------# 
# Section 9 - GUI Application #
#---------------------------------------# 
class BoundaryLayerIngestion:
    """
    Main GUI application for Boundary Layer Ingestion analysis
    Theory: Integrates aerodynamic models with interactive visualization
            - Processes flight conditions through multiple aerodynamic models
            - Provides comparative visualization of BLI effects
            - Implements interactive data exploration
    """
    
    def __init__(self, root):
        # 9.1 Initialize main application
        self.root = root
        self._configure_root()
        self._initialize_variables()
        self._create_main_frame()
        self._create_io_section()
        self._create_notebook()

    # === GUI Setup Methods ===
    def _configure_root(self):
        """Configure main window properties"""
        # 9.2 Window configuration
        self.root.title("Boundary Layer Ingestion Concept Data Screen")
        self.root.state('zoomed')  # Maximize window

    def _initialize_variables(self):
        """Initialize core application variables"""
        # 9.3 State variables
        self.FL = None  # Flight level [feet]
        self.Mach = None  # Mach number
        self.A_inlet = None  # Inlet area [m²]
        self.selected_data = {'x': None, 'y': None}  # Cursor position storage

    def _create_main_frame(self):
        """Create main container frame"""
        # 9.4 Main frame initialization
        self.main_frame = tk.Frame(self.root, padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def _create_io_section(self):
        """Create input/output panel"""
        # 9.5 IO panel structure
        self.io_frame = tk.Frame(self.main_frame, padx=10, pady=10)
        self.io_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self._create_input_frame()
        self._create_output_frame()
        self._create_cursor_label()

    def _create_input_frame(self):
        """Create input controls"""
        # 9.6 Input widgets
        input_frame = tk.LabelFrame(self.io_frame, text="Inputs", padx=10, pady=10)
        input_frame.pack(fill=tk.X, pady=5)

        # Flight level input
        tk.Label(input_frame, text="Enter the Flight Altitude (Feet):").pack(anchor='w')
        self.fl_entry = tk.Entry(input_frame)
        self.fl_entry.pack(fill=tk.X, pady=2)

        # Mach number input
        tk.Label(input_frame, text="Enter Mach Number:").pack(anchor='w')
        self.mach_entry = tk.Entry(input_frame)
        self.mach_entry.pack(fill=tk.X, pady=2)

        # Inlet area input
        tk.Label(input_frame, text="Enter Inlet Area (m²):").pack(anchor='w')
        self.area_entry = tk.Entry(input_frame)
        self.area_entry.pack(fill=tk.X, pady=2)

        # Visualization trigger
        submit_btn = tk.Button(input_frame, text="Visualize", command=self.visualize)
        submit_btn.pack(pady=10)

    def _create_output_frame(self):
        """Create output text panel"""
        # 9.7 Output console
        output_frame = tk.LabelFrame(self.io_frame, text="Outputs", padx=10, pady=10)
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, width=40, height=30)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def _create_cursor_label(self):
        """Create cursor tracking display"""
        # 9.8 Interactive feedback
        self.cursor_label = tk.Label(self.io_frame, text="Cursor Data: x=None, y=None", font=("Arial", 10))
        self.cursor_label.pack(pady=5)

    def _create_notebook(self):
        """Create visualization tabs"""
        # 9.9 Notebook (tabbed interface) setup
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 9.10 Tab initialization
        self.engine_tab = tk.Frame(self.notebook)
        self.velocity_tab = tk.Frame(self.notebook)
        self.fuselage_tab = tk.Frame(self.notebook)
        self.pressure_tab = tk.Frame(self.notebook)
        self.boundary_layer_tab = tk.Frame(self.notebook)

        # 9.11 Tab registration
        self.notebook.add(self.engine_tab, text="Engine Geometry")
        self.notebook.add(self.velocity_tab, text="Velocity Field")
        self.notebook.add(self.fuselage_tab, text="Source Strength Geometry")
        self.notebook.add(self.pressure_tab, text="Pressure Distribution")
        self.notebook.add(self.boundary_layer_tab, text="Boundary Layer")

    # === Utility Methods ===
    def update_cursor_data(self, x, y):
        """Update cursor position display"""
        # 9.12 Interactive data tracking
        self.selected_data['x'] = x
        self.selected_data['y'] = y
        self.cursor_label.config(text=f"Cursor Data: x={x:.2f}, y={y:.2f}")

    def _embed_figure_in_tab(self, fig, tab, cursor_artists=None):
        """Embed matplotlib figure in GUI tab"""
        # 9.13 Visualization integration
        # Clear existing content
        for widget in tab.winfo_children():
            widget.destroy()

        # 9.14 Canvas creation
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # 9.15 Toolbar integration
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

        # 9.16 Interactive cursor setup
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
        """Main analysis pipeline"""
        # 9.17 Clear previous results
        self.output_text.delete("1.0", tk.END)
        self._read_inputs()

        # 9.18 Execute analysis sequence
        flight_conditions, T, p, rho, v_inlet, v_freestream = self._process_flight_conditions()
        self.nacelle, results_nacelle, A_disk, v_disk, v_exhaust = self._process_nacelle_geometry(rho, p, v_inlet)
        mdot, T_thrust, P_disk, P_total, _ = self._process_actuator_disk_model(rho, A_disk, v_inlet, v_disk)
        Dzero, Cdzero, mu = self._process_drag_bli_engine(flight_conditions, self.nacelle)
        total_motor_mass = self._process_engine_mass_estimation(v_inlet, rho, p, self.nacelle)

        # 9.19 Generate visualizations
        self._plot_visualizations(results_nacelle, v_inlet, v_disk, v_exhaust, v_freestream, rho, mu)

    def _read_inputs(self):
        """Validate and convert user inputs"""
        # 9.20 Input processing
        self.FL = float(self.fl_entry.get())
        self.Mach = float(self.mach_entry.get())
        self.A_inlet = float(self.area_entry.get())

    # === Processing Methods ===
    def _process_flight_conditions(self):
        """Process atmospheric and flow properties"""
        # 9.21 Flight condition analysis
        flight_conditions = FlightConditions()
        T, p, rho = flight_conditions.calculate_atmospheric_properties(self.FL)
        v_inlet, v_freestream = flight_conditions.calculate_free_stream_velocity(self.Mach, self.FL)

        # 9.22 Output formatting
        self.output_text.insert(tk.END, 'Chapter 1: Flight Conditions\n')
        self.output_text.insert(tk.END, f"Temperature: {T:.2f} K\n")
        self.output_text.insert(tk.END, f"Pressure: {p:.2f} Pa\n")
        self.output_text.insert(tk.END, f"Density: {rho:.6f} kg/m³\n")
        self.output_text.insert(tk.END, f"Free-stream velocity: {v_freestream:.2f} m/s\n")
        self.output_text.insert(tk.END, '-----------------------------\n')

        return flight_conditions, T, p, rho, v_inlet, v_freestream

    def _process_nacelle_geometry(self, rho, p, v_inlet):
        """Analyze nacelle geometry parameters"""
        # 9.23 Nacelle performance calculation
        nacelle = NacelleParameters(v_inlet, self.A_inlet)
        results_nacelle = nacelle.variable_parameters(rho, p)
        A_disk = results_nacelle[1]
        v_disk = results_nacelle[5]
        v_exhaust = results_nacelle[6]

        # 9.24 Geometry output
        self.output_text.insert(tk.END, 'Chapter 2: Nacelle Geometry\n')
        self.output_text.insert(tk.END, f"Inlet Radius: {results_nacelle[3]:.2f} m\n")
        self.output_text.insert(tk.END, f"Inlet Area: {results_nacelle[0]:.2f} m²\n")
        self.output_text.insert(tk.END, f"Disk Velocity: {v_disk:.2f} m/s\n")
        self.output_text.insert(tk.END, f"Exhaust Velocity: {v_exhaust:.2f} m/s\n")
        self.output_text.insert(tk.END, '--------------------------------\n')

        return nacelle, results_nacelle, A_disk, v_disk, v_exhaust

    def _process_actuator_disk_model(self, rho, A_disk, v_inlet, v_disk):
        """Calculate propulsive performance"""
        # 9.25 Actuator disk analysis
        actuator_model = ActuatorDiskModel(rho, A_disk, v_inlet, v_disk)
        mdot, T_thrust, P_disk, P_total, _ = actuator_model.display_results()

        # 9.26 Propulsion output
        self.output_text.insert(tk.END, 'Chapter 3: Basic Actuator Disk Model\n')
        self.output_text.insert(tk.END, f"Mass flow rate: {mdot:.2f} kg/s\n")
        self.output_text.insert(tk.END, f"Thrust: {T_thrust:.2f} N\n")
        self.output_text.insert(tk.END, f"Disk Power: {P_disk:.2f} kW\n")
        self.output_text.insert(tk.END, f"Total Power: {P_total:.2f} kW\n")
        self.output_text.insert(tk.END, '--------------------------------------\n')

        return mdot, T_thrust, P_disk, P_total, None

    def _process_drag_bli_engine(self, flight_conditions, nacelle):
        """Calculate BLI-induced drag components"""
        # 9.27 Drag analysis
        bli_engine = DragbyBLIEngine(flight_conditions, nacelle, self.FL, self.Mach)
        Dzero, Cdzero, mu = bli_engine.calculate_zero_lift_drag()

        self.output_text.insert(tk.END, 'Chapter 4: Drag Generated by BLI Engine\n')
        self.output_text.insert(tk.END, f"Zero Lift Drag: {Dzero:.2f} N\n")
        self.output_text.insert(tk.END, '---------------------------------------\n')

        return Dzero, Cdzero, mu

    def _process_engine_mass_estimation(self, v_inlet, rho, p, nacelle):
        """Estimate propulsion system mass"""
        # 9.28 Mass estimation
        nacelle.variable_parameters(rho, p)  # Ensure parameters are updated
        actuator_disk = ActuatorDiskModel(
            rho=rho,
            A_effective=nacelle.effective_A_disk,
            v_inlet=v_inlet,
            v_disk=nacelle.v_disk,
            eta_disk=nacelle.ηdisk,
            eta_motor=nacelle.ηmotor,
            eta_prop=nacelle.ηprop
        )
        engine_mass = EngineMassEstimation(actuator_disk)
        total_motor_mass = engine_mass.calculate_total_motor_mass()

        self.output_text.insert(tk.END, 'Chapter 5: Engine Mass Estimation\n')
        self.output_text.insert(tk.END, f"Total Mass: {total_motor_mass:.2f} kg\n")
        self.output_text.insert(tk.END, '--------------------------------------\n')

        return total_motor_mass

    # === Plotting Methods ===       
    def _plot_visualizations(self, results_nacelle, v_inlet, v_disk, v_exhaust, v_freestream, rho, mu):
        """Coordinate all visualization tasks"""
        # 9.29 Visualization pipeline
        self._plot_engine_geometry(results_nacelle, v_inlet, v_disk, v_exhaust)
        A_inlet = results_nacelle[0]    
        delta_p = results_nacelle[10]
        p = results_nacelle[-1]
        
        # 9.30 Initialize fuselage flow model
        self.fuselage = Flow_around_fuselage(v_freestream, self.Mach, rho, mu, delta_p, A_inlet,p)
        self.fuselage.solve_boundary_layer()
        self.fuselage.run_simulation()

        # 9.31 Generate aerodynamic visualizations
        self._plot_source_field()
        self._plot_velocity_field()
        self._plot_pressure_distribution()
        self._plot_boundary_layer_thickness()

    def _plot_engine_geometry(self, results_nacelle, v_inlet, v_disk, v_exhaust):
        """Visualize nacelle geometry"""
        # 9.32 Engine geometry visualization
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
        visualization.plot_geometry(self.engine_tab)
            
    def _plot_source_field(self):
        """Visualize source strength distribution"""
        # 9.33 Source strength visualization
        self.fuselage.plot_source_strength(self.fuselage_tab)

    def _plot_velocity_field(self):
        """Visualize velocity field"""
        # 9.34 Velocity field visualization
        self.fuselage.plot_velocity_streamlines(self.velocity_tab)

    def _plot_pressure_distribution(self):
        """Visualize pressure distribution"""
        # 9.35 Pressure visualization
        self.fuselage.plot_pressure_distribution(self.pressure_tab)

    def _plot_boundary_layer_thickness(self):
        """Visualize boundary layer development"""
        # 9.36 Boundary layer visualization
        self.fuselage.plot_boundary_layer_thickness(self.boundary_layer_tab)

# === Main Program ===
if __name__ == "__main__":
    root = tk.Tk()
    app = BoundaryLayerIngestion(root)
    root.mainloop()