 

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
    def __init__(self, flight_conditions, nacelle_params, FL, v_freestream, Mach, rho, mu, delta_p, A_inlet, p,
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
        self.flight_conditions = flight_conditions
        self.nacelle_params = nacelle_params
        self.drag_bli_engine = DragbyBLIEngine(flight_conditions, nacelle_params, FL, Mach)


        
        # 6.3 Calculate BLI parameters
        # Theory: Empirical scaling for maximum boundary layer thickness (Kaiser et al.)
        self.delta_99_max = 0.18 * self.fuselage_length * (A_inlet/12.0)**0.25
        
        # 6.4 Propulsor disk geometry
        # Theory: 10% area contraction for flow acceleration
        A_disk = A_inlet * 0.9  # Effective disk area [m²]
        self.disk_radius = math.sqrt(A_disk / math.pi)  # Disk radius [m]
        disk_diameter = 2 * self.disk_radius  # Propulsor diameter [m]
        self.inlet_radius = math.sqrt(A_inlet / math.pi)  # Inlet radius [m]
        
        # 6.5 Suction parameters initialization
        # Theory: Base suction strength proportional to capture area
        self.suction_strength = 0.04 * A_inlet  # Initial suction parameter [m²/s]
 

        # 6.6 Mass flow ratio calculation
      

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
        R_fus = self.y_upper[idx]
        self.effective_A_disk = np.pi * (self.disk_radius**2 - R_fus**2)
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
        R_fus = self.y_upper[idx]  # Fuselage radius at propulsor [m]
        R_inlet = self.inlet_radius   # Propulsor outer radius [m]
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
                       * (1 - (r - R_fus)/delta_99)**(1/7) 
                        )
            else:  # Laminar flow
                # Theory: Quadratic profile (Kaiser Eq.10b)
                delta_99_clamped = max(delta_99, delta_99_min)
                term = (r - R_fus)/delta_99_clamped
                return self.free_stream_velocity * (2*term - term**2)

        # 6.126 Sample velocity across disk radius
        if (R_fus + delta_99) >= R_inlet:  # Entire disk in BL
            r = np.linspace(R_fus, R_inlet, 100)  # Radial points [m]
            velocities = [boundary_layer_velocity(x_disk, y) for y in r]
        else:  # Combined BL + freestream
            # 6.127 Split into BL and freestream regions
            r_bl = np.linspace(R_fus, R_fus + delta_99, 50)
            v_bl = [boundary_layer_velocity(x_disk, y) for y in r_bl]
            r_fs = np.linspace(R_fus + delta_99 + 1e-6, R_inlet, 50)
            v_fs = [self.free_stream_velocity] * len(r_fs)
            r = np.concatenate([r_bl, r_fs])
            velocities = np.concatenate([v_bl, v_fs])

        # 6.128 Compute mass-averaged velocity (Kaiser Eq.11)
        # Theory: V_avg = (2∫u(r)·r dr) / (R_inlet² - R_fus²)
        numerator = np.trapz([u*r_i for u, r_i in zip(velocities, r)], r)
        denominator = 0.5 * (R_inlet**2 - R_fus**2)
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
 
        # Compute Reynolds number for nacelle
        Re = self.rho * self.free_stream_velocity * self.nacelle.nac_length / self.mu

        Dzero, Cdzero, mu = self.drag_bli_engine.calculate_zero_lift_drag()
        nacelle_drag = Dzero  # Replace existing nacelle_drag calculation

        return D_NoBLI - D_BLI - nacelle_drag

    def compute_PSC(self):  
        """
        Calculate Power Saving Coefficient (PSC)
        """
        Vinf = self.free_stream_velocity

        # Calculate baseline and BLI-affected drag (existing code)
        D_NoBLI = np.trapz(
            self.rho * Vinf**2 * 
            self.results_without_propulsor["theta"] * 2 * np.pi * self.R,
            self.x
        )
        D_BLI = np.trapz(
            self.rho * Vinf**2 * 
            self.results_with_propulsor["theta"] * 2 * np.pi * self.R,
            self.x
        )

        # Get Dzero from DragbyBLIEngine
        Dzero, Cdzero, mu = self.drag_bli_engine.calculate_zero_lift_drag()
        nacelle_drag = Dzero  # Replace existing nacelle_drag calculation

        # Calculate PSC
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

 
 