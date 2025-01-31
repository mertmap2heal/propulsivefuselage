import numpy as np
import matplotlib.pyplot as plt

class FuselageGeometry:
    def __init__(self, FlightConditions=None):
        # Airbus A320-like parameters
        self.fuselage_length = 37.57
        self.fuselage_radius = 2.0
        self.nose_length = 3.0
        self.tail_length = 10.0
        self.free_stream_velocity = 230  # m/s

        # Discretization (increased resolution)
        N = 1000
        self.x = np.linspace(0, self.fuselage_length, N)
        self.y_upper = np.zeros(N)
        self.y_lower = np.zeros(N)

        # Geometry construction
        for i, xi in enumerate(self.x):
            if xi <= self.nose_length:
                # Parabolic nose section
                y = self.fuselage_radius * (1 - ((xi - self.nose_length)/self.nose_length)**2)
                self.y_upper[i] = y
                self.y_lower[i] = -y
            elif xi >= self.fuselage_length - self.tail_length:
                # Parabolic tail section
                x_tail = xi - (self.fuselage_length - self.tail_length)
                y = self.fuselage_radius * (1 - (x_tail/self.tail_length)**2)
                self.y_upper[i] = y
                self.y_lower[i] = -y
            else:
                # Constant cylindrical section
                self.y_upper[i] = self.fuselage_radius
                self.y_lower[i] = -self.fuselage_radius

    def compute_Q_analytical(self):
        """Analytical derivative of R² to avoid numerical artifacts"""
        dr2_dx = np.zeros_like(self.x)
        
        for i, xi in enumerate(self.x):
            if xi <= self.nose_length:
                # Nose section derivative
                term = (xi - self.nose_length)/self.nose_length
                dr2_dx[i] = 2 * self.fuselage_radius**2 * (1 - term**2) * (-2 * term / self.nose_length)
            elif xi >= self.fuselage_length - self.tail_length:
                # Tail section derivative
                x_tail = xi - (self.fuselage_length - self.tail_length)
                term = x_tail/self.tail_length
                dr2_dx[i] = 2 * self.fuselage_radius**2 * (1 - term**2) * (-2 * term / self.tail_length)
            else:
                # Cylindrical section: explicit zero
                dr2_dx[i] = 0.0
                
        return self.free_stream_velocity * np.pi * dr2_dx

    def plot_geometry(self):
        """Plot fuselage with analytically computed source strength"""
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Plot geometry
        ax1.plot(self.x, self.y_upper, 'b', label='Upper Surface')
        ax1.plot(self.x, self.y_lower, 'b', label='Lower Surface')
        ax1.fill_between(self.x, self.y_upper, self.y_lower, color='lightblue', alpha=0.3)
        ax1.set_xlabel('Axial Position [m]')
        ax1.set_ylabel('Vertical Position [m]', color='b')
        ax1.set_title('Fuselage Geometry with Analytical Source Strength Q(x)')
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax1.grid(True)
        ax1.legend(loc='upper right')
        ax1.set_ylim(-self.fuselage_radius*1.5, self.fuselage_radius*1.5)

        # Plot analytical source strength
        ax2 = ax1.twinx()
        Q_analytical = self.compute_Q_analytical()
        ax2.plot(self.x, Q_analytical, 'r', linewidth=1.5, label='Source Strength $Q(x)$')
        ax2.set_ylabel('Source Strength $Q(x)$ [m²/s]', color='r')
        ax2.legend(loc='lower right')
        ax2.set_ylim(-1.2*np.max(Q_analytical), 1.2*np.max(Q_analytical))

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    fuselage = FuselageGeometry()
    fuselage.plot_geometry()