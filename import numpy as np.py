import numpy as np
import matplotlib.pyplot as plt

class FuselageGeometry:
    def __init__(self,):
        #Based on Airbus A320 
        self.fuselage_length=37.57 
        self.fuselage_radius=2
        self.nose_length=3
        self.tail_length=10

        N = 300
        self.x = np.linspace(0, self.fuselage_length, N)
        self.y_upper = np.zeros(N)  # Upper surface  
        self.y_lower = np.zeros(N)  # Lower surface  

        for i in range(N):
            if self.x[i] <= self.nose_length:
                # Nose 
                self.y_upper[i] = self.fuselage_radius * (1 - ((self.x[i] - self.nose_length) / self.nose_length) ** 2)
                self.y_lower[i] = -self.y_upper[i]
            elif self.x[i] > self.fuselage_length - self.tail_length:
                # Tail 
                self.y_upper[i] = self.fuselage_radius * (1 - ((self.x[i] - (self.fuselage_length - self.tail_length)) / self.tail_length) ** 2)
                self.y_lower[i] = -self.y_upper[i]
            else:
                # Cylindrical section
                self.y_upper[i] = self.fuselage_radius
                self.y_lower[i] = -self.fuselage_radius


    def plot_geometry(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.x, self.y_upper, 'b', label='Upper Surface')
        plt.plot(self.x, self.y_lower, 'b', label='Lower Surface')
        plt.fill_between(self.x, self.y_upper, self.y_lower, color='lightblue', alpha=0.3)
        
        plt.xlabel('Axial Position [m]')
        plt.ylabel('Vertical Position [m]')
        plt.title('Fuselage Geometry from side view')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Centerline
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  
        plt.show()


fuselage = FuselageGeometry()
fuselage.plot_geometry()
