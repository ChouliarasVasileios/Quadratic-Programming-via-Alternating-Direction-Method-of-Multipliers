# Importing Modules
import numpy as np # Linear Algebra
from numpy import ndarray # Matrix Enumeration
import matplotlib.pyplot as plt # Visuals
from matplotlib.patches import Circle # Visuals
#

class HoperRobot:
    """ It implement the forward and inverse dynamics of the HoperRobot system
        and also contains method for discretization of this system
    """

    def __init__(self) -> None:
        self.name = "Hoper_Robot" # The Name of the Robot
        self.sys_dim_space = 2 # The DOFs of the Robot, It can move in 2D space
        self.base_position : ndarray = np.array([[0.,0.]]).T # (x,y) # The start position of the f force 
        return None


    # x = [x,y,x_dot,y_dot].T -> State 
    # u = [f,tau].T -> Controls signals 
    # m = 5kg -> The mass of the Hoper
    # g = 10 m/s^2 -> The gravity acceleration
    def dynamics(self,x : ndarray, u : ndarray, m : int =5, g : float = 10) -> ndarray:
        """Implementing the continues forward dynamics of the HoperRobot System"""
        
        # rx and ry is used to calculate the projected f and T forces in x and y axes
        rx = x[0,0]/np.sqrt((x[0,0])**2 + (x[1,0])**2) # sin(theta)

        ry = x[1,0]/np.sqrt((x[0,0])**2 + (x[1,0])**2) # cos(theta)

        A = (rx*u[0,0]-ry*u[1,0])/m # The acceleration in x axes

        B = ((ry*u[0,0]+rx*u[1,0])/m) - g # The acceleration in y axes

        x_dot_state = np.array([[x[2,0],x[3,0],A,B]]).T
        return x_dot_state
    
    def inverse_dynamics(self,x:ndarray ,acc :ndarray,m : int = 5, g: float = 10):
        """Implementing the continues inverse dynamics of the HoperRobot System"""
        rx = x[0,0]/np.sqrt((x[0,0])**2 + (x[1,0])**2)

        ry = x[1,0]/np.sqrt((x[0,0])**2 + (x[1,0])**2)
        
        # Solving the system with respect to u
        XDT = np.array([[rx/m,-ry/m],[ry/m,rx/m]]) # [acc_x,acc_y].T = XDT @ [f,T].T - [0,g].T 
        
        acc = acc + np.array([[0,g]]).T
        
        u = np.linalg.solve(XDT,acc)
        #
        
        return u


    def euler_discrete(self, x : ndarray, u : ndarray, dt :float = 0.01) -> ndarray:
        return x + self.dynamics(x,u)*dt

    def semi_implicit_euler_discrete(self, x : ndarray, u : ndarray, dt :float = 0.01) -> ndarray:
        
        # ================================
        # Remind that x is the state and
        # x = [x,y,x_dot,y_dot].T
        # ================================
        x_new = np.copy(x)
        # x_dot of the State
        # x_dot = [vx,vy,ax,ay].T
        # Which v stand for velocity and a for acceleration
        x_dot = self.dynamics(x_new,u)
        # ================================

        # For the x coordinate
        x_new[2,0] = x[2,0] + x_dot[0,0] * dt

        x_new[0,0] = x[0,0] + x_new[2,0] * dt

        # ================================

        # For the y coordinate
        x_new[3,0] = x[3,0] + x_dot[1,0] * dt

        x_new[1,0] = x[1,0] + x_new[3,0] * dt

        return x_new

    def runge_kutta4_discrete(self, x : ndarray, u : ndarray, dt :float = 0.01) -> ndarray:
        f1 = self.dynamics(x, u)
        f2 = self.dynamics(x + f1 *  (dt / 2), u)
        f3 = self.dynamics(x + f2 * (dt / 2), u)
        f4 = self.dynamics(x + f3 * dt, u)
        return x + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4) 

    def midpoint_discrete(self, x : ndarray, u : ndarray, dt :float = 0.01) -> ndarray:

        xm = x + self.dynamics(x,u) * (dt / 2)

        x = x + self.dynamics(xm,u) * dt

        return x


class PD_Controller:
    
    def __init__(self,kp : ndarray,kd : ndarray):
        self.kp = kp # A 2x4 matrix cause error is 4x1 matrix
        self.kd = kd
        
    def pd(self,error,pre_error,step):
        if isinstance(self.kp,int) and isinstance(self.kd,int):
            return self.kp*error+self.kd*((error-pre_error)/step)
        else:  
            return self.kp@error+self.kd@((error-pre_error)/step)
    def __str__(self):
        return f"PD Controller kp = {self.kp} kd = {self.kd}"
    

class Visualization:

    def __init__(self, base_coord : ndarray, ball_size : int, step = 0.01) -> None:
        # To extend the visual window
        self._offset_up = 8
        self._offset_down = -8
        
        self.step = step # The simulation step == dt, discrete step ZOH
        self.fig , self.ax = plt.subplots()
        # Paint the starting position of the system before x_init implies
        self.sc = self.ax.scatter(base_coord[0,0], base_coord[1,0],s=ball_size,c = "black",marker="o",linewidths=1,edgecolors="black")
        self.ax.set_xlim((self._offset_down,self._offset_up))
        self.ax.set_ylim((self._offset_down,self._offset_up))
        plt.ion()  # Turn on interactive mode
        plt.grid()
        plt.draw()
        plt.pause(self.step)


    def update(self,x : float, y : float):
        """ Repainting the canvas of the visual"""
        
        # Checking if the system is out of the limits to extend the view
        if abs(x) >= self._offset_up or abs(y) >= self._offset_up:
            self._offset_up += 30
            self._offset_down -= 30
        
        # Extending the limits if needed 
        self.ax.set_xlim((self._offset_down,self._offset_up))
        self.ax.set_ylim((self._offset_down,self._offset_up))
        # Set green color to HoperRobot
        self.sc.set_color("lightgreen")
        # Paint the new position of the HoperRobot
        self.sc.set_offsets(np.column_stack((x,y)))
        # For the tail trace
        self.ax.scatter(x,y,12,"black","o",alpha=0.35)
        plt.draw()
        plt.pause(self.step)

def on_close(event):
    """A close event handler to close the terminal when visual is terminated"""
    print(f"User_Interacted ended on on_close, event_triggered: {event}")
    exit(1)

def main() -> int:
    
    # Initializing the starting state
    x_init = np.array([[0.,1.,0.,0.]]).T
    u = np.array([[0.,0.]]).T
    # The discrete and simulation step
    step = 0.01
    # Create an instance of HoperRobot
    hr = HoperRobot()
    
    # PD Gains
    # Gains for position where x < y , x < 0 and y > 0
    kp = np.array([[296,385,200,200],[65,177,35,100]])
    kd = np.array([[41.5,25,9.2,9.5],[4.6,4.6,1.4,3.4]])    

    # Other combinations
    # kd = np.array([[28,28,28,28],[5,20,5,20]])
    
    # kp = np.array([[800,800,800,800],[100,400,100,400]])
    # kd = np.array([[45,45,45,45],[20,40,20,40]])
    #
    
    # Creating a PD controller instance
    pd = PD_Controller(kp=kp,kd=kd)
    
    # Creating a visual instance
    visual = Visualization(base_coord=hr.base_position,ball_size=150,step=step)
    # Attach the event handler of the closing window
    visual.fig.canvas.mpl_connect('close_event',on_close)
    
    #  Initializing the target position
    # x_target = np.array([[-25,55,0,0]]).T
    # x_target = np.array([[-8,10,0,0]]).T
    # x_target = np.array([[-2,2,0,0]]).T
    # x_target = np.array([[-3,5,0,0]]).T
    x_target = np.array([[-10,30,0,0]]).T
    
    
    # Paint the target position in simulation
    visual.ax.scatter(x_target[0],x_target[1],None,"r")
    # Paint the accepted range of the error in simulation
    accepted_range = Circle((x_target[0],x_target[1]),radius = 0.1,fill = False,edgecolor = "red")
    visual.ax.add_patch(accepted_range)
    visual.ax.set_aspect("equal") # Make it look like a Circle
    plt.draw()
    plt.pause(0.8) # Pause the simulation 0.8 sec to see the initial position of the HoperRobot and 
    # the target position. The initial position is that is shown is the base position before starting position.
    
    it = 0 # count the iterations
    
    # PD controller variables
    error = np.copy(x_target-x_init)
    pre_error = np.array([[0],[0]]).T
    
    x = np.copy(x_init)
    
    # The position error, Target position - Current Position
    position_error_deltaX = error[:2]
    while np.linalg.norm(position_error_deltaX) >= 1e-1 :
        
        print("error:",np.linalg.norm(position_error_deltaX),"\t\nx:\n",x,"\n\n")
        
        # The signal that PD created
        control_signal = pd.pd(error=error,pre_error=pre_error,step=step)
        
        u = np.copy(control_signal)

        # Discrete to get the next state
        # x =  hr.runge_kutta4_discrete(x,u,dt=step)
        # visual.fig.suptitle(f"Runge_Kutta 4rd step = {step}")
        
        x = hr.euler_discrete(x,u,dt = step)
        visual.fig.suptitle(f"Euler step = {step}\nTarget = {x_target[:2].T}")
        
        # x = hr.midpoint_discrete(x,u,dt = step)
        # visual.fig.suptitle(f"Midpoint step = {step}")
        
        # x = hr.semi_implicit_euler_discrete(x,u,dt = step)
        # visual.fig.suptitle(f"Semi_Implicit_Euler step = {step}")
        
        # Update the visuals in simulation
        visual.update(x[0],x[1])
        
        pre_error = np.copy(error)
        error = np.copy(x_target - x)
        
        position_error_deltaX = error[:2]
        
        if it>50000:
            break
        it += 1
    
    print(f"Iterations : {it}\nThe Final position error :\n{position_error_deltaX}\n Last position:\n{x}")
    plt.ioff()
    plt.draw()
    plt.show()    

    return 0

if __name__ == "__main__":
    print("Main ended:",main())
