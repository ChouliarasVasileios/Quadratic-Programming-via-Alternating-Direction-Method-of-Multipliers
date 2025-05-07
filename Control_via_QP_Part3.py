import numpy as np
import matplotlib.pyplot as plt
import proxsuite
import Modeling_Part1 as mp1
import QP_ADMM_Part2 as qp2


# The trajectory generator for position
def q(t,A = 1,f = 5,phi = np.pi/2):
    omega = 2*np.pi*f
    return A*np.sin(omega*t + phi) 

# The trajectory generator for velocity with respect to the position above
def q_dot(t,A = 1,f = 5,phi = np.pi/2):
    omega = 2*np.pi*f
    return A*omega*np.cos(omega*t + phi)

# The trajectory generator for acceleration with respect to the velocity above
def q_ddot(t,A = 1,f = 5,phi = np.pi/2):
    omega = 2*np.pi*f
    return -A*omega*omega*np.sin(omega*t + phi)

# It is limiting the controls u 
def limit_controls(u):
    upper_bound = 200
    lower_bound = -200
    row_dim,col_dim = np.shape(u)

    for row in range(row_dim):
        for col in range(col_dim):
            
            if u[row,col] > upper_bound:
                u[row,col] = upper_bound
            
            if u[row,col] < lower_bound:
                u[row,col] = upper_bound
    return u

# A function that update the QP matrixes 
def update_QP_Matrixes(Mass_Matrix,x_state,x_ddot,y_ddot):    
    
    q1 = np.array([[-2*(x_ddot),-2*(y_ddot),0,0]]).T
    
    rx = x_state[0,0]/np.sqrt((x_state[0,0])**2 + (x_state[1,0])**2)
    ry = x_state[1,0]/np.sqrt((x_state[0,0])**2 + (x_state[1,0])**2)
    
    # The S matrix here is not the identity In, where n is the joints
    S = -1*np.array([[rx,-ry],[ry,rx]])
    
    # The equality constrain matrix Ax = b
    A = np.hstack((Mass_Matrix,S))
        
    return q1,A

def main():
    # Initial State
    x_init = np.array([[0,1,0,0]]).T
    u_init = np.array([[0.,0.]]).T
    
    # For the visualization of the HoperRobot
    hr = mp1.HoperRobot()
    visual = mp1.Visualization(base_coord=hr.base_position,ball_size=150,step=0.005)
    visual._offset_up = 6
    visual._offset_down = -6
    # visual.fig.canvas.mpl_connect('close_event',mp1.on_close)
    plt.draw()
    
    # The mass and the gravity acceleration
    m = 5
    g = 10
    
    # Quadratic term of the Quadratic function
    Q = np.eye(4)
    
    # Linear term of the Quadratic function
    q1 = np.array([[0,0,0,0]]).T
    
    # The Mass Matrix of the system, it extract from system equations
    Mass_Matrix = np.eye(2)*m
    
    rx = x_init[0,0]/np.sqrt((x_init[0,0])**2 + (x_init[1,0])**2)
    ry = x_init[1,0]/np.sqrt((x_init[0,0])**2 + (x_init[1,0])**2)
    
    # The S matrix here is not the identity In, where n is the joints
    S = -1*np.array([[rx,-ry],[ry,rx]])
    
    # The equality constrain respect the system dynamics
    
    # The equality constrain matrix Ax = b
    A = np.hstack((Mass_Matrix,S))
    
    # The equality constrain vector Ax = b
    b = np.array([[0,-m*g]]).T
    
    qp = proxsuite.proxqp.dense.QP(4,2,0)
    qp.init(Q,q1,A,b,None,None,None)
    
    
    # TODO COMMENT OUT TO TEST
    # My Implementation - QP ADMM
    
    # z_0 = l = u = np.copy(b)
    # y_0 = np.zeros_like(z_0)
    
    # ###########################
    
    
    # Starting time
    t = 0
    dt = 0.005
        
    # The control inputs
    u = np.copy(u_init)
    
    # Current Position and Velocities
    # Upacking the state array to scalar variables
    x = x_init[0,0] 
    y = x_init[1,0]
    x_dot = x_init[2,0]
    y_dot = x_init[3,0]
    
    # Initializing the state and control u
    x_state = np.copy(x_init)
    u = np.copy(u_init)
    
    # Calculating the initial acceleration
    # Unpacking the initial acceleration to scalar variables
    x_state_dot = hr.dynamics(x_state,u)
    x_ddot = x_state_dot[2,0] # The current x acceleration
    y_ddot = x_state_dot[3,0] # The current y acceleration
    
    # The trajectory that need to be follow
    # The Actual Plot
    Time = np.linspace(0,10,350)
    
    X_plot = q(Time,phi = 0) + 1
    Y_plot = q(Time,A = 2,f = 1,phi = np.pi/2) - 1
    
    plt.plot(X_plot,Y_plot,"r",alpha = 0.3)
    plt.draw()
    # Actual Plot End

    while t < 10 :
        
        t += dt
        
        # Desired Position
        xd = q(t,phi = 0) + 1
        yd = q(t,A = 2,f = 1,phi = np.pi/2) - 1
        
        # Desired Velocity
        xd_dot = q_dot(t,phi = 0)
        yd_dot = q_dot(t,A = 2,f = 1,phi = np.pi/2)
        
        # Calculating the desired acceleration
        xd_ddot = x_ddot + 1500*(x - xd) + 800 * (x_dot - xd_dot)  
        yd_ddot = y_ddot + 1500*(y - yd) + 800 * (y_dot - yd_dot)
        
        # QP Solver
        
        # ###########################
        # TODO COMMENT OUT TO TEST QP-ADMM
        # My Implementation - QP ADMM
        
        # rho = 1000
        # sigma = 1e-6
        # alpha = 1.5
        # # solver = QP_ADMM_LC(P,q,A,l,u,rho,sigma,alpha)
        # solver = qp2.QP_ADMM_LC(Q,q1,A,l,u,rho,sigma,alpha)
        # x_0 = np.array([[0.,0.,0.,0.]]).T
        # xk,_1,_2,_3,_4 = solver.converge(x_0,y_0,z_0)
        # u = np.array([[xk[2,0],xk[3,0]]]).T
        # visual.fig.suptitle("QP - ADMM")
        # # ###########################   
        
        # ###########################        
        # TODO COMMENT OUT TO TEST QP - Proxsuite
        z = np.array([[0.,0.,0.,0.]]).T
        qp.solve(z,None,None)
        z = qp.results.x
        z = np.copy(z.reshape(4,1))
        
        u = np.array([[z[2,0],z[3,0]]]).T
        visual.fig.suptitle("QP - Proxsuite")
        # ###########################

        # u = limit_controls(u)
        
        x_state_dot = hr.dynamics(x_state,u) 
        x_state = hr.runge_kutta4_discrete(x_state,u,dt) # Calculating the next state
        
        # Unpacking the variables
        x = x_state[0,0] 
        y = x_state[1,0]
        x_dot = x_state[2,0]
        y_dot = x_state[3,0]
        
        x_ddot = x_state_dot[2,0]
        y_ddot = x_state_dot[3,0]
        
        # updating the QP Matrixes
        q1,A = update_QP_Matrixes(Mass_Matrix,x_state,x_ddot - xd_ddot ,x_ddot - yd_ddot)
        qp.update(Q,q1,A,b,None,None,None)
        
        # Updating the visualization
        visual.update(x,y)
        
        print("Controls u")
        print(u)



if __name__ == "__main__":
    main()