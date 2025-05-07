import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import proxsuite


class QP_ADMM_LC:
    
    def __init__(self,P: ndarray, q: ndarray, A: ndarray, l :ndarray, u:ndarray, rho: float, sigma: float, alpha: float):
        """ This is a Quadratic Programming Solver that use the ADMM: Alternative Direction Method of Multipliers
            to solve LC : Linear Constrains
            
            Objective Function has the form of: (For Solving the Linear Systems)
            
                (1/2)x.T @ P @ x + q.T @ x + (sigma/2) * || x - xk ||**2 + (rho/2) * || z - zk + (1/rho)*yk  ||**2
                
                with the equality constrain: A @ x = z
                
                Also all the norm is the Frobineus norm, ord = 2
                
        """
        if rho <= 0.0 or sigma <=0.0:
            print(f"Please provide positive number for the rho and sigma,\nThe number of rho and sigma is {rho} {sigma} respectfully")
            return None
        
        if alpha <= 0 or alpha >= 2:
            print(f"Please provide a number for the alpha is containing in the following space (0,2)\nThe given value for alpha is {alpha}")
            return None
        
        self.rho = rho
        self.sigma = sigma
        self.alpha = alpha
        self.P = P # Is the matrix that describe the second order term in Quadratic function, It need to be a positive-semidefinite matrix
        self.q = q # Is the matrix that describe the first order term in Quadratic function.
        self.A = A # Is the matrix that describe the linear constrain of the Quadratic problem
        self.l = l # The lower limit of the C convex set
        self.u = u # The upper limit of the C convex set
        # The above limits are the same when solving for equality constrains
        
        N = np.shape(P)[0] # This is the numbers of optimization variables, so it is the Input Dimension
        M = np.shape(A)[0] # This is the numbers of equality constrains equations. 
        
        self.N = N
        self.M = M
        
    def detect_primal_infeasibility(self,delta_yk: ndarray):
        
        # A tolerance level
        e_pinf = 1e-4
        
        t1 = np.linalg.norm(self.A.T@delta_yk, ord = np.inf)
        t2 = e_pinf*np.linalg.norm(delta_yk, ord = np.inf)
        
        
        # Calculating the Dy+ and Dy- paragraph 3.4 of paper
        delta_yk_plus = np.zeros_like(delta_yk)
        delta_yk_minus = np.zeros_like(delta_yk)
    
    
        row_ydim , col_ydim = np.shape(delta_yk)
        for row in range(row_ydim):
            for col in range(col_ydim):
                
                yi = delta_yk[row,col]
                
                if yi < 0:
                    delta_yk_minus[row,col] = yi
                    delta_yk_plus[row,col] = 0
                
                if yi >= 0 :
                    
                    delta_yk_minus[row,col] = 0
                    delta_yk_plus[row,col] = yi
        
        t3 = self.u.T@delta_yk_plus + self.l.T@delta_yk_minus
        
        if t1 <= t2 and t3 <= t2:
            # print("The system satisfied the primal infeasibility")
            return True
        
        # print("The system is NOT satisfied the primal infeasibility")
        return False
        
    
    def detect_dual_infeasibility(self,delta_xk: ndarray):
        
        # A tolerance level
        e_dinf = 1e-4
        
        t1 = np.linalg.norm(self.P @ delta_xk, ord = np.inf)
        t2 = e_dinf* np.linalg.norm(delta_xk, ord = np.inf)
        t3 = self.q.T@delta_xk
        

        cond3 = self.check_3rd_cond_dual_infeasibility(delta_xk,e_dinf)
        
        if t1 <= t2 and t3 <= t2 and cond3:
            # print("The system satisfied the dual infeasibility")
            return True
        
        # print("The system is NOT satisfied the dual in infeasibility")
        return False



    def check_3rd_cond_dual_infeasibility(self,delta_xk,e_dinf):
        t2 = e_dinf* np.linalg.norm(delta_xk, ord = np.inf)
        
        delta_ = self.A @ delta_xk
        
        row_dim, col_dim = np.shape(delta_)
        
        for row in range(row_dim):
            for col in range(col_dim):
                
                if np.isposinf(self.u[row,col]):
                    
                    if delta_[row,col] < - t2:
                        return False
                    
                elif np.isneginf(self.l[row,col]):
                    
                    if delta_[row,col] > t2:
                        return False
                    
                else:
                    if delta_[row,col] > t2 and delta_[row,col] < -t2:
                        return False
        return True
    
    def _euclidean_projection(self ,z):
        row_ldim , col_ldim = np.shape(z)
        for row in range(row_ldim):
            for col in range(col_ldim):
                
                zi = z[row,col]
                li = self.l[row,col]
                ui = self.u[row,col]
                
                if zi < li:
                    zi = li
                    z[row,col] = zi
                
                if zi > ui:
                    zi = ui
                    z[row,col] = zi
        
        return z
    
    def converge(self,x_init : ndarray, y_init : ndarray, z_init : ndarray,visual: object = None) -> list:
        
        
        visual_flag: bool = True
        if visual == None:
            # print("No Visualization added to Solver")
            visual_flag = False
        
        xk = x_init
        yk = y_init
        zk = z_init
        
        e_abs = 1e-4
        
        e_rel = 1e-3
        
        
        it = 0
        while True:
            KKT_matrix = np.block([[self.P + self.sigma * np.eye(self.N),self.A.T],[self.A,-np.eye(self.M)/self.rho]])
            KKT_TARGET_VECTOR = np.block([[self.sigma * xk - self.q],[zk - yk/self.rho]])
            # Solving the system
            DZ = np.linalg.solve(KKT_matrix,KKT_TARGET_VECTOR)
            
            # dim = 2
            
            xk_1 = DZ[:self.N]
            
            vk_1 = DZ[self.N:] # Lagrange Multipliers for equality constrain
            
            zk_1 = zk + ( ( vk_1 - yk ) / self.rho )
            
            xk_1 = self.alpha * xk_1 + ( 1 - self.alpha ) * xk
            
            zk_1 = self.alpha * zk_1 + ( 1 - self.alpha ) * zk + ( yk / self.rho )
            zk_1 = self._euclidean_projection(zk_1)
            
            yk_1 = yk + self.rho * ( self.alpha * zk_1 + ( 1 - self.alpha ) * zk - zk_1 )
            
            
            
            # Checking the primal and dual infeasibility of the problem
            
            if self.detect_primal_infeasibility(delta_yk = yk_1 - yk):
                
                if self.detect_dual_infeasibility(delta_xk = xk_1 - xk):
                    
                    # print("System satisfied the primal and dual infeasibility and we get the solution !")
                    # print(f"delta_xk:\n{xk_1 - xk}\n")
                    # print(f"delta_yk:\n{yk_1 - yk}\n")
                    # print(f"delta_zk:\n{zk_1-zk}\n")
                    xk_1 = xk_1 - xk
                    yk_1 = yk_1 - yk
            
            
            xk = xk_1
            yk = yk_1
            zk = zk_1
            
            r_prim = self.A @ xk - zk
            r_dual = self.P @ xk + self.q + self.A.T @ yk
        
            e_prim = e_abs + e_rel * max(np.linalg.norm(self.A,ord = np.inf), np.linalg.norm(zk,ord = np.inf))
            e_dual = e_abs + e_rel * max(np.linalg.norm(self.P @ xk, ord = np.inf), np.linalg.norm(self.A.T @ yk, ord = np.inf), np.linalg.norm(self.q,ord = np.inf))

            if np.linalg.norm(r_prim, ord = np.inf) <= e_prim or np.linalg.norm(r_dual, ord = np.inf) <= e_dual:
                print("Solution Found !")
                # print(f"xk:\n{xk}\n\nyk:\n{yk}\n\nzk:\n{zk}\n\nvk+1:\n{vk_1}\n\niterations:\n{it}\n\n")
                return [xk,yk,zk,vk_1,it]
    
            if it == 8000:
                print(f"Couldn't Found a solution after {it} iterations")
                # print(f"xk:\n{xk}\n\n yk:\n{yk}\n\n zk:\n{zk}\n\n vk+1:\n{vk_1}")
                return [xk,yk,zk,vk_1,it]
            it +=1
            
            if visual_flag:
                visual.update(xk[0],xk[1])                
            
            
    
    def __str__(self) -> str:
        return f"QP_ADMM_LS Solver with initial conditions:\n\nrho:\t{self.rho}\nsigma:\t{self.sigma}\nalpha:\t{self.alpha}\n"

class Visualization:
    """ This class provide visualization only for 3d problems that translated to 2d space 
        It actually help visualize some problems and how the implemented qp are solving
    """
    
    def __init__(self,x_init,func,cons,l,u):
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)
        
        # Plotting the start point with blue color
        self.axes.plot(x_init[0],x_init[1],"bx")
        
        x = np.linspace(-abs(np.min(l)) - 2 ,np.max(u) + 2,50)
        y = np.linspace(-abs(np.min(l)) - 2 ,np.max(u) + 2,50)
        
        X,Y = np.meshgrid(x,y)
        
        plt.ion()  
        
        f = func(X,Y)
        plt.contour(X,Y,f)
        
        for cons_func in cons:
            c = cons_func(X,Y)
            plt.contour(X,Y,c,levels = [0])
        plt.draw()
        plt.pause(1)
        
    def update(self,x: float ,y: float):
        # Plotting the next x that qp solver is calculating
        self.axes.plot(x,y,"rx")
        plt.pause(0.4)

def on_close(event):
    print(f"User_Interacted ended on on_close, event_triggered: {event}")
    exit(1)


# This is the quadratic function 
def func(x1,x2):
    return x1**2 + x2**2

# This is the equality constrains of the problem 
# We define the equality constrains like this 
# because on plotting is used the level = [0]
def cons0(x1,x2):
    return x1+x2-5

def cons1(x1,x2):
    return x1-x2-3

def main() -> int:    
    # The following quadratic function is :
    # f(x) = x1**2 + x2**2 minimazation with respect to x1,x2
    # Also equality constrains x1 + x2 = 5 and x1-x2 = 3
    
    x_init = np.array([[-1,1]]).T
    z_init = np.array([[5,3]]).T
    
    P = np.array([[2,0],[0,2]])
    q = np.array([[0,0]]).T
    A = np.array([[1,1],[1,-1]])
    
    # This is for the equality constrain, Ax = z | z E C , or  l<z<u  where C = [l,u] 
    l = np.array([[5,3]]).T
    u = np.array([[5,3]]).T
    cons = [cons0,cons1]
    visual = Visualization(x_init,func,cons,l,u)
    visual.fig.suptitle("QP-ADMM")
    visual.fig.canvas.mpl_connect('close_event',on_close)    
    
    y_init = np.zeros_like(z_init)
    
    rho = 1000
    sigma = 1e-6
    alpha = 1.5
    solver = QP_ADMM_LC(P,q,A,l,u,rho,sigma,alpha)
    print(solver)
    [xk,yk,zk,vk_1,it] = solver.converge(x_init,y_init,z_init,visual)
    print(40*"=")
    print(f"xk:\n{xk}\n\nyk:\n{yk}\n\nzk:\n{zk}\n\nvk+1:\n{vk_1}\n\niterations:\n{it}\n\n")
    print(40*"=")
    plt.pause(5)
    
    
    
    # Solving the same problem using Proxsuite to test QP-ADMM
    print("Now Solving with Proxsuite below")
    qp = proxsuite.proxqp.dense.QP(2,2,0)
    qp.init(P,q,A,z_init,None,None,None)
    
    x_init_prox = np.copy(x_init)
    
    qp.solve(x_init_prox,None,None)
    x_prox = qp.results.x
    x_prox = np.copy(x_prox.reshape(2,1))
    
    print(f"Proxsuite Solution :\n{x_prox}\n")
    
    
    print("Comparing the solutions")
    print("||xk - x_prox||",np.linalg.norm(xk - x_prox),"\n")
    print("Should be zero !\n")
    
    
    return 0

if __name__ == "__main__":
    print("Main has ended with",main())