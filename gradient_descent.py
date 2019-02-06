def descent_update_AG(x, desc_dir, alpha=0.25, beta=0.5):
    
    """ 
    x: is a d-dimensional vector giving the current iterate.
       It is represented as numpy array with shape (d,)
    desc_dir: is a d-dimensional vector giving the descent direction, the next iterate
             is along this direction.
             It is represented as numpy array with shape (d,)
    Returns
    
    Next iterate where step size is chosen to satisfy Armijo-Goldstein conditions.
    Should be represented by numpy array with shape (d,)
    """
    get_grad = get_gradient(x)
    val = get_value(x)
    grad_component = np.dot(get_grad, desc_dir)
    if grad_component >=0 :
        grad_component = -1 * grad_component
      
    cand_eta = 1
    min_eta = 0 # the smallest eta to try
    max_eta = np.inf # the largest eta to try
    while True:
        cand_x = x + cand_eta * desc_dir
        f = get_value(cand_x)
        f1 = val + beta * cand_eta * grad_component  
        f2 = val + alpha * cand_eta * grad_component #f2 is always greater than f1.

        if f>=f1 and f<=f2:
            #The value at this point is between the lines so terminate       
            break
            
        if f<f1: 
            # value at point is below both lines, so increase eta, 
            # only try eta larger than current eta
            min_eta = cand_eta
            cand_eta = min(2.0*cand_eta, (cand_eta+ max_eta)/2.0) 
            # increasing eta to mid point of current eta and max eta. Setting to 2*current eta, if max eta =infinity.
            
        if f>f2:
            # value at point is above both lines, so decrease eta, 
            # only try eta smaller than current eta
            max_eta = cand_eta
            cand_eta = (cand_eta+min_eta)/2.0 
            # decreasing eta to mid point of current eta and min eta.
    
    return cand_x
   
def descent_update_FR(x, desc_dir):
    """ 
    x: is a d-dimensional vector giving the current iterate.
       It is represented as numpy array with shape (d,)
    desc_dir: is a d-dimensional vector giving the descent direction, the next iterate
             is along this direction.
             It is represented as numpy array with shape (d,)

    Returns
    Next iterate where step size is chosen to satisfy full relaxation conditions (approximately)
    Should be represented by numpy array with shape (d,)
    """
    start=0#A candidate eta to start with
    end=10#A candidate eta to end with
    while True:
        mid=(start+end)/2.0;
        grad = np.dot(get_gradient(x + mid * desc_dir).T,desc_dir)#gets gradient of g(eta) at mid
        grad1 = np.dot(get_gradient(x + start * desc_dir).T,desc_dir)#gets gradient of g(eta) at start
        grad2 = np.dot(get_gradient(x + end * desc_dir).T,desc_dir)#gets gradient of g(eta) at end
        if((grad1<0 and grad2>0) or (grad1>0 and grad2<0)):#If the gradient at start and end points are of opposite sign then the gradient becomes zero somewhere in between(which is checked recursively)
            if((grad1<0 and grad)>0 or (grad1>0 and grad<0)):
                end=mid
            else:
                start=mid
            if(abs(end-start)<tol):#tol was defined at 1e-6 in the template
                break#If start and end points are close together then break
        else:
            break#Breaks if the gradients at the starting and ending points are of the same sign
    return(x+mid*desc_dir)

        

def BFGS_update(H, s, y):
    """ Returns H_{k+1} given H_k and s and y based on BFGS update.
    H: numpy array with shape (d,d)
    s: numpy array with shape (d,)
    y: numpy array with shape (d,)
    """
    mat = np.eye(s.size)#Idendity matrix of d dimension
	#The parameters are calculated according to the BFGS_update rule
    rho=1/(np.dot(s,y))
    mat1=mat - (rho * np.dot(s[:,None],y[None,:]))
    mat2=rho * np.dot(s[:,None],s[None,:])
    H1=np.dot(mat1,np.dot(H,mat1.T)) + mat2
    return(H1)


def gradient_descent(x0, num_iter=100, eta='AG'):
    """Runs gradient descent till convergence or till number of iterations.
    
    x0: Initial point , represented by numpy array with shape (d,)
    num_iter: number of iterations to run
    eta: The rule by which step size is set. It can take the string values of "AG" or "FR" 
         corresponding to Armijo-Goldstein and Full relaxation criteria. It can also take a
         positive real value, in which case the step size is set to that constant.
    
    Returns:
    Final iterate which is a d-dimensional vector represented by numpy array with shape (d,). 
    The algorithm can be stopped if either  the number of iterations is reached
    or if the change in iterates is less than tol. i.e. ||x_{t+1} -x_{t}||<=tol.
    
    """
    x=x0
    for i in range(num_iter):
        if(eta=='AG'):
            x = descent_update_AG(x,get_gradient(x)*(-1), alpha=0.25, beta=0.5)#Runs gradient_descent with step size chosen according to Armijo-Goldstein method
        elif(eta=='FR'):
            x = descent_update_FR(x,get_gradient(x)*(-1))#Runs gradient_descent with step size chosen according to full relaxation method
        else:
            cand_eta=eta
            x = x - (cand_eta * get_gradient(x))##Runs gradient_descent with step size give already
    return x

def quasi_Newton(x0, H0, num_iter=10000, eta='AG'):
    #print("q")
    """Runs Quasi Newton with BFGS till convergence or till number of iterations.
    
    x0: Initial point , represented by numpy array with shape (d,)
    H0: Initial inverse Hessian estimate. Represented by numpy array with shape (d,d)
    num_iter: number of iterations to run
    eta: The rule by which step size is set. It can take the string values of "AG" or "FR" 
         corresponding to Armijo-Goldstein and Full relaxation criteria. It can also take a
         positive real value, in which case the step size is set to that constant.
    
    Returns:
    Final iterate which is a d-dimensional vector represented by numpy array with shape (d,). 
    The algorithm can be stopped if either  the number of iterations is reached
    or if the change in iterates is less than tol. i.e. ||x_{t+1} -x_{t}||<=tol.
    
    """
    for i in range(num_iter):
        desc_dir = np.dot(H0,get_gradient(x0))#descent direction
        if (eta=='AG'):
            if(np.dot(get_gradient(x0), desc_dir)>=0):#If the descent direction is not in the negative direction the multiply by -1 else the descent direction is itself               
                x=descent_update_AG(x0,(-1)*desc_dir,alpha=0.25,beta=0.5)
                H0=(-1)*H0
            else:
                x=descent_update_AG(x0,desc_dir,alpha=0.25,beta=0.5)
            s=x-x0
            
        elif(eta=='FR'):
            if(np.dot(get_gradient(x0), desc_dir)>=0):#If the descent direction is not in the negative direction the multiply by -1 else the descent direction is itself             
                x=descent_update_FR(x0,(-1)*desc_dir)
                H0=(-1)*H0
            else:
                x=descent_update_FR(x0,desc_dir)
        else:
            x=x0 - eta* desc_dir#Updation of x
        s=x-x0
        if(s.all()==0):#If the updated x is same as the previous x then it implies it has converged
            break
        y=get_gradient(x) - get_gradient(x0)
        H0 = BFGS_update(H0,s,y)
        x0=x    
    return x0
