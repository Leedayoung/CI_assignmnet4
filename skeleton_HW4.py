#Filename: HW4_skeleton.py
#Author: Christian Knoll, Florian Kaum
#Edited: May, 2018

import math
import scipy.spatial.distance as dist
import numpy as np
from numpy import unravel_index
import pylab
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats
import functools

from scipy.stats import multivariate_normal

#--------------------------------------------------------------------------------
# Assignment 4
def main():
    
    # choose the scenario
    # scenario = 1    # all anchors are Gaussian
    # scenario = 2    # 1 anchor is exponential, 3 are Gaussian
    scenario = 3    # all anchors are exponential
    
    # specify position of anchors
    p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
    nr_anchors = np.size(p_anchor,0)
    
    # position of the agent for the reference mearsurement
    p_ref = np.array([[0,0]])
    # true position of the agent (has to be estimated)
    p_true = np.array([[2,-4]])
#    p_true = np.array([[2,-4])
                       
    # plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref)
    
    # load measured data and reference measurements for the chosen scenario
    data,reference_measurement = load_data(scenario)
    
    # get the number of measurements 
    assert(np.size(data,0) == np.size(reference_measurement,0))
    nr_samples = np.size(data,0)
    
    #1) ML estimation of model parameters 
    params = parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref)
    
    #2) Position estimation using least squares
    position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, False)

    if(scenario == 3):
        # TODO: don't forget to plot joint-likelihood function for the first measurement

        x = np.arange(-5, 5, .05)
        y = np.arange(-5, 5, .05)
        xx, yy = np.meshgrid(x, y)
        # construct a mesh for each anchor of the distance from the anchor -- to be used in determining likelihoods
        distances = [np.sqrt((anchor[0] - xx)**2 + (anchor[1] - yy)**2) for anchor in p_anchor]
        likelihoods = [
            np.where(data[0][i] > distances[i], params[0][i] * np.exp(-1 * params[0][i] * (data[0][i] - distances[i])), 0)
            for i in range(nr_anchors)]

        # multiply the likelihood mesh from each anchor together to obtain the joint likelihood
        joint_likelihood = functools.reduce(np.multiply, likelihoods)
        plt.contour(x, y, joint_likelihood)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Joint Likelihood Distribution of the First Data Point')
        plt.show()

        #likeleyhoods = [params[0][i] * np.exp(-params[0][i] * (data[0][i] - ))]

        #3) Postion estimation using numerical maximum likelihood
        position_estimation_numerical_ml(data,nr_anchors,p_anchor, params, p_true)
    
        #4) Position estimation with prior knowledge (we roughly know where to expect the agent)
        # specify the prior distribution
        prior_mean = p_true
        prior_cov = np.eye(2)
        position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov, params, p_true)

    pass


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def parameter_estimation(reference_measurement,nr_anchors,p_anchor,p_ref):
    """ estimate the model parameters for all 4 anchors based on the reference measurements, i.e., for anchor i consider reference_measurement[:,i]
    Input:
        reference_measurement... nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2
        p_ref... reference point, 2x2 """
    params = np.zeros([1, nr_anchors])
    for i in range(0,nr_anchors):
        #(1) check whether a given anchor is Gaussian or exponential
        ref = reference_measurement[:,i]
        true_distance = dist.euclidean(p_ref,p_anchor[i])
        ref = ref - true_distance
        D, _ = stats.kstest(ref, "expon")

        # print("The "+str(i+1)+"th anchor "+ str(D))
        #(2) estimate the according parameter based
        # print("-------------------------------------------------------")
        if D < 0.1 : #exponential distribution
            params[0][i] = 1/np.mean(np.mean(ref))
            # print("The "+str(i+1)+"th anchor follows the Exponential model.")
            # print("The parameter lambda is "+str(params[0][i]))
        else: # gaussian distribution
            params[0][i] = math.pow(np.var(ref),2)
            # print("The "+str(i+1)+"th anchor follows the Gaussian model.")
            # print("The parameter sigma square is "+str(params[0][i]))
    
    return params
#--------------------------------------------------------------------------------
def position_estimation_least_squares(data,nr_anchors,p_anchor, p_true, use_exponential):
    """estimate the position by using the least squares approximation. 
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        p_true... true position (needed to calculate error) 2x2 
        use_exponential... determines if the exponential anchor in scenario 2 is used, bool"""
    nr_samples = np.size(data,0)

    tol = 0.000005  # tolerance
    max_iter = 10  # maximum iterations for GN
    
    # estimate position 
    p_expected = np.zeros([nr_samples,2])
    

    #iterate through sample
    for i in range(0, nr_samples):
        #uniform distribution within 4 achors
        if use_exponential == True:
            del_anchor = np.random.randint(0,3)
            sel_anchors = np.delete(p_anchor,del_anchor,0)
            ran_p = np.sort(np.random.rand(2, 1), axis=0)
            p_start = np.transpose(np.column_stack([ran_p[0], ran_p[1]-ran_p[0], 1.0-ran_p[1]]) @ sel_anchors)
            p_expected[i,:] = least_squares_GN(p_anchor,p_start, data[i], max_iter, tol)
        else:
            sel_anchors = p_anchor[1:4,:]
            ran_p = np.sort(np.random.rand(2, 1), axis=0)
            p_start = np.transpose(np.column_stack([ran_p[0], ran_p[1]-ran_p[0], 1.0-ran_p[1]]) @ sel_anchors)
            p_expected[i,:] = least_squares_GN(sel_anchors,p_start, data[i,1:4], max_iter, tol)

	# calculate error measures and create plots----------------
    p_error = (-1)*p_expected + p_true
    p_error = p_error * p_error
    p_error = np.sqrt(p_error[:,0]+p_error[:,1])
    per_mean = np.mean(p_error, axis =0)
    per_variance = np.var(p_error, axis =0)

    print('--------------------------------------------------------------------------------')
    print("The mean of the position estimation error is " + str(per_mean))
    print("The variance of the position estimation error is " + str(per_variance))
    print('--------------------------------------------------------------------------------')
    
    p_mean = np.mean(p_expected, axis = 0)
    p_cov = np.cov(p_expected[:,0],p_expected[:,1])
    pT_expected = np.transpose(p_expected)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(pT_expected[0], pT_expected[1], 'ro', ms=1)

    plot_gauss_contour(p_mean,p_cov,1,3,-6,-2,"Scatter plots of the estimated positions")
    
    Fx,x = ecdf(p_error) 
    plt.plot(x,Fx)
    plt.title('cumulative distribution function (CDF) of the position estimation error')
    plt.ylabel('F(x)')
    plt.xlabel('x')
    plt.show()

    pass
#--------------------------------------------------------------------------------
def position_estimation_numerical_ml(data,nr_anchors,p_anchor, lambdas, p_true):
    """ estimate the position by using a numerical maximum likelihood estimator
    Input:
        data...distance measurements to unkown agent, nr_measurements x nr_anchors
        nr_anchors... scalar
        p_anchor... position of anchors, nr_anchors x 2 
        lambdas... estimated parameters (scenario 3), nr_anchors x 1
        p_true... true position (needed to calculate error), 2x2 """

    x = np.arange(-5, 5, .05)
    y = np.arange(-5, 5, .05)
    xx, yy = np.meshgrid(x, y)
    # construct a mesh for each anchor of the distance from the anchor -- to be used in determining likelihoods
    distances = [np.sqrt((anchor[0] - xx) ** 2 + (anchor[1] - yy) ** 2) for anchor in p_anchor]

    position_estimations = []
    for d in data:
        likelihoods = [
            np.where(d[i] > distances[i], lambdas[0][i] * np.exp(-lambdas[0][i] * (d[i] - distances[i])), 0) for i
            in range(nr_anchors)]

        # multiply the likelihood mesh from each anchor together to obtain the joint likelihood
        joint_likelihood = functools.reduce(np.multiply, likelihoods)

        ml_index = unravel_index(joint_likelihood.argmax(), joint_likelihood.shape)
        p = (x[ml_index[1]], y[ml_index[0]])
        position_estimations.append(p)

    p_est_x = [p[0] for p in position_estimations]
    p_est_y = [p[1] for p in position_estimations]
    plt.plot(p_est_x, p_est_y, 'bo')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Numerical ML Estimation of the Target Point (2, -4)')
    pylab.xlim([-5, 5])
    pylab.ylim([-5, 5])
    plt.show()


#--------------------------------------------------------------------------------
def position_estimation_bayes(data,nr_anchors,p_anchor,prior_mean,prior_cov,lambdas, p_true):
    """ estimate the position by accounting for prior knowledge that is specified by a bivariate Gaussian
    Input:
         data...distance measurements to unkown agent, nr_measurements x nr_anchors
         nr_anchors... scalar
         p_anchor... position of anchors, nr_anchors x 2
         prior_mean... mean of the prior-distribution, 2x1
         prior_cov... covariance of the prior-dist, 2x2
         lambdas... estimated parameters (scenario 3), nr_anchors x 1
         p_true... true position (needed to calculate error), 2x2 """

    x = np.arange(-5, 5, .05)
    y = np.arange(-5, 5, .05)
    xx, yy = np.meshgrid(x, y)
    # construct a mesh for each anchor of the distance from the anchor -- to be used in determining likelihoods
    distances = [np.sqrt((anchor[0] - xx) ** 2 + (anchor[1] - yy) ** 2) for anchor in p_anchor]
    # construct a mesh of the gaussian pdf for the point p
    prior_pdf = mlab.bivariate_normal(xx, yy, np.sqrt(prior_cov[0][0]), np.sqrt(prior_cov[1][1]), prior_mean[0][0], prior_mean[0][1], prior_cov[0][1])

    position_estimations = []
    for d in data:
        likelihoods = [
            np.where(d[i] > distances[i], lambdas[0][i] * np.exp(-lambdas[0][i] * (d[i] - distances[i])), 0) for i
            in range(nr_anchors)]

        # multiply the likelihood mesh from each anchor together to obtain the joint likelihood
        joint_likelihood = functools.reduce(np.multiply, likelihoods)
        # bayesian_likelihood is p(r|p) * p(p)
        bayesian_likelihood = joint_likelihood * prior_pdf

        ml_index = unravel_index(bayesian_likelihood.argmax(), bayesian_likelihood.shape)
        p = (x[ml_index[1]], y[ml_index[0]])
        position_estimations.append(p)

    p_est_x = [p[0] for p in position_estimations]
    p_est_y = [p[1] for p in position_estimations]
    plt.plot(p_est_x, p_est_y, 'bo')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bayesian Estimation of the Target Point (2, -4)')
    pylab.xlim([-5, 5])
    pylab.ylim([-5, 5])
    plt.show()



#--------------------------------------------------------------------------------
def least_squares_GN(p_anchor,p_start, r, max_iter, tol):
    """ apply Gauss Newton to find the least squares solution
    Input:
        p_anchor... position of anchors, nr_anchors x 2
        p_start... initial position, 2x1
        r... distance_estimate, nr_anchors x 1
        max_iter... maximum number of iterations, scalar
        tol... tolerance value to terminate, scalar"""
    
    a_size = int(np.size(p_anchor)/2)
    J = np.zeros([a_size,2])
    b = np.zeros([a_size,1])
    
    for _ in range(0,max_iter):
        for k in range(0,a_size):
            p_start_p_anchor = dist.euclidean(p_start,p_anchor[k])
            b[k] = r[k]-p_start_p_anchor
            J[k][0] = -(p_start[0] - p_anchor[k][0])/p_start_p_anchor
            J[k][1] = -(p_start[1] - p_anchor[k][1])/p_start_p_anchor
        solution = np.linalg.lstsq(J, b)[0]
        p_next = p_start - solution
        if dist.euclidean(p_next,p_start) < tol :
            break
        else :
            p_start = p_next
    return np.ndarray.flatten(p_start)
    
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Helper Functions
#--------------------------------------------------------------------------------
def plot_gauss_contour(mu,cov,xmin,xmax,ymin,ymax,title="Title"):
    
    """ creates a contour plot for a bivariate gaussian distribution with specified parameters
    
    Input:
      mu... mean vector, 2x1
      cov...covariance matrix, 2x2
      xmin,xmax... minimum and maximum value for width of plot-area, scalar
      ymin,ymax....minimum and maximum value for height of plot-area, scalar
      title... title of the plot (optional), string"""
    
	#npts = 100
    delta = 0.025
    x = np.arange(xmin, xmax, delta)
    y = np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X,Y,np.sqrt(cov[0][0]),np.sqrt(cov[1][1]),mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    plt.show()
    return

#--------------------------------------------------------------------------------
def ecdf(realizations):   
    """ computes the empirical cumulative distribution function for a given set of realizations.
    The output can be plotted by plt.plot(x,Fx)
    
    Input:
      realizations... vector with realizations, Nx1
    Output:
      x... x-axis, Nx1
      Fx...cumulative distribution for x, Nx1"""
    x = np.sort(realizations)
    Fx = np.linspace(0,1,len(realizations))
    return Fx,x

#--------------------------------------------------------------------------------
def load_data(scenario):
    """ loads the provided data for the specified scenario
    Input:
        scenario... scalar
    Output:
        data... contains the actual measurements, nr_measurements x nr_anchors
        reference.... contains the reference measurements, nr_measurements x nr_anchors"""
    data_file = 'measurements_' + str(scenario) + '.data'
    ref_file =  'reference_' + str(scenario) + '.data'
    
    data = np.loadtxt(data_file,skiprows = 0)
    reference = np.loadtxt(ref_file,skiprows = 0)
    
    return (data,reference)
#--------------------------------------------------------------------------------
def plot_anchors_and_agent(nr_anchors, p_anchor, p_true, p_ref=None):
    """ plots all anchors and agents
    Input:
        nr_anchors...scalar
        p_anchor...positions of anchors, nr_anchors x 2
        p_true... true position of the agent, 2x1
        p_ref(optional)... position for reference_measurements, 2x1"""
    # plot anchors and true position
    plt.axis([-6, 6, -6, 6])
    for i in range(0, nr_anchors):
        plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
        plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
    plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
    plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
    if p_ref is not None:
        plt.plot(p_ref[0, 0], p_ref[0, 1], 'r*')
        plt.text(p_ref[0, 0] + 0.2, p_ref[0, 1] + 0.2, '$p_{ref}$')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.show()
    pass

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
