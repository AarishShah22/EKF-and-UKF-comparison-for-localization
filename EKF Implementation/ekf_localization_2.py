"""
Here, I have implemented an Extended Kalman 
Filter for Mobile Robot Localization. I have
used the UTIAS Localization and Mapping Dataset 
for this project.

Data available: wheel odometry data (linear
velocity and angular velocity), sensor data
(range and bearing of obstacles) and map
(XY postions of all the landmarks).
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

# Data Handling: Loading all data from .dat files to np arrays
lm_file_path = r"C:\Users\aarishs\Desktop\Academics\Sem I\AEROSP 567\Final Project\Implementation\Landmark_Groundtruth.dat"
truth_file_path = r"C:\Users\aarishs\Desktop\Academics\Sem I\AEROSP 567\Final Project\Implementation\Robot1_Groundtruth.dat"
barcodes_file_path = r"C:\Users\aarishs\Desktop\Academics\Sem I\AEROSP 567\Final Project\Implementation\Barcodes.dat"
meas_file_path = r"C:\Users\aarishs\Desktop\Academics\Sem I\AEROSP 567\Final Project\Implementation\Robot1_Measurement.dat"
odom_file_path = r"C:\Users\aarishs\Desktop\Academics\Sem I\AEROSP 567\Final Project\Implementation\Robot1_Odometry.dat"

lm_pos = np.zeros((15,3))
lm_std = np.zeros((15,3))
i = 0
lm_file = open(lm_file_path, "r")
for line in lm_file:
    if not line.startswith("#"):
        values = line.split()
        lm_pos[i,:] = np.array([values[0],values[1],values[2]])
        lm_std[i,:] = np.array([values[0],values[3], values[4]])
        i = i+1
lm_file.close()

l = 0
robot_true_pose = np.zeros((81877,4))
truth_file = open(truth_file_path, "r")
for line in truth_file:
    if not line.startswith("#"):
        values = line.split()
        robot_true_pose[l,:] = np.array([[values[0],values[1], values[2], values[3]]])
        l = l+1

truth_file.close()

k = 0
barcodes = np.zeros((20,2))
barcodes_file = open(barcodes_file_path, "r")
for line in barcodes_file:
    if not line.startswith("#"):
        values = line.split()
        barcodes[k,:] = np.array([[values[0], values[1]]])
        k = k+1

valid_barcodes = barcodes[5:,:]

barcodes_file.close()

m = 0
measurements = np.zeros((5723,4))
meas_file = open(meas_file_path, "r")
for line in meas_file:
    if not line.startswith("#"):
        values = line.split()
        measurements[m,:] = np.array([[values[0],values[1], values[2], values[3]]])
        m = m+1

meas_file.close()

n = 0
odom = np.zeros((97890,3))
odom_file = open(odom_file_path, "r")
for line in odom_file:
    if not line.startswith("#"):
        values = line.split()
        odom[n,:] = np.array([[values[0],values[1], values[2]]])
        n = n+1

odom_file.close()

# Observing path of robot based on just odometry
# measurements - "dead reckoning"
dead_reckon_pose = np.zeros((97891,4))
dead_reckon_pose[0,0] = odom[0,0]
dead_reckon_pose[0,1:] = robot_true_pose[0,1:]
noise = 0

for i in range(1,len(dead_reckon_pose)):
    
    t0 = dead_reckon_pose[i-1,0]
    x0 = dead_reckon_pose[i-1,1]
    y0 = dead_reckon_pose[i-1,2]
    theta0 = dead_reckon_pose[i-1,3]

    t1 = odom[i-1,0]
    v = odom[i-1,1]
    w = odom[i-1,2]

    dt = t1-t0
    dx = v*np.cos(theta0)*dt
    dy = v*np.sin(theta0)*dt
    dtheta = w*dt

    x1 = x0 + dx + np.random.randn()*noise
    y1 = y0 + dy + np.random.randn()*noise
    theta1 = theta0 + dtheta + np.random.randn()*noise
    dead_reckon_pose[i,0] = t1
    
    dead_reckon_pose[i,1:] = np.array([x1, y1, theta1])

def get_std(cov):
    """Get square root of diagonals (standard deviations) from covariances """
    
    N, d, d = cov.shape
    std_devs = np.zeros((N, d))
    for ii in range(N):
        std_devs[ii, :] = np.sqrt(np.diag(cov[ii, :, :]))
        if np.all(np.diag(cov[ii,:,:])>0) == False:
            # print(np.diag(cov[ii,:,:]))
            pass
    return std_devs

def robot_motion(t0,init_pose,t1,odom):
    """
    Function that returns robot's motion
    based on the odometry
    """
    x0 = init_pose[0]
    y0 = init_pose[1]
    theta0 = init_pose[2]
    v = odom[0]
    w = odom[1]

    dt = t1-t0
    dx = v*np.cos(theta0)*dt
    dy = v*np.sin(theta0)*dt
    dtheta = w*dt

    x1 = x0 + dx
    y1 = y0 + dy 
    theta1 = theta0 + dtheta

    final_pose = np.array([x1,y1,theta1])

    return final_pose

def obsv_model(robot_pose, data, barcodes, lm_pos):

    barcode = data[1]
    robot_x = robot_pose[0]
    robot_y = robot_pose[1]
    robot_theta = robot_pose[2]

    # if robot_theta>np.pi:
    #     robot_theta-=2*np.pi
    # elif robot_theta<-np.pi:
    #     robot_theta+=2*np.pi

    for i in range(barcodes.shape[0]):
        if barcode == barcodes[i,1]:
            id = barcodes[i,0]
            
    for j in range(lm_pos.shape[0]):
        if id == lm_pos[j,0]:
            lm_x = lm_pos[j,1]
            lm_y = lm_pos[j,2]

    r_hat = np.sqrt((lm_x-robot_x)**2+(lm_y-robot_y)**2)
    phi_hat = np.arctan2((lm_y-robot_y),(lm_x-robot_x))-robot_theta

    zhat = np.array([r_hat, phi_hat, barcode]) 

    q = (lm_x-robot_pose[0])**2+(lm_y-robot_pose[1])**2
    Hk = np.array([[-(lm_x-robot_x)/np.sqrt(q), -(lm_y-robot_y)/np.sqrt(q),0],[(lm_y-robot_y)/q, -(lm_x-robot_x)/q, -1],[0,0,0]])

    return zhat,Hk

def ekf_pred_step(mean,cov,Q,t0,t1,odom):
    """
    Predict step for EKF localization
    """

    theta = mean[2]
    v = odom[0]
    w = odom[1]
    pred_mean = robot_motion(t0,mean,t1,odom)
    Ak = np.array([[0, 0, -v*np.sin(theta)],[0,0,v*np.cos(theta)],[0,0,0]])
    pred_cov = Ak@cov@Ak.T + Q

    return pred_mean, pred_cov

def ekf_update_step(mean,cov,R,data,barcodes,lm_pos):
    """
    Update step for EKF localization
    """

    if data[1] == -1000:
        return mean,cov
    else:
        mu, Hk = obsv_model(mean, data, barcodes,lm_pos)
        U = cov@Hk.T
        S = Hk@cov@Hk.T + R
        Yk = np.array([data[2], data[3], data[1]])
        # print(mu,Yk,data)
        update_mean = mean + U@np.linalg.inv(S)@(Yk-mu)
        update_cov = cov - U@np.linalg.inv(S)@U.T

        theta = update_mean[2]

        if theta>np.pi:
            theta-=2*np.pi
        elif theta<-np.pi:
            theta+=2*np.pi

        return update_mean, update_cov

def ekf_localization(prior_mean,prior_cov,Q,R,data,barcodes,lm_pos,odom):

    t0 = odom[0,0]
    mean = prior_mean
    cov = prior_cov
    mean_store = np.array([prior_mean])
    cov_store = np.array([prior_cov])
    t_store = np.array([t0])

    for i in range(odom.shape[0]):
        if i%1000 == 0:
            print(i)
        t1 = odom[i,0]
        odomc = odom[i,1:]
        data_observed = False
        data_t = []
        indices_data = []
        
        for j in range(data.shape[0]):
            
            if data[j,0]>=t0 and data[j,0]<=t1:
                data_t += [data[j,0]]
                indices_data += [j]
                data_observed = True

            times_data = np.zeros((len(data_t),2))

            for jj in range(times_data.shape[0]):
                times_data[jj,0] = data_t[jj]
                times_data[jj,1] = indices_data[jj]
                                
        if data_observed == True:
            
            for k in range(times_data.shape[0]):
                t_data = times_data[k,0]
                index = int(times_data[k,1])
                if t_data == t0:
                    pass
                pred_mean, pred_cov = ekf_pred_step(mean,cov,Q,t0,t_data,odom[i,1:])
                update_mean, update_cov = ekf_update_step(pred_mean,pred_cov,R,data[index,:],barcodes,lm_pos)
                mean = update_mean
                cov = update_cov
                t0 = t_data
                
            result_mean, result_cov = ekf_pred_step(mean,cov,Q,t0,t1,odom[i,1:])
            
        else:
            result_mean, result_cov = ekf_pred_step(mean,cov,Q,t0,t1,odom[i,1:])
    
        mean = result_mean
        result_mean = result_mean.reshape(1,3)
        cov = result_cov
        mean_store = np.append(mean_store,result_mean,axis=0)
        cov_store = np.append(cov_store,np.array([result_cov]),axis=0)
        t0 = t1
        t_store = np.append(t_store,t0)

    return mean_store,cov_store,t_store

std_dx = 0.004 
std_dy = 0.004 
std_dtheta = 0.0085 

std_r = 0.002 
std_b  = 0.0085

Q = np.array([[std_dx**2, 0, 0],
            [0, std_dy**2, 0],
            [0, 0, std_dtheta**2]])
R = np.array([[std_r**2, 0,0],
            [0, std_b**2,0],[0,0,0.001]])
prior_mean = robot_true_pose[0,1:]
prior_cov = np.eye(3)

invalid_barcodes = [5,14,41,32,23]
for i in range(measurements.shape[0]):
    if measurements[i,1] in invalid_barcodes:
        measurements[i,1] = -1000

filter_till = 20000

mean_store,cov_store,t_store = ekf_localization(prior_mean,prior_cov,Q,R,measurements[0:1000,:],valid_barcodes,lm_pos,odom[0:filter_till,:])

std_devs = get_std(cov_store)
avg_std = np.mean(std_devs,axis=0)
print(avg_std)

MSE_x = 0
MSE_y = 0
MSE_theta = 0
num = 0
m_used = []
n_used = []

for m in range(t_store.shape[0]):
    for n in range(robot_true_pose[0:filter_till,0].shape[0]):
        if robot_true_pose[n,0] - 0.01 <= t_store[m] <= robot_true_pose[n,0] + 0.01:
            
            if n not in n_used and m not in m_used:
                MSE_x += (robot_true_pose[n,1]-mean_store[m,0])**2
                MSE_y += (robot_true_pose[n,2]-mean_store[m,1])**2
                MSE_theta += (robot_true_pose[n,3]-mean_store[m,2])**2
                num += 1

            m_used += [m]
            n_used += [n]

MSE_x = MSE_x/num
MSE_y = MSE_y/num
MSE_theta = MSE_theta/num

print(MSE_x,MSE_y,MSE_theta,num)

plt.figure()
plt.plot(robot_true_pose[0:filter_till,1], robot_true_pose[0:filter_till,2])
plt.plot(dead_reckon_pose[0:filter_till,1], dead_reckon_pose[0:filter_till,2])
plt.plot(mean_store[:,0],mean_store[:,1])
for i in range(lm_pos.shape[0]):
    plt.plot(lm_pos[i,1], lm_pos[i,2],"x",color="red")
    plt.annotate(str(lm_pos[i,0]),(lm_pos[i,1],lm_pos[i,2]))
plt.xlabel("Global X Positon")
plt.ylabel("Global Y Position")
plt.title("Trajectory of Robot")
plt.legend(("True","Dead Reckoning","EKF"))

plt.figure()
plt.plot(robot_true_pose[0:filter_till,0],robot_true_pose[0:filter_till,1])
plt.plot(dead_reckon_pose[0:filter_till,0], dead_reckon_pose[0:filter_till,1])
plt.plot(t_store[:],mean_store[:,0])
plt.xlabel("Time")
plt.ylabel("Global X Position")
plt.legend(("True", "Dead Reckon","Filter"))
plt.title("X")

plt.figure()
plt.plot(robot_true_pose[0:filter_till,0],robot_true_pose[0:filter_till,2])
plt.plot(dead_reckon_pose[0:filter_till,0], dead_reckon_pose[0:filter_till,2])
plt.plot(t_store[:],mean_store[:,1])
plt.xlabel("Time")
plt.ylabel("Global Y Position")
plt.legend(("True", "Dead Reckon","Filter"))
plt.title("Y")

plt.figure()
plt.plot(robot_true_pose[0:filter_till,0],robot_true_pose[0:filter_till,3])
plt.plot(dead_reckon_pose[0:filter_till,0], dead_reckon_pose[0:filter_till,3])
plt.plot(t_store[:],mean_store[:,2])
plt.xlabel("Time")
plt.ylabel(r"$\theta$")
plt.legend(("True", "Dead Reckon","Filter"))
plt.title(r"$\theta$")

plt.show()
