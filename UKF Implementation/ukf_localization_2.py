"""
Here, I have implemented an Unscented Kalman 
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
from scipy import linalg
import pickle

lm_filepath = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\ds0_Landmark_Groundtruth.dat"
groundtruth_filepath = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\ds0_Groundtruth.dat"
barcodes_filepath = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\ds0_Barcodes.dat"
measure_filepath = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\ds0_Measurement.dat"
odom_filepath = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\ds0_Odometry.dat"
dim = 3  
num_pts = 2*dim+1 

beta = 2
alpha = 10**-5
lamda = (alpha**2)*dim-dim

std_dx = 0.004 
std_dy = 0.004 
std_dtheta = 0.0085 

std_r = 0.002 
std_b  = 0.0085

lm_mat = np.zeros((15,3))
lm_std = np.zeros((15,3))
i = 0
lm_file = open(lm_filepath, "r")
for line in lm_file:
    if not line.startswith("#"):
        values = line.split()
        lm_mat[i,:] = np.array([values[0],values[1],values[2]])
        lm_std[i,:] = np.array([values[0],values[3], values[4]])
        i = i+1
lm_file.close()

def get_std(cov):
    """Get square root of diagonals (standard deviations) from covariances """
    
    N, d, d = cov.shape
    std_devs = np.zeros((N, d))
    for ii in range(N):   
        std_devs[ii, :] = np.sqrt(np.diag(cov[ii, :, :]))
    return std_devs

def landmark_data(lm_dict):
    file = open(lm_filepath, "r")
    for line in file:
        if not line.startswith("#"):
            values = line.split()
            lm_dict.update({float(values[0]) : [float(values[1]), float(values[2])]})

    file.close()

def barcode_data(barcodes_dict):
    file = open(barcodes_filepath, "r")
    for line in file:
        if not line.startswith("#"):
            values = line.split()

            key = int(values[1])
            subject = int(values[0])
            # landmarks have numbers 6 -> 20
            if subject >= 6:
                # key is the barcode number
                # element if the subject number
                barcodes_dict.update({key : subject})

    file.close()

def measurement_data(measurement_mat):
    file = open(measure_filepath, "r")
    for line in file:
        if not line.startswith("#"):
            values = line.split()
            meas = [float(values[0]), int(values[1]), float(values[2]), float(values[3])]
            measurement_mat.append(meas)
    file.close()

def odometry_data(odometry_mat):
    file = open(odom_filepath, "r")
    for line in file:
        if not line.startswith("#"):
            values = line.split()
            odom = [float(values[0]), float(values[1]), float(values[2])]
            odometry_mat.append(odom)
    file.close()

def ground_truth_data():
    
    truth_file_path = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\ds0_Groundtruth.dat"
    l = 0
    robot_true_pose = np.zeros((87676,4))
    truth_file = open(truth_file_path, "r")
    for line in truth_file:
        if not line.startswith("#"):
            values = line.split()
            robot_true_pose[l,:] = np.array([[values[0],values[1], values[2], values[3]]])
            l = l+1

    truth_file.close()

    return robot_true_pose

def dead_reck_data():
    dr_file_path = r"C:\Users\aarishs\Downloads\XYZ\Implementation 2\dead_reckon_data"
    file = open(dr_file_path, "rb")

    dead_reck = pickle.load(file)

    file.close()
    return np.array(dead_reck[:,:])

def robot_motion(odom, init_pose, dt):
    """
    Function that returns robot's motion
    based on the odometry
    """
    x0 = init_pose[0]
    y0 = init_pose[1]
    theta0 = init_pose[2]
    v = odom[0]
    w = odom[1]

    dx = v*np.cos(theta0)*dt
    dy = v*np.sin(theta0)*dt
    dtheta = w*dt

    x1 = x0 + dx
    y1 = y0 + dy 
    theta1 = theta0 + dtheta

    n = theta1/(2*np.pi)
    m = n - int(n)
    theta1 = m*2*np.pi

    if theta1 > np.pi:
        theta1 -= 2*np.pi
    elif theta1 < -np.pi:
        theta1 += 2*np.pi

    final_pose = np.array([x1,y1,theta1])

    return final_pose

def obsv_model(landmark, pose):

    xr = pose[0]
    yr = pose[1]
    thetar = pose[2]
    lm_x = landmark[0]
    lm_y = landmark[1]
    range = np.sqrt((xr-lm_x)**2 + (yr-lm_y)**2)
    bearing = np.arctan2(lm_y-yr, lm_x-xr) - thetar
    if bearing > np.pi:
        bearing -= 2*np.pi
    elif bearing < -np.pi:
        bearing += 2*np.pi

    return [range, bearing]

class UKF(object):

    def __init__(self):

        self.wm = lamda/(dim+lamda)
        self.wc = lamda/(dim+lamda) + (1 - (alpha)**2 + beta)
        self.w = 1/(2*(dim+lamda))

        self.R = np.array([[std_dx**2, 0, 0],
                            [0, std_dy**2, 0],
                            [0, 0, std_dtheta**2]])
        self.Q = np.array([[std_r**2, 0],
                          [0, std_b**2]])


    def wrap_two_pi(self, angle):
        """ wraps and angle between 0 and 2pi """

        num_rev = angle/(2*np.pi)
        rev_frac = num_rev - int(num_rev)
        angle = rev_frac*2*np.pi

        return angle

    def wrap_pi(self, angle):
        """ wraps angle pi to -pi """
        angle = self.wrap_two_pi(angle)

        if angle > np.pi:
            angle -= 2*np.pi
        elif angle < -np.pi:
            angle += 2*np.pi

        return angle


    def unscented_transform(self, mean, cov):

        L = np.linalg.cholesky(cov)
        pts = np.array([mean])

        for i in range(0, dim):
            pt = mean + np.sqrt(dim+lamda)*L[i,:]
            pts = np.append(pts, [pt], axis=0)

        for i in range(0, dim):
            pt = mean - np.sqrt(dim+lamda)*L[i,:]  
            pts = np.append(pts, [pt], axis=0)

        return pts


    def propagator(self, pts, odom, dt):

        propagated_pt = []
        for i in range(0, num_pts):

            temp = robot_motion(odom, pts[0,:], dt)
            propagated_pt.append(temp)

        propagated_pts = np.array(propagated_pt)
        return propagated_pts


    def predict_mean(self, sigma_mat_star):

        mu_bar = np.array([0, 0, 0])

        for i in range(0, num_pts):
            if i == 0:
                w_m =  self.wm
            else:
                w_m = self.w

            mu_bar = mu_bar + w_m * sigma_mat_star[i,:]

        return mu_bar


    def predict_covariance(self, mu_bar, sigma_mat_star):

        cov_mat_bar = np.zeros((dim,dim))

        for i in range(0, num_pts):

            if i == 0:
                w_c = self.wc
            else:
                w_c = self.w

            delta1 = sigma_mat_star[i,:] - mu_bar
            delta1 = delta1[np.newaxis]  # 1x3
            delta2 = delta1.T   # 3x1

            cov_mat_bar = np.add(cov_mat_bar, w_c * np.dot(delta2, delta1))

        cov_mat_bar = np.add(cov_mat_bar, self.R)

        return cov_mat_bar


    def observation_sigma(self, sigma_mat_new, landmark):

        obs = []

        for i in range (0, num_pts):
            meas = obsv_model(landmark, sigma_mat_new[i,:])
            obs.append(meas)

        obs_mat = np.array(obs)
        return obs_mat


    def predicted_observation(self, obs_mat):

        z_hat = np.array([0, 0])

        for i in range(0, num_pts):

            if i == 0:
                w_m = self.wm
            else:
                w_m = self.w

            z_hat = z_hat + w_m * obs_mat[i,:]

        z_hat[1] = self.wrap_pi(z_hat[1])

        return z_hat


    def uncertainty(self, obs_mat, z_hat):

        uncert_mat = np.zeros((2,2))

        for i in range(0, num_pts):

            if i == 0:
                w_c = self.wc
            else:
                w_c = self.w

            delta1 = obs_mat[i,:] - z_hat
            delta1 = delta1[np.newaxis] # 1x2
            delta2 = delta1.T   #2x1

            uncert_mat = np.add(uncert_mat, w_c * np.dot(delta2, delta1))

        uncert_mat = uncert_mat + w_c*np.dot(delta2, delta1) + self.Q

        return uncert_mat


    def cross_covariance(self, sigma_mat_new, mu_bar, obs_mat, z_hat):

        cross_cov_mat = np.zeros((3,2))

        for i in range(0, num_pts):

            if i == 0:
                w_c = self.wc
            else:
                w_c = self.w

            delta_states = sigma_mat_new[i,:] - mu_bar
            delta_states = delta_states[np.newaxis] # 1x3
            delta_states = delta_states.T # 3x1

            delta_obs = obs_mat[i,:] - z_hat
            delta_obs = delta_obs[np.newaxis] # 1x2

            cross_cov_mat = cross_cov_mat + w_c * np.dot(delta_states, delta_obs)

        return cross_cov_mat


    def kalman_gain(self, cross_cov_mat, uncert_mat):

        uncert_inv = np.linalg.inv(uncert_mat)

        kal_gain = np.dot(cross_cov_mat, uncert_inv)

        return kal_gain


    def update_mean(self, mu_bar, kal_gain, z, z_hat):

        delta_meas = z - z_hat
        delta_meas = delta_meas[np.newaxis] # 1x2
        delta_meas = delta_meas.T # 2x1

        new_mean = mu_bar + np.dot(kal_gain, delta_meas).T

        new_mean = new_mean.ravel()

        return new_mean


    def update_covariance(self, cov_mat_bar, kal_gain, uncert_mat):

        new_cov_mat = cov_mat_bar - np.dot(np.dot(kal_gain, uncert_mat), kal_gain.T)

        return new_cov_mat


    def unscented_kalman_filter(self, mu, cov_mat, u, meas, dt):

        if dt != None:
            pts = self.unscented_transform(mu, cov_mat)
            propagated_pts = self.propagator(pts, u, dt)
            pred_mean = self.predict_mean(propagated_pts)
            pred_cov = self.predict_covariance(pred_mean, propagated_pts)

            if np.all(meas) == None:
                return pred_mean, pred_cov

        else:
            pred_mean = mu
            pred_cov = cov_mat

        landmark = [meas[0], meas[1]]
        z = [meas[2], meas[3]]

        sigma_mat_new = self.unscented_transform(pred_mean, pred_cov)
        obs_mat = self.observation_sigma(sigma_mat_new, landmark)
        z_hat = self.predicted_observation(obs_mat)
        uncert_mat = self.uncertainty(obs_mat, z_hat)
        cross_cov_mat = self.cross_covariance(sigma_mat_new, pred_mean, obs_mat, z_hat)
        kal_gain = self.kalman_gain(cross_cov_mat, uncert_mat)
        new_mean = self.update_mean(pred_mean, kal_gain, z, z_hat)
        new_cov_mat = self.update_covariance(pred_cov, kal_gain, uncert_mat)

        return new_mean, new_cov_mat

class Localization(object):

    def __init__(self):

        self.mean = np.array([1.29812900, 1.88315210, 2.82870000])

        self.cov = np.array([[.10, 0, 0],
                                 [0, .10, 0],
                                 [0, 0, .20]])


        self.lm_dict = {}
        self.lm_barcodes = {}
        self.odometry = np.zeros((95818, 3), dtype=float)
        self.measurement = np.zeros((7720, 4), dtype=float)
        self.length = 95818
        self.num_z = 0
        self.lm_detected = False
        self.use_meas = False
        self.barcode = None
        self.odom1 = 0
        self.odom2 = 1
        self.meas_index = 0

        landmark_data(self.lm_dict)
        barcode_data(self.lm_barcodes)
        meas = []
        measurement_data(meas)
        self.measurement = np.array(meas)
        odom = []
        odometry_data(odom)
        self.odometry = np.array(odom)
        self.ground_truth = ground_truth_data()
        self.dead_reck = dead_reck_data()

    def ukf_localization(self, filter_till):

        ukf = UKF()
        curr_time = 0
        mean = self.mean
        cov = self.cov
        t0_odom = self.odometry[self.odom1, 0]
        t_store = np.array([[t0_odom]])

        mean_store = np.array([[self.mean[0],self.mean[1],self.mean[2]]])
        cov_store = np.array([cov])

        while(self.odom1 != filter_till):

            self.lm_detected = False
            self.use_meas = False

            # get odometry time stamps
            t0_odom = self.odometry[self.odom1, 0]
            t1_odom = self.odometry[self.odom2, 0] 

            if self.meas_index < 7720:
                t_data = self.measurement[self.meas_index, 0] 

            else:
                t_data = 0


            current_odom = [self.odometry[self.odom1, 1], self.odometry[self.odom1, 2]]
            if t0_odom <= t_data and t_data <= t1_odom:
                code = self.measurement[self.meas_index, 1]
                if code in self.lm_barcodes.keys():
                    self.barcode = code
                    self.lm_detected = True

                else:
                    self.meas_index += 1
                    self.lm_detected = False

            else:
                self.lm_detected = False

                dt = t0_odom - curr_time
                curr_time = t0_odom

                mean, cov = ukf.unscented_kalman_filter(mean, cov, current_odom, None, dt)
                self.odom1 += 1
                self.odom2 += 1

            if self.lm_detected == True:

                if self.meas_index == 0:
                    self.use_meas = True

                elif self.measurement[self.meas_index, 0] != self.measurement[self.meas_index-1, 0]:
                    self.use_meas = True

                elif self.measurement[self.meas_index, 0] == self.measurement[self.meas_index-1, 0]:
                    self.use_meas = False
                    self.meas_index += 1

            if self.use_meas == True:
                    subject = self.lm_barcodes[self.barcode]
                
                    dt = t_data - curr_time
                    curr_time = t_data

                    lm_pos = self.lm_dict[subject]

                    r = self.measurement[self.meas_index, 2]
                    b = self.measurement[self.meas_index, 3]
                    z = np.array([lm_pos[0], lm_pos[1], r, b])

                    self.meas_index += 1

                    mean, cov = ukf.unscented_kalman_filter(mean, cov, current_odom, z, dt)
            mean_ = mean.reshape(1,3)
            mean_store = np.append(mean_store,mean_,axis=0)
            
            t_store = np.append(t_store,t1_odom)

            cov_store = np.append(cov_store,np.array([cov]),axis=0)

        std_devs = get_std(cov_store)

        MSE_x = 0
        MSE_y = 0
        MSE_theta = 0
        num = 0
        m_used = []
        n_used = []

        for m in range(t_store.shape[0]):
            for n in range(self.ground_truth[0:filter_till,0].shape[0]):
                if self.ground_truth[n,0] - 0.01 <= t_store[m] <= self.ground_truth[n,0] + 0.01:
                    
                    if n not in n_used and m not in m_used:
                        MSE_x += (self.ground_truth[n,1]-mean_store[m,0])**2
                        MSE_y += (self.ground_truth[n,2]-mean_store[m,1])**2
                        MSE_theta += (self.ground_truth[n,3]-mean_store[m,2])**2
                        num += 1

                    m_used += [m]
                    n_used += [n]

        MSE_x = MSE_x/num
        MSE_y = MSE_y/num
        MSE_theta = MSE_theta/num

        print(MSE_x,MSE_y,MSE_theta,num)

        avg_std = np.mean(std_devs,axis=0)
        print(avg_std)

        l=1000

        plt.figure()
        plt.plot(mean_store[:,0], mean_store[:,1])
        plt.plot(self.dead_reck[0:filter_till-l,1], self.dead_reck[0:filter_till-l,2])
        plt.plot(self.ground_truth[0:filter_till-l,1], self.ground_truth[0:filter_till-l,2])
        for i in range(lm_mat.shape[0]):
                plt.plot(lm_mat[i,1], lm_mat[i,2],"x",color="red")
                plt.annotate(str(lm_mat[i,0]),(lm_mat[i,1],lm_mat[i,2]))
        plt.xlabel("Global X Positon")
        plt.ylabel("Global Y Position")
        plt.title("Trajectory of Robot")
        plt.legend(("UKF","Dead Reckoning","Ground Truth","Landmarks"))

        plt.figure()
        plt.plot(self.ground_truth[0:filter_till-l,0],self.ground_truth[0:filter_till-l,1])
        plt.plot(self.dead_reck[0:filter_till-l,0], self.dead_reck[0:filter_till-l,1])
        plt.plot(t_store[:],mean_store[:,0])
        plt.fill_between(t_store[:],mean_store[:,0]-2*std_devs[:,0],mean_store[:,0]+2*std_devs[:,0],color="red",alpha=0.3)
        plt.xlabel("Time")
        plt.ylabel("Global X Position")
        plt.legend(("True", "Dead Reckon","Filter",r"2 $\sigma$"))
        plt.title("X")

        plt.figure()
        plt.plot(self.ground_truth[0:filter_till-l,0],self.ground_truth[0:filter_till-l,2])
        plt.plot(self.dead_reck[0:filter_till-l,0], self.dead_reck[0:filter_till-l,2])
        plt.plot(t_store[:],mean_store[:,1])
        plt.fill_between(t_store[:],mean_store[:,1]-2*std_devs[:,1],mean_store[:,1]+2*std_devs[:,1],color="red",alpha=0.3)
        plt.xlabel("Time")
        plt.ylabel("Global Y Position")
        plt.legend(("True", "Dead Reckon","Filter",r"2 $\sigma$"))
        plt.title("Y")

        plt.figure()
        plt.plot(self.ground_truth[0:filter_till-l,0],self.ground_truth[0:filter_till-l,3])
        plt.plot(self.dead_reck[0:filter_till-l,0], self.dead_reck[0:filter_till-l,3])
        plt.plot(t_store[:],mean_store[:,2])
        plt.fill_between(t_store[:],mean_store[:,2]-2*std_devs[:,2],mean_store[:,2]+2*std_devs[:,2],color="red",alpha=0.3)
        plt.xlabel("Time")
        plt.ylabel(r"\theta")
        plt.legend(("True", "Dead Reckon","Filter",r"2 $\sigma$"))
        plt.title("Theta")

        plt.show()

        return mean_store, t_store

filter_till = 20000
robot_local = Localization()
robot_local.ukf_localization(filter_till)