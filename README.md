# Comparison of EKF and UKF for Mobile Robot Localization on the UTIAS Dataset

Mobile robot localization is the problem
of determining the pose of a robot relative to a given
map of the environment. In other words, with knowledge
of the map of the environment, localization places the robot
in the map, with the help of sensor data, motion models and
control inputs. Localization is naturally very important for
mobile robot navigation. The Gaussian filtering algorithms
are one of the categories of algorithms used for mobile
robot localization. For this project, I have implemented the
Extended Kalman Filter (EKF) and the Unscented Kalman
Filter (UKF) algorithms for mobile robot localization and I
compared their performance. Both of these algorithms are
Gaussian filtering algorithms, however, ExKF is based on
the idea of linearizing whereas UKF is an integration-based
Gaussian filtering algorithm.

I have implemented local, passive localization in a static
environment for a single robot. Local localization is the
case of localization when the initial pose of the robot is
known. Passive localization means that the control law of the
robot’s motion is not designed specifically for localization;
the motion might be random. I have implemented these algorithms
on the UTIAS Multi-Robot Cooperative Localization
and Mapping Dataset. This dataset contains the following
relevant data for 5 robots:
* Odometry: The odometry data has the forward and
angular velocity of the mobile robot.
* Measurement data (range and bearing): This contains
estimates of distances and angles at which the landmarks
were detected by the robot.
Accurate groundtruth data: The groundtruth data is
the true pose of the robot as it moves around in the
environment.
* Position and identity of landmarks: The true positions
of all the landmarks is provided. Further, each landmark
is linked with a unique barcode which the robot detects
while obtaining the measurement data. Thus, the robot
can recognise the landmark being observed and place
itself on the map accordingly.

This dataset includes data for 5 robots. Since I am only
performing localization for one robot, all measurement data
corresponding to other robots will be ignored. This can be
done because like the landmarks, all robots also have a
unique barcode assigne to it, hence the robot can detect
whether it is detecting a landmark or another robot.

For further details, please refer to the project report.
