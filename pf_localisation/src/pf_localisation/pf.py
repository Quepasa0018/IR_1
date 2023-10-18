from geometry_msgs.msg import Pose, PoseArray, Quaternion, Point
from .pf_base import PFLocaliserBase
import math
from .util import rotateQuaternion, getHeading
import random
import numpy as np

class PFLocaliser(PFLocaliserBase):
    def __init__(self):
        super(PFLocaliser, self).__init__()
        self.num_poses = 500  # Set number of particles
        self.latest_odom_x = 0
        self.latest_odom_y = 0
        self.latest_odom_heading = 0
        self.NUMBER_PREDICTED_READINGS = 20  # Sensor model parameters from second code

    def initialise_particle_cloud(self, initialpose):
        intX = initialpose.pose.pose.position.x
        intY = initialpose.pose.pose.position.y
        poseArray = PoseArray()
        for i in range(self.num_poses):
            x = random.gauss(intX, 1)
            y = random.gauss(intY, 1)
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            rotation = random.gauss(0, math.pi)
            quat_msg = Quaternion(0, 0, 0, 1)  # Fixed quaternion initialization
            myRotateQuaternion = rotateQuaternion(quat_msg, rotation)
            pose.orientation = myRotateQuaternion
            poseArray.poses.append(pose)
        return poseArray

    def update_particle_cloud(self, scan):
        prev_odom_list = [
            self.prev_odom_x == self.latest_odom_x,
            self.prev_odom_y == self.latest_odom_y,
            self.prev_odom_heading == self.latest_odom_heading
        ]
        if all(prev_odom_list):
            return
        self.latest_odom_x = self.prev_odom_x
        self.latest_odom_y = self.prev_odom_y
        self.latest_odom_heading = self.prev_odom_heading
        pose = self.particlecloud
        weights = []
        for i, p in enumerate(pose.poses):
            likhweight = self.sensor_model.get_weight(scan, p)
            weights.append(likhweight)
        total = sum(weights)
        norm = [w / total for w in weights]
        M = self.num_poses
        cdf = []
        for i, n in enumerate(norm):
            cdf.append(n if i == 0 else cdf[-1] + n)
        thold = 1 / M
        u = random.uniform(0, thold)
        i = 0
        poseArray = PoseArray()
        for _ in range(M):
            while u > cdf[i]:
                i = i + 1
            particles = self.particlecloud.poses[i]
            x = particles.position.x
            y = particles.position.y
            q = particles.orientation
            t = getHeading(q)
            rx = random.gauss(x, norm[i])
            ry = random.gauss(y, norm[i])
            rt = random.gauss(t, norm[i])
            rPoint = Point(rx, ry, 0.0)
            rotateQ = rotateQuaternion(q, rt - t)
            newPose = Pose(rPoint, rotateQ)
            u = u + thold
            poseArray.poses.append(newPose)
        self.particlecloud = poseArray

    def estimate_pose(self):
        particles = self.particlecloud.poses
        particles_sorted = [[i, particles[i].orientation.w] for i in range(len(particles))]
        particles_sorted = sorted(particles_sorted, key=lambda x: x[1], reverse=True)
        indexs = [i[0] for i in particles_sorted]
        best_particles = [particles[indexs[i]] for i in range(len(indexs) // 2)]
        xpos, ypos, xor, yor, zor, wor = [], [], [], [], [], []
        for particle in best_particles:
            xpos.append(particle.position.x)
            ypos.append(particle.position.y)
            xor.append(particle.orientation.x)
            yor.append(particle.orientation.y)
            zor.append(particle.orientation.z)
            wor.append(particle.orientation.w)
        est_pose = Pose()
        est_pose.position.x = sum(xpos) / len(xpos)
        est_pose.position.y = sum(ypos) / len(ypos)
        est_pose.orientation.x = sum(xor) / len(xor)
        est_pose.orientation.y = sum(yor) / len(yor)
        est_pose.orientation.z = sum(zor) / len(zor)
        est_pose.orientation.w = sum(wor) / len(wor)
        return est_pose
