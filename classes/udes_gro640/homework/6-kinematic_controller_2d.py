#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 28

@author: Simon Chamorro CHAS2436
------------------------------------

Devoir #6 b)
Null space kinematic controller
"""

import numpy as np
import matplotlib.pyplot as plt


class NlinkArm():
    def __init__(self, dof=5, q=[np.pi/4, np.pi/4, 0, -np.pi/2, 0], links=[1, 1, 1, 1, 1], make_plot=True):
        self.make_plot = make_plot
        self.links = links
        self.q = q
        self.k = 1
        self.dof = dof
        self.r = np.zeros([len(q), 2])
        self.forward_kinematics()
        if self.make_plot:
            plt.figure()

    def set_q(self, q):
        self.q = q
        forward_kinematics()

    def forward_kinematics(self):
    	# Compute position for each joint
        for i in range(len(self.q)):
            self.r[i][0] = sum([self.links[j]*np.sin(sum([self.q[jj] for jj in range(j + 1)])) for j in range(i + 1)])
            self.r[i][1] = sum([self.links[j]*np.cos(sum([self.q[jj] for jj in range(j + 1)])) for j in range(i + 1)])

        # Update plot
        if self.make_plot:
            self.print_state()
            self.plot()

    def plot(self):
        plt.clf()
        plt.axis([-6, 6, -6, 6])
        x = [0]
        y = [0]
        for i in range(self.dof):
            x.append(self.r[i][0])
            y.append(self.r[i][1])
        plt.plot(x, y, color='blue')
        plt.scatter(x, y, color='red')
        plt.scatter(x[-1], y[-1], color='green')
        plt.draw()
        # plt.pause(0.03)
        plt.show()

    def print_state(self):
        print("q: " + str(self.q))
        print("r: " + str(self.r[-1]))

    
arm = NlinkArm()
# TODO : Add inverse kinematics


