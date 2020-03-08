# -*- coding: utf-8 -*-
"""
Created on Web Feb 19, 2020
@author: amy-f
"""
import time as t
import datetime as dt
import matplotlib.pyplot as plt

"""
       Stopwatch class to calculate and analyse the time taken to execute a given step
"""


class Stopwatch:
    def __init__(self):
        self.final_time = 0
        self.cur_time = 0
        self.start_time = 0
        self.laps_time = []
        self.steps = []
        self.units = 'seconds'

    def start(self):
        self.start_time = self.cur_time = t.time()

    def stop(self):
        self.final_time = t.time() - self.start_time

    def start_new_lap(self, step):
        new_time = t.time()
        lap_time = new_time - self.cur_time
        self.laps_time.append(lap_time)
        self.steps.append(step)
        print("Step time : ", dt.timedelta(seconds=lap_time))
        self.cur_time = new_time

    def create_graph(self):
        fig, ax = plt.subplots()

        # laps_and_steps = [tuple(self.steps[i], self.laps_time[i]) for i in range(len(self.steps))]
        # print(laps_and_steps)

        ax.barh(self.steps, self.laps_time, align='center')
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Time (secs)')

        plt.show()

    def create_mean_graph(self):
        fig, ax = plt.subplots()

        laps_and_steps = [tuple(self.steps[i], self.laps_time[i]) for i in range(len(self.steps))]

        ax.barh(self.steps, self.laps_time, align='center')
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Time (secs)')

        plt.show()

    def to_string(self):
        print("Elapsed time : ", dt.timedelta(seconds=self.final_time))
