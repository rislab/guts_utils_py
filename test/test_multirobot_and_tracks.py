import pickle
import numpy as np
import scipy.linalg
import pyproj
import matplotlib.pyplot as plt
import math
import time

from guts_utils import GUTS

from grid import Grid, Cell

num_flyovers = 0 # number of times we enter the cell that contains the target
width = 7 # number of cells in width
height = 7 # number of cells in height
grid_resolution = 15 # meters
y_confidence = 0.005 # confidence for a negative observation (i.e., we don't see a target)

# Updates the X and Y matrices to incorporate a target detection
def append_target_observation(X, Y, grid, target_loc, target_confidence):
    global width
    global height

    idx = grid.point_to_index(target_loc)
    X = np.vstack((X, np.zeros((1,width*height))))
    Y = np.vstack((Y, np.array([1, target_confidence]).reshape(1,2)))
    X[-1,idx] = 1.
    return (X, Y)

# Calculates distance between two points
def calculate_point_distance(p1, p2):
    return math.sqrt((p1[0,0]-p2[0,0])**2 + (p1[1,0]-p2[1,0])**2)

# Updates the X and Y matrices to execute an action between a start and end point
def execute_action(start_pt, end_pt, grid, X, Y, target_locs, target_confidence):
    global num_flyovers
    global width
    global height
    global y_confidence

    success, cells = grid.raytrace(start_pt, end_pt);

    if success == False:
        raise Exception("Failed to raytrace")

    # Go through each cell and update X and Y
    for c in cells:
        if len(X) == 0:
            X = np.zeros((1,width*height))
            Y = np.array([0, y_confidence]).reshape(1,2)
        else:
            X = np.vstack((X, np.zeros((1,width*height))))
            Y = np.vstack((Y, np.array([0, y_confidence]).reshape(1,2)))
            grid.set(c, 1.0) # mark cell as visited
            idx = grid.cell_to_index(c)
            X[-1,idx] = 1.

        robot_idx = grid.cell_to_index(c)

        for target_loc in target_locs:
            target_idx = grid.point_to_index(target_loc)

            if robot_idx == target_idx:
                num_flyovers = num_flyovers + 1

            # Places robot position at the center of the cell
            robot_pt = grid.cell_to_point(c) + np.array([grid_resolution/2, grid_resolution/2]).reshape((2,1))
            target_pt = target_loc

            # Assume we can see the target within two grid cells
            if calculate_point_distance(robot_pt, target_pt) < grid_resolution*2:
                (X, Y) = append_target_observation(X, Y, grid, target_loc, target_confidence)
    return (X, Y)

# Evaluate all actions to determine which is the next best one
def get_next_best_action(end_pt, grid, guts):
    global width
    global height
    global grid_resolution

    curr_min = 1e9
    min_row = 0
    min_col = 0
    for row in range(0, height):
        for col in range(0, width):
            pt = grid.cell_to_point(Cell(row, col))
            curr_point = end_pt
            next_point = np.array([pt[0] + grid_resolution/2,
                                   pt[1] + grid_resolution/2]).reshape((2,1))
            score = guts.score(curr_point, next_point, grid)

            if curr_min > score:
                curr_min = score
                min_row = row
                min_col = col

    en = Cell(min_row, min_col)
    best_action = grid.cell_to_point(en).reshape((2,1)) + np.array([grid_resolution/2, grid_resolution/2]).reshape((2,1))
    return best_action

# Plot all target locations
def plot_target_locs(grid, locs):
    for i in range(0, len(locs)):
        plt.plot(locs[i][0], locs[i][1], 'b+', mew=3, ms=15)

# Plot all robot actions
def plot_actions(actions, grid, color):

    start_pt = actions[0]

    for i in range(0, len(actions)):
        end_pt = actions[i]
        st = start_pt
        en = end_pt

        plt.plot([st[0], en[0]], [st[1], en[1]], color)
        start_pt = end_pt

# Set grid origin
origin = np.array([0.0, 0.0]). reshape((2,1))
grid = Grid(origin, width, height, grid_resolution);

# Initialize guts class instance
guts =  GUTS()

# Assume two robots here: start_pt1 and start_pt2
# First action for both robots is to remain in one cell
start_pt1 = np.array([grid_resolution/2, grid_resolution/2]).reshape((2,1))
start_pt2 = np.array([grid_resolution/2, grid_resolution/2]).reshape((2,1))
end_pt1 = np.array([grid_resolution/2, grid_resolution/2]).reshape((2,1))
end_pt2 = np.array([grid_resolution/2, grid_resolution/2]).reshape((2,1))

# List of actions for robot 1
actions1 = []
actions1.append(start_pt1)
actions1.append(end_pt1)

# List of actions for robot 2
actions2 = []
actions2.append(start_pt2)
actions2.append(end_pt2)

# Locations of targets. Select one or two targets
#target_locs = [np.array([67.5, 67.5]).reshape((2,1))] # One target
target_locs = [np.array([67.5, 67.5]).reshape((2,1)),
               np.array([30.5, 30.5]).reshape((2,1))] # Two targets

X = []
Y = []

# TODO: USER SHOULD SET THE target_confidence.
# Use a value between 0.005 and 1.0
#target_confidence = 0.005 # Noisiest, 148 for two targets, 83 for one target
#target_confidence = 0.2 # 99 for two targets, 54 for one target
target_confidence = 0.5 #  73 for two targets, 38 for one target
for i in range(0, 50):

    # One thing to note is that we are not incorporating
    # the other robot's next action in the calculation of
    # the reward for this example script.
    #
    # Adding this functionality amounts to a little extra
    # bookkeeping.

    # Execute action for robot 1, compute the posterior, and get the next best action
    (X, Y) = execute_action(start_pt1, end_pt1, grid, X, Y, target_locs, target_confidence)
    guts.posterior(X, Y)
    best_action1 = get_next_best_action(end_pt1, grid, guts)

    # Execute action for robot 2, compute the posterior, and get the next best action
    (X, Y) = execute_action(start_pt2, end_pt2, grid, X, Y, target_locs, target_confidence)
    guts.posterior(X, Y)
    best_action2 = get_next_best_action(end_pt2, grid, guts)

    # Keep track of all actions
    actions1.append(best_action1)
    actions2.append(best_action2)

    # Update start point with the current end point
    # Update end point with the best action found so far
    start_pt1 = end_pt1
    end_pt1 = best_action1

    start_pt2 = end_pt2
    end_pt2 = best_action2

    print('Completed action ' + str(i))

# Plot the actions
plot_actions(actions1, grid, 'r') # robot 1 is in red
plot_actions(actions2, grid, 'g') # robot 2 is in green

# Plot the targets
plot_target_locs(grid, target_locs)
plt.axis('equal')

# Report number of flyovers
print('Number of flyovers ' + str(num_flyovers))
plt.show()
