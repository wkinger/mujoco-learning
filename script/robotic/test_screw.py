import modern_robotics as mr
import numpy as np
# bionic robot
if __name__ == "__main__":
    d2 = 263
    d3 = 292
    a = 17
    M = np.array([[1, 0,  0, d2 + d3],
                    [ 0, 0,  1, 0],
                    [ 0, -1, 0, 0],
                    [ 0, 0,  0, 1]])
    Slist = np.array([[0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, a, -d2, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, -d2 - d3, 0],
                    [0, 1, 0, 0, 0, d2 + d3]]).T
    thetalist = np.array([0, 0, 0, 0, 0, 0, 0])
    T = mr.FKinSpace(M, Slist, thetalist)
    print(f"T {T}")


