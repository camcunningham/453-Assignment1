import numpy as np

GAMMA = 0.85

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def initGrid(n):
    return np.zeros(shape=(n,n))

# Returns the index of the row and col for the vector
def getV(n, row, col):
    return (row) * n + col

def optimizeGrid(grid, aPosition, bPosition):
    # Run through each value in the grid
    rowCount = 0
    n = len(grid)
    a = initGrid(n * n)
    tempB = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            # First check if we are on position A or B
            if r == aPosition[0][0] and c == aPosition[0][1]:
                # We are at A, move to the other target with prob 1, reward = 10
                reward = 10
                tempB.append(reward)
                newPosition = getV(n, aPosition[1][0], aPosition[1][1])
                a[rowCount][newPosition] += GAMMA
                rowCount += 1
            elif r == bPosition[0][0] and c == bPosition[0][1]:
                # We are at B, move to the other target with prob 1, reward = 5
                reward = 5
                tempB.append(reward)
                newPosition = getV(n, bPosition[1][0], bPosition[1][1])
                a[rowCount][newPosition] += GAMMA
                rowCount += 1
            else:
                # Calculate the reward for going U, D, L, R (each with probability 1/4)
                upReward = 0 if r > 0 else -1
                downReward = 0 if r < (n-1) else -1
                leftReward = 0 if c > 0 else -1
                rightReward = 0 if c < (n-1) else -1
                reward = sum([upReward, downReward, leftReward, rightReward]) * 0.25
                tempB.append(reward)

                # Now calculate the coefficients for the equations for each value
                
                # For moving up, we look at grid[r-1][c]
                upPos = (r-1,c) if r > 0 else (r,c)
                # For moving down, we look at grid[r+1][c]
                downPos = (r+1,c) if r < (n-1) else (r,c)
                # For moving left, we look at grid[r][c-1]
                leftPos = (r,c-1) if c > 0 else (r,c)
                # For moving right, we look at grid[r][c+1]
                rightPos = (r,c+1) if c < (n-1) else (r,c) 

                # Get the indexes of the moves
                upV = getV(n, upPos[0], upPos[1])
                downV = getV(n, downPos[0], downPos[1])
                rightV = getV(n, rightPos[0], rightPos[1])
                leftV = getV(n, leftPos[0], leftPos[1])

                a[rowCount][rowCount] = 1

                # Add to matrix to solve
                if upV != rowCount:
                    a[rowCount][upV] += GAMMA * -0.25  
                else:
                    a[rowCount][upV] -= (GAMMA * 0.25)
                
                if downV != rowCount:
                    a[rowCount][downV] += GAMMA * -0.25 
                else:
                    a[rowCount][downV] -= (GAMMA * 0.25)
                
                if rightV != rowCount:
                    a[rowCount][rightV] += GAMMA * -0.25  
                else:
                    a[rowCount][rightV] -= (GAMMA * 0.25)

                if leftV != rowCount:
                    a[rowCount][leftV] += GAMMA * -0.25  
                else:
                    a[rowCount][leftV] -= (GAMMA * 0.25)

                # Used to tell which row we're on
                rowCount += 1

                
            
    # B contains the vector of rewards for each val
    b = np.array(tempB)
    return (a, b)

def solveGrid(A, b):
    return np.linalg.solve(A,b)


def main():
    n = 5
    # Coorinates for special locations
    A = [(0,1),(4,1)]
    B = [(0,3),(2,3)]
    grid = initGrid(n)
    # print(len(grid))
    A, b = optimizeGrid(grid, A, B)
    matprint(A)
    print(b)
    print(solveGrid(A,b))

    

if __name__ == "__main__":
    main()