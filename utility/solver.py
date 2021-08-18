from pulp import LpVariable
from pulp import LpProblem
from pulp import lpSum
from pulp import value
from pulp import LpStatus

from itertools import product


def get_solution(input):
    # Solution a sudoku puzzle and return the solution

    # Create coordinates and boxes
    rows = cols = vals = range(1, 10)
    coordinates = list(product(rows, cols))
    boxes = []

    # Initialize the prob object
    prob = LpProblem('Sudoku')

    # Create decision variables
    decisions = LpVariable.dicts('decision', (rows, cols, vals), cat='Binary')

    # Fill in the constraints
    ## 1. Each coordinate can take only 1 value
    ## Fix the row and column number, iterate through each value and make sure the sum is 1
    for row, col in coordinates:
        constraint = lpSum([decisions[row][col][val] for val in vals]) == 1
        prob += constraint
    ## 2. Each value appears only once per row
    ## Fix the column number and value, iterate through each row and make sure the sum is 1
    for val in vals:
        for col in cols:
            constraint = lpSum([decisions[row][col][val] for row in rows]) == 1
            prob += constraint
    ## 3. Each value appears only once per column
    ## Fix the row number and value, iterate through each column and make sure the sum is 1
    for val in vals:
        for row in rows:
            constraint = lpSum([decisions[row][col][val] for col in cols]) == 1
            prob += constraint
    ## 3. Each value appear only once per box
    ## Fix the value, iterate through the coordinates in each box
    for val in vals:
        for box in boxes:
            constraint = lpSum([decisions[coordinate[0]][coordinate[1]][val] for coordinate in box]) == 1
            prob += constraint

    for (c, r, v) in input:
        constraint = decisions[c][r][v] == 1
        prob += constraint

    # The problem data is written to an .lp file
    prob.writeLP('Sudoku.lp')

    # The problem is solved using PuLP's choice of solver
    prob.solve()

    # Return the final result
    solution = []
    for r in rows:
        for c in cols:
            for v in vals:
                if value(decisions[r][c][v]) == 1:
                    solution.append((r, c, v))
    return solution
