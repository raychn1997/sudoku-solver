import pandas as pd


def get_dataset():
    df = pd.read_csv('data/data.csv', header=None)[0].to_list()
    return df


def get_board(data):
    # Take a list of (r, c, v) points and draw a sudoku board
    text = ''
    for r in range(1, 10):
        if r in [1, 4, 7]:
            text += ('+-------+-------+-------+\n')
        for c in range(1, 10):
            if c in [1, 4, 7]:
                text += ('| ')
            number = ' '
            for v in range(1, 10):
                if (r, c, v) in data:
                    number = str(v)
                    break
            text += number + ' '
            if c == 9:
                text += '|\n'
    text += ('+-------+-------+-------+\n')
    return text


def parse_board(board):
    # Take a sudoku board and return a list of (r, c, v) points
    rows = (1, 2, 3, 5, 6, 7, 9, 10, 11)
    cols = (2, 4, 6, 10, 12, 14, 18, 20, 22)
    board = board.split('\n')
    data = []
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            v = board[r][c]
            if v != ' ':
                data.append((ri+1, ci+1, int(v)))
    return data


def convert_to_data(text):
    # Take a flat string like 070000043040009610800634900094052000358460020000800530080070091902100005007040802
    # And convert it to our data format
    data = []
    for r in range(1, 10):
        for c in range(1, 10):
            index = (r-1)*9 + (c-1)
            if text[index] != '0':
                data.append((r, c, int(text[index])))
    return data