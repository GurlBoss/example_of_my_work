"""
Author: Michal Mikeska
Reference:
Heuristic -  https://kartikkukreja.wordpress.com/2013/03/30/heuristic-function-for-reversiothello/
Task - https://cw.fel.cvut.cz/wiki/courses/b4b33rph/cviceni/reversi
Previous assigment from 2021L reversi tournament KUI

The next move of player
is based on minimax algorithm described here:
https://en.wikipedia.org/wiki/Minimax
Depth is set as 2 since there is 1 second limit for move.
The heuristic function of the move is inspired by:
https://kartikkukreja.wordpress.com/2013/03/30/heuristic-function-for-reversiothello/
"""
import copy
import numpy as np
import math

borders = -2
free_box = -1

# checking for the heuristic of the mobility
def mobility(player_moves, opponent_moves):
    if not player_moves and not opponent_moves:
        opponent_moves = 0
        player_moves = 0
    elif player_moves:
        opponent_moves = 0
    else:
        player_moves = 0
    if player_moves + opponent_moves != 0:
        return 100 * (player_moves - opponent_moves) / (player_moves + opponent_moves)
    else:
        return 0


# checking for the heuristic of the corners
def h_corners(board, player):
    score = [0, 0]
    board_size = len(board)
    dx = [0, board_size - 1, 0, board_size - 1]
    dy = [0, 0, board_size - 1, board_size - 1]
    for i in range(4):
        if board[dx[i]][dy[i]] == player.my_color:
            score[0] += 1
        elif board[dx[i]][dy[i]] == player.opponent_color:
            score[1] += 1
    if (score[0] + score[1]) != 0:
        ret = 100 * (score[0] - score[1]) / (score[0] + score[1])
    else:
        ret = 0
    return ret


# count all heuristic functions together
def heuristic_function(player, board):
    opponent_p = MyPlayer(player.opponent_color, player.my_color)
    scores = get_score(board, player.board_size, player.my_color, player.opponent_color)
    h_coin_parity = 100 * ((scores[0] - scores[1]) / sum(scores))
    h_mobility = mobility(len(player.get_possible_moves(board)),
                          len(opponent_p.get_possible_moves(board)))
    h_corn = h_corners(board, player)
    return h_corn + h_mobility + h_coin_parity

# get scores of the players
def get_score(board, board_size, color_p1, color_p2):
    stones = [0, 0]
    for i in range(0, board_size):
        for j in range(0, board_size):
            if board[i][i] == color_p1:
                stones[0] += 1
            if board[i][j] == color_p2:
                stones[1] += 1
    return stones

#generete next node based on previous parameters in minimax
def get_next_node(coords, board, next_player, node, depth, player, maximizing):
    board_copy = copy.deepcopy(board)
    next_player.play_move(board_copy, coords)
    node_copy = copy.copy(node)
    next_node = minmax(player, depth - 1, board_copy, node_copy, maximizing)
    return next_node

#setting node through minimax algorithm
def set_node(player, depth, board, node, maximizing):
    if maximizing:
        node.value = -math.inf
        next_player = player
        nxt_max = False
    else:
        node.value = math.inf
        next_player = MyPlayer(player.opponent_color, player.my_color)
        nxt_max = True
    next_coords = MyPlayer.get_possible_moves(next_player, board)
    if next_coords:
        for idx, coord in enumerate(next_coords):
            next_node = get_next_node(coord, board, next_player, node, depth, player, nxt_max)
            if (maximizing and next_node.value > node.value) or \
                    (not maximizing and next_node.value < node.value):
                node.value = next_node.value
                node.next_coord = coord
                if maximizing:
                    node.alpha = next_node.value
                else:
                    node.beta = next_node.value
                if node.alpha >= node.beta:
                    break
    else:
        node.value = heuristic_function(player, board)
        node.next_coord = None
    return node

#minimax for the player - maximizing the result for player
def minmax(player, depth, board, actual_node, maximizing):
    if depth == 0:
        actual_node.value = heuristic_function(player, board)
        actual_node.next_coord = None
        ret_node = actual_node
    else:
        ret_node = set_node(player, depth, board, actual_node, maximizing)
    return ret_node

#node for minimax
class Node:
    def __init__(self):
        self.value = None
        self.next_coord = None
        self.alpha = None
        self.beta = None


class MyPlayer:
    """
    Class MyPlayer represents player for the game Othello.
    """

    def __init__(self, my_color, opponent_color):
        self.name = 'mikesmi4'  # username studenta
        self.my_color = my_color
        self.opponent_color = opponent_color

    #get the best move based on minimax
    def move(self, board):
        self.board_size = len(board)
        node = Node()
        node.alpha = -math.inf
        node.beta = math.inf
        depth = 2
        node = minmax(self, depth, board, node, True)
        return int(node.next_coord[0]), int(node.next_coord[1])

    # find all possible coord based on the possible vectors -
    # e.g. row,columns, diagonals
    def find_possible_coord(self, possible_vectors):
        possible_position = []
        for vector in possible_vectors:
            if self.my_color in vector[:, 0] and self.opponent_color in vector[:, 0]:
                my_color_starts = False
                theoretical_position = None
                opponent_color_starts = False
                opponent_counter = 0
                for idx, box in enumerate(vector[:, 0]):
                    if box == free_box:
                        if my_color_starts and opponent_counter > 0:
                            possible_position.append((vector[idx, 1], vector[idx, 2]))
                        my_color_starts = False
                        opponent_color_starts = False
                        opponent_counter = 0
                        theoretical_position = (vector[idx, 1], vector[idx, 2])
                    elif box == self.opponent_color:
                        if not my_color_starts and theoretical_position:
                            opponent_color_starts = True
                        opponent_counter += 1
                    elif box == self.my_color:
                        if opponent_color_starts and opponent_counter > 0 \
                                and theoretical_position:
                            possible_position.append(theoretical_position)
                            theoretical_position = None
                            opponent_color_starts = False
                        my_color_starts = True
                        opponent_counter = 0

        return possible_position

    #get all possible moves - for all rows,columns,diagonals
    def get_possible_moves(self, board):
        self.board_size = len(board)
        board = np.expand_dims(np.array(board), axis=2)
        indices = np.indices((self.board_size, self.board_size)).T
        indices = np.transpose(indices, (1, 0, 2))
        board_w_info = np.concatenate((board, indices), axis=2)
        board_w_info_flipped = np.fliplr(board_w_info)
        rows = board_w_info
        columns = [board_w_info[:, i, :] for i in range(self.board_size)]
        diagonals_1 = [np.diagonal(board_w_info, offset=i) for i in range(-self.board_size + 1, self.board_size)]
        diagonals_1 = [np.transpose(elem, (1, 0)) for elem in diagonals_1]
        diagonals_2 = [np.diagonal(board_w_info_flipped, offset=i) for i in
                       range(-self.board_size + 1, self.board_size)]
        diagonals_2 = [np.transpose(elem, (1, 0)) for elem in diagonals_2]

        possible_positions = self.find_possible_coord(rows) + self.find_possible_coord(columns) \
                             + self.find_possible_coord(diagonals_1) + self.find_possible_coord(diagonals_2)
        return possible_positions

    # simulate move on the board
    def play_move(self, board, move):
        board[move[0]][move[1]] = self.my_color
        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, 1, 1, 0, -1, -1]
        for i in range(len(dx)):
            if self.__confirm_direction(move, dx[i], dy[i], board):
                self.change_stones_in_direction(move, dx[i], dy[i], board)

    # confirm the direction of the move
    def __confirm_direction(self, move, dx, dy, board):
        posx = move[0] + dx
        posy = move[1] + dy
        if self.__is_valid_position(posx, posy):
            if board[posx][posy] == self.opponent_color:
                while self.__is_valid_position(posx, posy):
                    posx += dx
                    posy += dy
                    if self.__is_valid_position(posx, posy):
                        if board[posx][posy] == -1:
                            return False
                        if board[posx][posy] == self.my_color:
                            return True
        return False

    #check if position is on the border
    def __is_valid_position(self, posx, posy):
        return ((posx >= 0) and (posx < self.board_size) and
                (posy >= 0) and (posy < self.board_size))

    #change the stones in given direction
    def change_stones_in_direction(self, move, dx, dy, board):
        players_color = self.my_color
        posx = move[0] + dx
        posy = move[1] + dy
        while not (board[posx][posy] == players_color):
            board[posx][posy] = players_color
            posx += dx
            posy += dy
