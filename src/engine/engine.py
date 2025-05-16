from __future__ import print_function

import time, math, os, random
import zlib
from itertools import count
from collections import namedtuple, defaultdict

piece = {"P": 100, "N": 280, "B": 320, "R": 480, "Q": 930, "K": 60000}
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            80,  85,  85,  85,  85,  85,  85,  80,
            10,  30,  35,  40,  40,  35,  30,  10,
           -15,  15,   0,  15,  15,   0,  15, -15,
           -25,   5,  10,  10,  10,  10,   5, -25,
           -20,   5,   5,  -5,  -5,   5,   5, -20,
           -30,   5,  -5, -35, -35,  -5,   5, -30,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -65, -55, -55, -10, -10, -55, -55, -65,
            -5,  -5,  60,   0,   0,  60,  -5,  -5,
            10,  65,  15,  75,  75,  15,  65,  10,
            20,  25,  40,  35,  35,  40,  25,  20,
             0,   5,  30,  20,  20,  30,   5,   0,
           -15,  10,  15,  20,  20,  15,  10, -15,
           -20, -15,   0,   0,   0,   0, -15, -20,
           -70, -25, -25, -20, -20, -25, -25, -70),
    'B': ( -50, -40, -60, -25, -25, -60, -40, -50,
           -15,  15,  35, -40, -40,  35,  15, -15,
           -10,  30, -15,  45,  45, -15,  30, -10,
            20,  15,  25,  30,  30,  25,  15,  20,
            10,  15,  15,  20,  20,  15,  15,  10,
            15,  20,  20,  10,  10,  20,  20,  15,
            15,  20,  10,   5,   5,  10,  20,  15,
           -10,   0, -10, -15, -15, -10,   0, -10),
    'R': (  40,  30,  35,  35,  35,  35,  30,  40,
            55,  30,  55,  60,  60,  55,  30,  55,
            20,  30,  30,  40,  40,  30,  30,  20,
             0,   5,  15,  15,  15,  15,   5,   0,
           -30, -25, -15, -15, -15, -15, -25, -30,
           -40, -30, -35, -25, -25, -35, -30, -40,
           -50, -40, -35, -30, -30, -35, -40, -50,
           -30, -25, -20,   0,   0, -20, -25, -30),
    'Q': (  15,  10,  20,  25,  25,  20,  10,  15,
            15,  30,  60,  65,  65,  60,  30,  15,
             0,  40,  35,  65,  65,  35,  40,   0,
             0, -15,  20,  20,  20,  20, -15,   0,
           -15, -15,  -5,  -5,  -5,  -5, -15, -15,
           -25, -10, -15, -15, -15, -15, -10, -25,
           -35, -20, -15, -15, -15, -15, -20, -35,
           -40, -35, -30, -30, -30, -30, -35, -40),
    'K': (  10,  50,  50, -80, -80,  50,  50,  10,
           -15,  10,  55,  55,  55,  55,  10, -15,
           -45,  15, -25,  35,  35, -25,  15, -45,
           -50,  10,   5,  -5,  -5,   5,  10, -50,
           -50, -45, -50, -30, -30, -50, -45, -50,
           -40, -35, -40, -65, -65, -40, -35, -40,
             0,   5, -15, -50, -50, -15,   5,   0,
            15,  25,   0, -10, -10,   0,  25,  15),
}
# Pad tables and join piece and pst dictionaries
for k, table in pst.items():
    padrow = lambda row: (0,) + tuple(x + piece[k] for x in row) + (0,)
    pst[k] = sum((padrow(table[i * 8 : i * 8 + 8]) for i in range(8)), ())
    pst[k] = (0,) * 20 + pst[k] + (0,) * 20

###############################################################################
# Global constants
###############################################################################

A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    "         \n"  #   0 -  9
    "         \n"  #  10 - 19
    " rnbqkbnr\n"  #  20 - 29
    " pppppppp\n"  #  30 - 39
    " ........\n"  #  40 - 49
    " ........\n"  #  50 - 59
    " ........\n"  #  60 - 69
    " ........\n"  #  70 - 79
    " PPPPPPPP\n"  #  80 - 89
    " RNBQKBNR\n"  #  90 - 99
    "         \n"  # 100 -109
    "         \n"  # 110 -119
)

N, E, S, W = -10, 1, 10, -1
directions = {
    "P": (N, N+N, N+W, N+E),
    "N": (N+N+E, E+N+E, E+S+E, S+S+E, S+S+W, W+S+W, W+N+W, N+N+W),
    "B": (N+E, S+E, S+W, N+W),
    "R": (N, E, S, W),
    "Q": (N, E, S, W, N+E, S+E, S+W, N+W),
    "K": (N, E, S, W, N+E, S+E, S+W, N+W)
}

MATE_LOWER = piece["K"] - 10 * piece["Q"]
MATE_UPPER = piece["K"] + 10 * piece["Q"]

QS = 40
QS_A = 140
EVAL_ROUGHNESS = 15

# minifier-hide start
# opt_ranges = dict(
#     QS = (0, 300),
#     QS_A = (0, 300),
#     EVAL_ROUGHNESS = (0, 50),
# )
# minifier-hide end


###############################################################################
# Chess logic
###############################################################################


Move = namedtuple("Move", "i j prom")


class Position(namedtuple("Position", "board score wc bc ep kp")):
    """A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square
    kp - the king passant square
    """

    def gen_moves(self):
        for i, p in enumerate(self.board):
            if not p.isupper():
                continue
            for d in directions[p]:
                for j in count(i + d, d):
                    q = self.board[j]
                    # Stay inside the board, and off friendly pieces
                    if q.isspace() or q.isupper():
                        break
                    # Pawn move, double move and capture
                    if p == "P":
                        if d in (N, N + N) and q != ".": break
                        if d == N + N and (i < A1 + N or self.board[i + N] != "."): break
                        if (
                            d in (N + W, N + E)
                            and q == "."
                            and j not in (self.ep, self.kp, self.kp - 1, self.kp + 1)
                            #and j != self.ep and abs(j - self.kp) >= 2
                        ):
                            break
                        # If we move to the last row, we can be anything
                        if A8 <= j <= H8:
                            for prom in "NBRQ":
                                yield Move(i, j, prom)
                            break
                    # Move it
                    yield Move(i, j, "")
                    # Stop crawlers from sliding, and sliding after captures
                    if p in "PNK" or q.islower():
                        break
                    # Castling, by sliding the rook next to the king
                    if i == A1 and self.board[j + E] == "K" and self.wc[0]:
                        yield Move(j + E, j + W, "")
                    if i == H1 and self.board[j + W] == "K" and self.wc[1]:
                        yield Move(j + W, j + E, "")

    def rotate(self, nullmove=False):
        """Rotates the board, preserving enpassant, unless nullmove"""
        return Position(
            self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
            119 - self.ep if self.ep and not nullmove else 0,
            119 - self.kp if self.kp and not nullmove else 0,
        )

    def move(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i + 1 :]
        # Copy variables and reset ep and kp
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # Actual move
        board = put(board, j, board[i])
        board = put(board, i, ".")
        # Castling rights, we move the rook or capture the opponent's
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        # Castling
        if p == "K":
            wc = (False, False)
            if abs(j - i) == 2:
                kp = (i + j) // 2
                board = put(board, A1 if j < i else H1, ".")
                board = put(board, kp, "R")
        # Pawn promotion, double move and en passant capture
        if p == "P":
            if A8 <= j <= H8:
                board = put(board, j, prom)
            if j - i == 2 * N:
                ep = i + N
            if j == self.ep:
                board = put(board, j + S, ".")
        # We rotate the returned position, so it's ready for the next player
        return Position(board, score, wc, bc, ep, kp).rotate()

    def value(self, move):
        i, j, prom = move
        p, q = self.board[i], self.board[j]
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q.islower():
            score += pst[q.upper()][119 - j]
        # Castling check detection
        if abs(j - self.kp) < 2:
            score += pst["K"][119 - j]
        # Castling
        if p == "K" and abs(i - j) == 2:
            score += pst["R"][(i + j) // 2]
            score -= pst["R"][A1 if j < i else H1]
        # Special pawn stuff
        if p == "P":
            if A8 <= j <= H8:
                score += pst[prom][j] - pst["P"][j]
            if j == self.ep:
                score += pst["P"][119 - (j + S)]
        return score


###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple("Entry", "lower upper")


class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0

        from engine.book import PolyglotBook
        self.book = PolyglotBook()
        book_path = "C:\\Tanh\\ChessProgram\\src\\book\\Titans.bin"
        try:
            self.book.load(book_path)
            print(f"Successfully loaded opening book: {book_path}")
        except Exception as e:
            print(f"Warning: Could not load opening book: {e}")
            # Continue without book

        self.opening_moves = {
            "initial": [
                (Move(85, 65, ""), 25),    
                (Move(84, 64, ""), 18),    
                (Move(96, 85, ""), 15),    
                (Move(94, 76, ""), 10),    
                (Move(83, 63, ""), 10),    
                (Move(82, 62, ""), 5),    
                (Move(86, 66, ""), 5)     
            ],

            "e4e5": [
                (Move(96, 85, ""), 20),    
                (Move(94, 76, ""), 15),    
                (Move(92, 74, ""), 15)     
            ],
 
            "e4c5": [
                (Move(96, 85, ""), 20),   
                (Move(94, 76, ""), 15),   
                (Move(83, 63, ""), 10)     
            ],

            "d4d5": [
                (Move(83, 63, ""), 20),    
                (Move(94, 76, ""), 15),    
                (Move(96, 85, ""), 15)     
            ],

            "Nf3": [
                (Move(84, 64, ""), 20),    
                (Move(83, 63, ""), 15),    
                (Move(85, 65, ""), 15)     
            ],

            "c4": [
                (Move(94, 76, ""), 20),    
                (Move(96, 85, ""), 20),    
                (Move(85, 65, ""), 15)     
            ]
        }

    def bound(self, pos, gamma, depth, can_null=True):
        """ Let s* be the "true" score of the sub-tree we are searching.
            The method returns r, where
            if gamma >  s* then s* <= r < gamma  (A better upper bound)
            if gamma <= s* then gamma <= r <= s* (A better lower bound) """
        self.nodes += 1
        
        depth = max(depth, 0)

        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        entry = self.tp_score.get((pos, depth, can_null), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma: return entry.lower
        if entry.upper < gamma: return entry.upper

        if can_null and depth > 0 and pos in self.history:
            return 0

        def moves():
            if depth > 2 and can_null and abs(pos.score) < 500:
                yield None, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)

            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else.
            if depth == 0:
                yield None, pos.score

            # Look for the strongest ove from last time, the hash-move.
            killer = self.tp_move.get(pos)

            # If there isn't one, try to find one with a more shallow search.
            # This is known as Internal Iterative Deepening (IID). We set
            # can_null=True, since we want to make sure we actually find a move.
            if not killer and depth > 2:
                self.bound(pos, gamma, depth - 3, can_null=False)
                killer = self.tp_move.get(pos)

            # If depth == 0 we only try moves with high intrinsic score (captures and
            # promotions). Otherwise we do all moves. This is called quiescent search.
            val_lower = QS - depth * QS_A

            if killer and pos.value(killer) >= val_lower:
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1)

            # Then all the other moves
            for val, move in sorted(((pos.value(m), m) for m in pos.gen_moves()), reverse=True):
                # Quiescent search
                if val < val_lower:
                    break

                # If the new score is less than gamma, the opponent will for sure just
                # stand pat, since ""pos.score + val < gamma === -(pos.score + val) >= 1-gamma""
                # This is known as futility pruning.
                if depth <= 1 and pos.score + val < gamma:
                    # Need special case for MATE, since it would normally be caught
                    # before standing pat.
                    yield move, pos.score + val if val < MATE_LOWER else MATE_UPPER
                    # We can also break, since we have ordered the moves by value,
                    # so it can't get any better than this.
                    break

                yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                if move is not None:
                    self.tp_move[pos] = move
                break

        if depth > 2 and best == -MATE_UPPER:
            flipped = pos.rotate(nullmove=True)
            # Hopefully this is already in the TT because of null-move
            in_check = self.bound(flipped, MATE_UPPER, 0) == MATE_UPPER
            best = -MATE_LOWER if in_check else 0

        # Table part 2
        if best >= gamma:
            self.tp_score[pos, depth, can_null] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos, depth, can_null] = Entry(entry.lower, best)

        return best

    def zobrist_hash(self, position):
        """Calculate a Zobrist hash for a position compatible with polyglot format."""
        h = 0
        
        zobrist_pieces = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0x2E5B4D0, 0x6A45D0, 0x22B1F0, 0xB90C0, 0x42D840, 0x190F0,
            0x9B3D0, 0x891870, 0x5A3870, 0xD21870, 0x6C3870, 0x31FE70
        ]
        
        piece_index = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12
        }
        
        for i, p in enumerate(position.board):
            if p in piece_index:
                if 21 <= i < 99 and i % 10 >= 1 and i % 10 <= 8:
                    file = (i % 10) - 1
                    rank = 7 - ((i - 21) // 10)
                    sq64 = rank * 8 + file
                    
                    piece_idx = piece_index[p]
                    h ^= (zobrist_pieces[piece_idx] << sq64) & 0xFFFFFFFFFFFFFFFF
        
        castling_values = [0x31DC4, 0x77EF8, 0x6E6C4, 0x22714]
        if position.wc[1]: h ^= castling_values[0]  # White kingside
        if position.wc[0]: h ^= castling_values[1]  # White queenside
        if position.bc[1]: h ^= castling_values[2]  # Black kingside
        if position.bc[0]: h ^= castling_values[3]  # Black queenside
        
        ep_values = [0x7B3A0, 0x7BD50, 0x7C270, 0x7C6C0, 0x7CA40, 0x7CDC0, 0x7D080, 0x7D4B0]
        if position.ep:
            file = (position.ep % 10) - 1
            if 0 <= file < 8:
                h ^= ep_values[file]
        
        side_to_move = 0 if position.board.count('K') > 0 else 1  
        if side_to_move == 1:  
            h ^= 0x3FF8
        
        print(f"Calculated hash: {h:016x}")
        return h
    
    def search(self, history, use_book=True, start_time=None, time_limit=None):
        """Iterative deepening MTD-bi search"""
        self.nodes = 0
        self.history = set(history[:-1])
        pos = history[-1]
        
        found_book_move = False

        if use_book:
            try:
                pos_hash = self.zobrist_hash(pos)
                book_move_value = self.book.get_move(pos_hash)
                
                if book_move_value is not None:
                    print(f"Found book move value: {book_move_value:04x}")
                    
                    from_file = (book_move_value >> 6) & 7
                    from_rank = (book_move_value >> 9) & 7
                    to_file = (book_move_value >> 0) & 7
                    to_rank = (book_move_value >> 3) & 7
                    promotion = (book_move_value >> 12) & 7
                    
                    print(f"Book move: from_file={from_file}, from_rank={from_rank}, to_file={to_file}, to_rank={to_rank}, promotion={promotion}")
                    
                    from_sq = 21 + (7 - from_rank) * 10 + (from_file + 1)
                    to_sq = 21 + (7 - to_rank) * 10 + (to_file + 1)
                    
                    print(f"Converted coordinates: from_sq={from_sq}, to_sq={to_sq}")
                    
                    prom_piece = "" if promotion == 0 else "NBRQ"[promotion-1]
                    
                    legal_moves = list(pos.gen_moves())
                    print(f"Legal moves: {len(legal_moves)}")
                    
                    found_move = None
                    for move in legal_moves:
                        if move.i == from_sq and move.j == to_sq and move.prom == prom_piece:
                            found_move = move
                            print(f"Found matching legal move: {render(move.i)}{render(move.j)}{move.prom}")
                            break
                    
                    if found_move:
                        print(f"Using book move: {render(from_sq)}{render(to_sq)}{prom_piece}")
                        found_book_move = True
                        yield 1, 0, 1000, found_move
                        return
                    else:
                        print(f"Book move {render(from_sq)}{render(to_sq)}{prom_piece} not legal in current position")
                        
                        print("Trying alternative coordinate conversion...")
                        from_sq_alt = 91 - from_rank * 10 + from_file
                        to_sq_alt = 91 - to_rank * 10 + to_file
                        print(f"Alternative coordinates: from_sq={from_sq_alt}, to_sq={to_sq_alt}")
                        
                        for move in legal_moves:
                            if move.i == from_sq_alt and move.j == to_sq_alt and move.prom == prom_piece:
                                print(f"Found match with alternative coordinates: {render(move.i)}{render(move.j)}{move.prom}")
                                found_book_move = True
                                yield 1, 0, 1000, move
                                return
                else:
                    print("No book move found for this position hash")
                    
            except Exception as e:
                import traceback
                print(f"Error using opening book: {e}")
                traceback.print_exc()
        
        if use_book and not found_book_move and hasattr(self, 'opening_moves'):
            try:
                print("Trying custom opening moves...")
                
                key = None
                
                if pos.board == initial:
                    key = "initial"
                    print("Recognized initial position")
                
                elif len(history) >= 3 and history[-2].kp == 65:  # e4 vừa được đi
                    if pos.kp == 55:  # e5 vừa được đi
                        key = "e4e5"
                        print("Recognized e4e5 position")
                    elif pos.kp == 53:  # c5 vừa được đi
                        key = "e4c5"
                        print("Recognized e4c5 position")
                
                elif len(history) >= 3 and history[-2].kp == 64:  # d4 vừa được đi
                    if pos.kp == 54:  # d5 vừa được đi
                        key = "d4d5"
                        print("Recognized d4d5 position")
                
                elif len(history) >= 3 and history[-2].kp == 85:  # Nf3 vừa được đi
                    key = "Nf3"
                    print("Recognized Nf3 position")
                
                elif len(history) >= 3 and history[-2].kp == 63:  # c4 vừa được đi
                    key = "c4"
                    print("Recognized c4 position")
                
                if key and key in self.opening_moves:
                    print(f"Using custom moves for position: {key}")
                    moves = self.opening_moves[key]
                    
                    total_weight = sum(weight for _, weight in moves)
                    choice = random.randint(0, total_weight - 1)
                    current_sum = 0
                    
                    for move, weight in moves:
                        current_sum += weight
                        if current_sum > choice:
                            # Kiểm tra nước đi có hợp lệ không
                            legal_moves = list(pos.gen_moves())
                            for legal_move in legal_moves:
                                if legal_move.i == move.i and legal_move.j == move.j and legal_move.prom == move.prom:
                                    print(f"Using custom opening move: {render(move.i)}{render(move.j)}{move.prom}")
                                    yield 1, 0, 1000, legal_move
                                    return
                    
                    print("Selected custom move not legal in current position")
            except Exception as e:
                print(f"Error using custom opening moves: {e}")
                import traceback
                traceback.print_exc()
        
        # Nếu không tìm được nước đi từ opening book hoặc custom moves, tiếp tục với tìm kiếm thông thường
        for depth in range(1, 100):
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                gamma = (lower + upper + 1) // 2
                score = self.bound(pos, gamma, depth)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
            
            # Yield results
            self.tp_score.clear()
            
            # Have we found a "mate in x"?
            if lower >= MATE_LOWER:
                # If the score is a mate score, we convert it to "mate in x"
                mate_in = (MATE_UPPER - lower) // 2
                yield depth, upper, lower, self.tp_move.get(pos)
                break
            
            # Check time limit
            if start_time and time_limit and time.time() - start_time > time_limit * 0.8:
                break
            
            yield depth, upper, lower, self.tp_move.get(pos)

    

###############################################################################
# UCI User interface
###############################################################################


def parse(c):
    fil, rank = ord(c[0]) - ord("a"), int(c[1]) - 1
    return A1 + fil - 10 * rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord("a")) + str(-rank + 1)

hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]

