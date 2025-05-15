from __future__ import print_function

import time, math, os, random
import zlib
from itertools import count
from collections import namedtuple, defaultdict

piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
pst = {
    'P': (   0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  10,   9,   6,   1,   0, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'N': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'B': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'R': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -24, -18,   5,  -2, -18, -31, -32),
    'Q': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'K': (   4,  54,  47, -99, -99,  60,  83, -62,
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
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

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
MATE_LOWER = piece["K"] - 10 * piece["Q"]
MATE_UPPER = piece["K"] + 10 * piece["Q"]

# Constants for tuning search
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
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
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
        

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        # Look in the table if we have already searched this position before.
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        entry = self.tp_score.get((pos, depth, can_null), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma: return entry.lower
        if entry.upper < gamma: return entry.upper

        # Let's not repeat positions. We don't chat
        # - at the root (can_null=False) since it is in history, but not a draw.
        # - at depth=0, since it would be expensive and break "futulity pruning".
        if can_null and depth > 0 and pos in self.history:
            return 0

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # First try not moving at all. We only do this if there is at least one major
            # piece left on the board, since otherwise zugzwangs are too dangerous.
            # FIXME: We also can't null move if we can capture the opponent king.
            # Since if we do, we won't spot illegal moves that could lead to stalemate.
            # For now we just solve this by not using null-move in very unbalanced positions.
            # TODO: We could actually use null-move in QS as well. Not sure it would be very useful.
            # But still.... We just have to move stand-pat to be before null-move.
            #if depth > 2 and can_null and any(c in pos.board for c in "RBNQ"):
            #if depth > 2 and can_null and any(c in pos.board for c in "RBNQ") and abs(pos.score) < 500:
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

            # Only play the move if it would be included at the current val-limit,
            # since otherwise we'd get search instability.
            # We will search it again in the main loop below, but the tp will fix
            # things for us.
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

        # Stalemate checking is a bit tricky: Say we failed low, because
        # we can't (legally) move and so the (real) score is -infty.
        # At the next depth we are allowed to just return r, -infty <= r < gamma,
        # which is normally fine.
        # However, what if gamma = -10 and we don't have any legal moves?
        # Then the score is actaully a draw and we should fail high!
        # Thus, if best < gamma and best < 0 we need to double check what we are doing.

        # We will fix this problem another way: We add the requirement to bound, that
        # it always returns MATE_UPPER if the king is capturable. Even if another move
        # was also sufficient to go above gamma. If we see this value we know we are either
        # mate, or stalemate. It then suffices to check whether we're in check.

        # Note that at low depths, this may not actually be true, since maybe we just pruned
        # all the legal moves. So sunfish may report "mate", but then after more search
        # realize it's not a mate after all. That's fair.

        # This is too expensive to test at depth == 0
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

#input = raw_input

# minifier-hide start
# import sys, tools.uci
# tools.uci.run(sys.modules[__name__], hist[-1])
# sys.exit()
# minifier-hide end

# if __name__ == "__main__":
#     hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]

#     searcher = Searcher()
#     while True:
#         args = input().split()
#         if args[0] == "isready":
#             print("readyok")

#         elif args[0] == "quit":
#             break

#         elif args[:2] == ["position", "startpos"]:
#             del hist[1:]
#             for ply, move in enumerate(args[3:]):
#                 i, j, prom = parse(move[:2]), parse(move[2:4]), move[4:].upper()
#                 if ply % 2 == 1:
#                     i, j = 119 - i, 119 - j
#                 hist.append(hist[-1].move(Move(i, j, prom)))

#         elif args[0] == "go":
#             wtime, btime, winc, binc = [int(a) / 1000 for a in args[2::2]]
#             if len(hist) % 2 == 0:
#                 wtime, winc = btime, binc
#             think = min(wtime / 40 + winc, wtime / 2 - 1)

#             start = time.time()
#             move_str = None
#             for depth, gamma, score, move in Searcher().search(hist):
#                 # The only way we can be sure to have the real move in tp_move,
#                 # is if we have just failed high.
#                 if score >= gamma:
#                     i, j = move.i, move.j
#                     if len(hist) % 2 == 0:
#                         i, j = 119 - i, 119 - j
#                     move_str = render(i) + render(j) + move.prom.lower()
#                     print("info depth", depth, "score cp", score, "pv", move_str)
#                 if move_str and time.time() - start > think * 0.8:
#                     break

#             print("bestmove", move_str or '(none)')