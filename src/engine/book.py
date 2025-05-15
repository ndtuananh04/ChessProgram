import struct
import random
import os
from collections import namedtuple

# Define a BookMove structure to store move data
BookMove = namedtuple('BookMove', ['key', 'move', 'weight', 'learn'])

class PolyglotBook:
    def __init__(self, filename=None):
        self.entries = []
        if filename and os.path.exists(filename):
            self.load(filename)
    
    def load(self, filename):
        """Load a polyglot opening book file."""
        self.entries = []
        
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Book file not found: {filename}")
            print(f"Current directory: {os.getcwd()}")
            abs_path = os.path.abspath(filename)
            print(f"Absolute path: {abs_path}")
            raise FileNotFoundError(f"Book file not found: {filename}")
        
        # Each polyglot entry is 16 bytes
        entry_size = 16
        
        try:
            print(f"Loading book from {filename}")
            with open(filename, 'rb') as f:
                data = f.read()
                
            # Parse entries
            entry_count = 0
            for i in range(0, len(data), entry_size):
                if i + entry_size <= len(data):
                    entry_data = data[i:i+entry_size]
                    key = struct.unpack('>Q', entry_data[0:8])[0]
                    move = struct.unpack('>H', entry_data[8:10])[0]
                    weight = struct.unpack('>H', entry_data[10:12])[0]
                    learn = struct.unpack('>I', entry_data[12:16])[0]
                    self.entries.append(BookMove(key, move, weight, learn))
                    entry_count += 1
            
            # Sort entries by key for faster lookup
            self.entries.sort(key=lambda x: x.key)
            print(f"Loaded {len(self.entries)} book moves from {filename}")
            
            # Print some entries for debug
            if self.entries:
                print("Sample book entries:")
                for i in range(min(5, len(self.entries))):
                    entry = self.entries[i]
                    print(f"  Key: {entry.key:016x}, Move: {entry.move:04x}, Weight: {entry.weight}")
        except Exception as e:
            print(f"Error loading opening book: {e}")
            raise
    
    def find_moves(self, position_hash):
        """Find all moves for a given position hash."""
        # Binary search to find the first matching entry
        low, high = 0, len(self.entries) - 1
        first_index = -1
        
        while low <= high:
            mid = (low + high) // 2
            if self.entries[mid].key == position_hash:
                first_index = mid
                high = mid - 1
            elif self.entries[mid].key < position_hash:
                low = mid + 1
            else:
                high = mid - 1
        
        if first_index == -1:
            return []
        
        # Collect all entries with the same key
        moves = []
        i = first_index
        while i < len(self.entries) and self.entries[i].key == position_hash:
            moves.append(self.entries[i])
            i += 1
        
        return moves
    
    def get_move(self, position_hash):
        """Get a weighted random move for the given position hash."""
        moves = self.find_moves(position_hash)
        if not moves:
            return None
        
        # Choose move based on weight
        total_weight = sum(move.weight for move in moves)
        if total_weight == 0:
            return random.choice(moves).move
        
        choice = random.randint(0, total_weight - 1)
        current_sum = 0
        
        for move in moves:
            current_sum += move.weight
            if current_sum > choice:
                return move.move
        
        return moves[-1].move

    def polyglot_to_uci(self, move):
        """Convert a polyglot move to UCI format."""
        from_file = (move >> 6) & 7
        from_rank = (move >> 9) & 7
        to_file = (move >> 0) & 7
        to_rank = (move >> 3) & 7
        promotion = (move >> 12) & 7
        
        from_square = from_rank * 8 + from_file
        to_square = to_rank * 8 + to_file
        
        files = 'abcdefgh'
        ranks = '12345678'
        
        from_str = files[from_file] + ranks[from_rank]
        to_str = files[to_file] + ranks[to_rank]
        
        # Map promotion piece
        promotion_pieces = ['', 'n', 'b', 'r', 'q']
        promotion_str = promotion_pieces[promotion] if promotion < len(promotion_pieces) else ''
        
        return from_str + to_str + promotion_str