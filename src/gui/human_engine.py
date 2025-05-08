import pygame
import time
from engine.engine import Position, initial, Searcher, Move, render, MATE_UPPER, MATE_LOWER
import threading

TILE_SIZE = 64
BORDER = 20
BOARD_POS = (BORDER, BORDER)
COLOR_DARK = (181, 136, 99)
COLOR_LIGHT = (240, 217, 181)
COLOR_BG = (22, 21, 18)
IMAGE_PATH = "gui/images/"  

def create_board_surface():
    """Create the chessboard surface."""
    board_surface = pygame.Surface((TILE_SIZE * 8, TILE_SIZE * 8))
    dark = False
    for y in range(8):
        for x in range(8):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(board_surface, COLOR_DARK if dark else COLOR_LIGHT, rect)
            dark = not dark
        dark = not dark
    return board_surface

def draw_pieces(screen, position):
    """Draw the pieces on the board."""
    for i, piece in enumerate(position.board):
        if piece.isalpha():
            row, col = divmod(i - 21, 10)
            if 0 <= row < 8 and 0 <= col < 8:
                color = "white" if piece.isupper() else "black"
                piece_type = piece.lower()
                image_path = f"{IMAGE_PATH}{color}/{piece_type}.png"
                piece_image = pygame.image.load(image_path).convert_alpha()
                pos = pygame.Rect(BOARD_POS[0] + col * TILE_SIZE, BOARD_POS[1] + row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                screen.blit(piece_image, piece_image.get_rect(center=pos.center))

class GameWindow:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Chess")
        self.screen = pygame.display.set_mode((TILE_SIZE * 8 + BORDER * 2, TILE_SIZE * 8 + BORDER * 2))
        self.board_surface = create_board_surface()
        self.searcher = Searcher()
        self.position = Position(initial, 0, (True, True), (True, True), 0, 0)
        self.selected_square = None
        self.game_over = False
        self.thinking = False
        self.engine_time = 2 

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True
        print("Game started. Press 'r' to restart, 'q' to quit.")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: 
                        self.position = Position(initial, 0, (True, True), (True, True), 0, 0)
                        self.game_over = False
                    elif event.key == pygame.K_q:  
                        running = False
                elif not self.game_over and not self.thinking:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self.handle_mouse_down()
                    elif event.type == pygame.MOUSEBUTTONUP:
                        self.handle_mouse_up()

            self.update_screen()
            clock.tick(60)

        pygame.quit()

    def handle_mouse_down(self):
        """Handle mouse button down events."""
        x, y = pygame.mouse.get_pos()
        col = (x - BOARD_POS[0]) // TILE_SIZE
        row = (y - BOARD_POS[1]) // TILE_SIZE
        
        if 0 <= col < 8 and 0 <= row < 8:
            square = 21 + row * 10 + col
            if self.position.board[square].isupper(): 
                self.selected_square = square
                print(f"Selected square: {square} (Piece: {self.position.board[square]})")

    def handle_mouse_up(self):
        """Handle mouse button up events."""
        if self.selected_square is None:
            return
            
        x, y = pygame.mouse.get_pos()
        col = (x - BOARD_POS[0]) // TILE_SIZE
        row = (y - BOARD_POS[1]) // TILE_SIZE
        
        if 0 <= col < 8 and 0 <= row < 8:
            target = 21 + row * 10 + col
            move = Move(self.selected_square, target, "")
            
            if move in self.get_legal_moves():
                print(f"Move: {render(move.i)} to {render(move.j)}")
                self.position = self.position.move(move)
                self.selected_square = None
                
                if self.check_game_over():
                    return
                    
                self.engine_move()
            else:
                print("Illegal move")
                self.selected_square = None

    def get_legal_moves(self):
        """Generate all legal moves that don't leave the king in check."""
        legal_moves = []
        for move in self.position.gen_moves():
            new_pos = self.position.move(move)
            if not self.is_king_in_check(new_pos):
                legal_moves.append(move)
        return legal_moves

    def is_king_in_check(self, position):
        """Check if the king is in check in the given position."""
        for move in position.gen_moves():
            if position.board[move.j].lower() == 'k':  
                return True
        return False

    def engine_move(self):
        """Let Sunfish calculate and make a move with a strict time limit."""
        self.thinking = True
        start_time = time.time()
        best_move = None
        best_score = float('-inf')
        
        timer_expired = [False] 
        
        def stop_search():
            timer_expired[0] = True
        
        timer = threading.Timer(self.engine_time, stop_search)
        timer.start()
        
        try:
            for depth, gamma, score, move in self.searcher.search([self.position]):
                if timer_expired[0]:
                    break
                    
                if move and score >= gamma and score > best_score:
                    best_move = move
                    best_score = score
                    print(f"Depth {depth}: {render(move.i)}{render(move.j)} (score: {score})")
            
            if best_move:
                print(f"Engine moves: {render(best_move.i)}{render(best_move.j)}")
                print(f"Thinking time: {time.time() - start_time:.2f}s")
                
                if not self.position.board[best_move.j].isupper():
                    self.position = self.position.move(best_move)
                    if best_score == MATE_LOWER or best_score == MATE_UPPER:
                            print("Checkmate! Black wins.")
                            self.display_message("Checkmate! Black wins")
                            self.game_over = True
                            return
                    self.check_game_over()
                else:
                    print("Engine tried to make an illegal move!")
            else:
                print("Engine could not find a valid move")
                
        except Exception as e:
            print(f"Engine error: {e}")
        
        finally:
            timer.cancel()
            self.thinking = False

    def check_game_over(self):
        """Kiểm tra xem ván đấu đã kết thúc chưa (chiếu bí hoặc hòa cờ)."""
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            in_check = self.is_king_in_check(self.position)
            
            if in_check:
                is_white_turn = self.position.board.count('K') > 0
                winner = "Black" if is_white_turn else "White"
                
                print(f"Checkmate! {winner} wins.")
                self.display_message(f"Checkmate! {winner} wins")
            else:
                print("Stalemate! Game over.")
                self.display_message("Stalemate! Draw")
            
            self.game_over = True
            return True
        return False

    def display_message(self, message):
        """Display a message on the screen."""
        font = pygame.font.Font(None, 36)
        text = font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        
        bg = pygame.Surface((text.get_width() + 20, text.get_height() + 20))
        bg.fill((0, 0, 0))
        bg.set_alpha(150)
        bg_rect = bg.get_rect(center=text_rect.center)
        
        self.screen.blit(bg, bg_rect)
        self.screen.blit(text, text_rect)
        pygame.display.flip()

    def update_screen(self):
        """Update the game screen."""
        self.screen.fill(COLOR_BG)
        self.screen.blit(self.board_surface, BOARD_POS)
        
        if self.selected_square is not None:
            row, col = divmod(self.selected_square - 21, 10)
            rect = pygame.Rect(BOARD_POS[0] + col * TILE_SIZE, BOARD_POS[1] + row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            highlight = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            highlight.fill((255, 255, 0, 100))  
            self.screen.blit(highlight, rect)
        
        draw_pieces(self.screen, self.position)
        
        if self.thinking:
            font = pygame.font.Font(None, 24)
            text = font.render("Engine thinking...", True, (255, 255, 255))
            self.screen.blit(text, (10, 10))
            
        pygame.display.flip()

