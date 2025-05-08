import pygame
import time
import threading
from engine.engine import Position, initial, Searcher, Move, render, MATE_UPPER, MATE_LOWER

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

def draw_pieces(screen, position, rotate=False):
    """Draw the pieces on the board."""
    for i, piece in enumerate(position.board):
        if piece.isalpha():
            row, col = divmod(i - 21, 10)
            if 0 <= row < 8 and 0 <= col < 8:
                if rotate:
                    display_row, display_col = 7 - row, 7 - col
                    
                    is_upper = piece.isupper()
                    color = "black" if is_upper else "white"
                else:
                    display_row, display_col = row, col
                    color = "white" if piece.isupper() else "black"
                
                piece_type = piece.lower()
                image_path = f"{IMAGE_PATH}{color}/{piece_type}.png"
                piece_image = pygame.image.load(image_path).convert_alpha()
                
                pos = pygame.Rect(
                    BOARD_POS[0] + display_col * TILE_SIZE, 
                    BOARD_POS[1] + display_row * TILE_SIZE, 
                    TILE_SIZE, TILE_SIZE
                )
                screen.blit(piece_image, piece_image.get_rect(center=pos.center))

class EngineBattle:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Chess Engine Battle")
        self.screen = pygame.display.set_mode((TILE_SIZE * 8 + BORDER * 2 + 200, TILE_SIZE * 8 + BORDER * 2))
        self.board_surface = create_board_surface()
        
        self.white_searcher = Searcher()
        self.black_searcher = Searcher()
        
        self.position = Position(initial, 0, (True, True), (True, True), 0, 0)
        self.game_over = False
        self.thinking = False
        
        self.white_time = 3.0  
        self.black_time = 3.0  
        self.move_delay = 0.5  
        self.auto_play = False 
        
        self.move_count = 0
        self.white_time_used = 0
        self.black_time_used = 0

        self.current_turn = "white" 

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True
        print("Engine Battle started. Press 'r' to restart, 'q' to quit, 'space' to start/pause.")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  
                        self.position = Position(initial, 0, (True, True), (True, True), 0, 0)
                        self.game_over = False
                        self.move_count = 0
                        self.white_time_used = 0
                        self.black_time_used = 0
                        self.current_turn = "white"
                    elif event.key == pygame.K_q:  
                        running = False
                    elif event.key == pygame.K_SPACE: 
                        self.auto_play = not self.auto_play
                        if self.auto_play and not self.thinking and not self.game_over:
                            if self.current_turn == "white":
                                self.make_engine_move(is_white=True)
                            else:
                                self.make_engine_move(is_white=False)
                    elif event.key == pygame.K_UP:
                        self.white_time = min(10.0, self.white_time + 0.5)
                    elif event.key == pygame.K_DOWN:
                        self.white_time = max(0.5, self.white_time - 0.5)
                    elif event.key == pygame.K_RIGHT:
                        self.black_time = min(10.0, self.black_time + 0.5)
                    elif event.key == pygame.K_LEFT:
                        self.black_time = max(0.5, self.black_time - 0.5)

            if self.auto_play and not self.thinking and not self.game_over:
                current_time = pygame.time.get_ticks()
                if not hasattr(self, 'last_move_time') or current_time - self.last_move_time > self.move_delay * 1000:
                    self.last_move_time = current_time
                    if self.current_turn == "white":
                        self.make_engine_move(is_white=True)
                    else:
                        self.make_engine_move(is_white=False)

            self.update_screen()
            clock.tick(60)

        pygame.quit()

    def make_engine_move(self, is_white=True):
        """Let an engine calculate and make a move."""
        self.thinking = True
        start_time = time.time()
        best_move = None
        best_score = float('-inf')
        engine_time = self.white_time if is_white else self.black_time
        searcher = self.white_searcher if is_white else self.black_searcher

        timer_expired = [False]
        
        def stop_search():
            timer_expired[0] = True
        
        timer = threading.Timer(engine_time, stop_search)
        timer.start()
        
        try:
            for depth, gamma, score, move in searcher.search([self.position], start_time=start_time, time_limit=engine_time):
                if time.time() - start_time > engine_time * 0.95:
                    print(f"Time limit almost reached ({time.time() - start_time:.2f}s)")
                    break
                    
                if move and score >= gamma and score > best_score:
                    best_move = move
                    best_score = score
                    print(f"{self.current_turn} depth {depth}: {render(move.i)}{render(move.j)} (score: {score})")
            
            if best_move:
                move_time = time.time() - start_time
                print(f"{self.current_turn} moves: {render(best_move.i)}{render(best_move.j)}")
                print(f"{self.current_turn} thinking time: {move_time:.2f}s")
                
                self.move_count += 1
                self.current_turn = "black" if self.move_count % 2 == 1 else "white"
                if is_white:
                    self.white_time_used += move_time
                else:
                    self.black_time_used += move_time
                
                self.position = self.position.move(best_move)
                
                if best_score >= MATE_LOWER:
                    if self.current_turn == "black":
                        print(f"Checkmate! White wins.")
                        self.display_message(f"Checkmate! White wins")
                        self.game_over = True
                    else:
                        print(f"Checkmate! Black wins.")
                        self.display_message(f"Checkmate! Black wins")
                        self.game_over = True
                    
                self.check_game_over()
            else:
                print(f"Engine ({self.current_turn}) could not find a valid move")
                self.game_over = True
                
        except Exception as e:
            print(f"Engine error: {e}")
        
        finally:
            timer.cancel()
            self.thinking = False
            self.last_move_time = pygame.time.get_ticks()

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

    def check_game_over(self):
        """Check if the game is over (checkmate or draw)."""
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            in_check = self.is_king_in_check(self.position)
            
            if in_check:
                winner = "White" if self.current_turn == "black" else "Black"
                
                print(f"Checkmate! {winner} wins.")
                self.display_message(f"Checkmate! {winner} wins")
            else:
                print("Stalemate! Game is a draw.")
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
        
        rotate_board = (self.current_turn == "black")
        
        if rotate_board:
            rotated_board = pygame.Surface((TILE_SIZE * 8, TILE_SIZE * 8))
            dark = False
            for y in range(8):
                for x in range(8):
                    rect = pygame.Rect((7-x) * TILE_SIZE, (7-y) * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                    pygame.draw.rect(rotated_board, COLOR_DARK if dark else COLOR_LIGHT, rect)
                    dark = not dark
                dark = not dark
            self.screen.blit(rotated_board, BOARD_POS)
        else:
            self.screen.blit(self.board_surface, BOARD_POS)
        
        draw_pieces(self.screen, self.position, rotate=rotate_board)
        
        font_small = pygame.font.Font(None, 18)
        if rotate_board:
            for i in range(8):
                file_label = font_small.render(chr(97 + (7-i)), True, (200, 200, 200))
                self.screen.blit(file_label, (BOARD_POS[0] + i * TILE_SIZE + TILE_SIZE//2 - 4, 
                                            BOARD_POS[1] + 8 * TILE_SIZE + 5))

                rank_label = font_small.render(str(i + 1), True, (200, 200, 200))
                self.screen.blit(rank_label, (BOARD_POS[0] - 15, 
                                            BOARD_POS[1] + i * TILE_SIZE + TILE_SIZE//2 - 4))
        else:
            for i in range(8):
                file_label = font_small.render(chr(97 + i), True, (200, 200, 200))
                self.screen.blit(file_label, (BOARD_POS[0] + i * TILE_SIZE + TILE_SIZE//2 - 4, 
                                            BOARD_POS[1] + 8 * TILE_SIZE + 5))

                rank_label = font_small.render(str(8 - i), True, (200, 200, 200))
                self.screen.blit(rank_label, (BOARD_POS[0] - 15, 
                                            BOARD_POS[1] + i * TILE_SIZE + TILE_SIZE//2 - 4))
        
        font = pygame.font.Font(None, 24)
        y_offset = 20
        
        if self.thinking:
            text = font.render("Engine thinking...", True, (255, 255, 255))
            self.screen.blit(text, (BOARD_POS[0] + TILE_SIZE * 8 + 20, y_offset))
            y_offset += 30
                
        turn_text = "White's turn" if self.current_turn == "white" else "Black's turn"
        if self.game_over:
            turn_text = "Game Over"
        text = font.render(turn_text, True, (255, 255, 255))
        self.screen.blit(text, (BOARD_POS[0] + TILE_SIZE * 8 + 20, y_offset))
        y_offset += 25
        
        if rotate_board:
            rotation_text = "."
            text = font.render(rotation_text, True, (255, 200, 100))
            self.screen.blit(text, (BOARD_POS[0] + TILE_SIZE * 8 + 20, y_offset))
        y_offset += 30
        
        status = "Auto Play: ON" if self.auto_play else "Auto Play: OFF"
        text = font.render(status, True, (255, 255, 255))
        self.screen.blit(text, (BOARD_POS[0] + TILE_SIZE * 8 + 20, y_offset))
        y_offset += 30
        
        text = font.render(f"White time: {self.white_time:.1f}s", True, (255, 255, 255))
        self.screen.blit(text, (BOARD_POS[0] + TILE_SIZE * 8 + 20, y_offset))
        y_offset += 30
        
        text = font.render(f"Black time: {self.black_time:.1f}s", True, (255, 255, 255))
        self.screen.blit(text, (BOARD_POS[0] + TILE_SIZE * 8 + 20, y_offset))
        y_offset += 30
        
        text = font.render(f"Moves: {self.move_count}", True, (255, 255, 255))
        self.screen.blit(text, (BOARD_POS[0] + TILE_SIZE * 8 + 20, y_offset))
        y_offset += 30
        
        y_offset += 20
        text = font.render("Controls:", True, (200, 200, 200))
        self.screen.blit(text, (BOARD_POS[0] + TILE_SIZE * 8 + 20, y_offset))
        y_offset += 25
        
        controls = [
            "SPACE: Auto play On/Off",
            "R: Reset game",
            "Q: Quit"
        ]
        
        for control in controls:
            text = font.render(control, True, (200, 200, 200))
            self.screen.blit(text, (BOARD_POS[0] + TILE_SIZE * 8 + 20, y_offset))
            y_offset += 25
            
        pygame.display.flip()