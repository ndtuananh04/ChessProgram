import pygame
from engine.engine import Position, initial, Searcher, Move, parse, render

# Constants for the GUI
TILE_SIZE = 64
BORDER = 10
BOARD_POS = (BORDER, BORDER)
COLOR_DARK = (181, 136, 99)
COLOR_LIGHT = (240, 217, 181)
COLOR_BG = (22, 21, 18)
IMAGE_PATH = "gui/images/"  # Path to the folder containing piece images


def create_board_surface():
    """Create the chessboard surface."""
    board_surface = pygame.Surface((TILE_SIZE * 8, TILE_SIZE * 8))
    dark = False
    for y in range(8):
        for x in range(8):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(board_surface, pygame.Color(COLOR_DARK if dark else COLOR_LIGHT), rect)
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
        self.game_over = False  # Track if the game is over

    def run(self):
        """Main game loop."""
        print(initial)
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif not self.game_over:  # Only allow moves if the game is not over
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
        square = 21 + row * 10 + col
        print(f"Mouse down at ({x}, {y}) -> Board square: ({row}, {col}) -> Sunfish square index: {square}")
        if 0 <= col < 8 and 0 <= row < 8 and self.position.board[square].isupper():
            self.selected_square = square
            print(f"Selected square: {self.selected_square} (Piece: {self.position.board[square]})")
        else:
            print("Invalid selection or no piece selected.")

    def handle_mouse_up(self):
        """Handle mouse button up events."""
        if self.selected_square is not None and not self.game_over:
            x, y = pygame.mouse.get_pos()
            col = (x - BOARD_POS[0]) // TILE_SIZE
            row = (y - BOARD_POS[1]) // TILE_SIZE
            square = 21 + row * 10 + col
            print(f"Mouse up at ({x}, {y}) -> Board square: ({row}, {col}) -> Sunfish square index: {square}")
            if 0 <= col < 8 and 0 <= row < 8:
                move = Move(self.selected_square, square, "")
                if move in self.get_legal_moves():
                    print(f"Move is valid. Moving piece from {self.selected_square} to {square}.")
                    self.position = self.position.move(move)
                    print(self.position.board)
                    self.selected_square = None
                    if self.check_game_over():
                        return
                    self.engine_move()
                else:
                    print("Invalid move! King would be in check.")
            print(f"Score: {self.position.score}")
            print(f"White castling rights: {self.position.wc}")
            print(f"Black castling rights: {self.position.bc}")
            print(f"En passant square: {self.position.ep}")
            print(f"King passant square: {self.position.kp}")

    def get_legal_moves(self):
        """Generate all legal moves that do not leave the king in check."""
        legal_moves = []
        for move in self.position.gen_moves():
            new_position = self.position.move(move)
            if not self.is_king_in_check(new_position):
                legal_moves.append(move)
        return legal_moves

    def is_king_in_check(self, position):
        """Check if the king is in check in the given position."""
        for move in position.gen_moves():
            if position.board[move.j].lower() == 'k':  # If the king is attacked
                return True
        return False

    def engine_move(self):
        """Let Sunfish calculate and make a move."""
        if not self.game_over:
            move = next(self.searcher.search([self.position]))[3]
            if move:
                print(f"Engine move: {move}")
                self.position = self.position.move(move)
                if self.check_game_over():
                    return

    def check_game_over(self):
        """Check if the game is over (checkmate or stalemate)."""
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            # No legal moves, check if the king is in check
            if self.is_king_in_check(self.position):
                print("Checkmate! Game over.")
                self.display_message("Checkmate! Game over.")
            else:
                print("Stalemate! Game over.")
                self.display_message("Stalemate! Game over.")
            self.game_over = True
            return True
        return False

    def display_message(self, message):
        """Display a message on the screen."""
        font = pygame.font.Font(None, 36)
        text = font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        pygame.time.wait(3000)  # Wait for 3 seconds before continuing

    def update_screen(self):
        """Update the game screen."""
        self.screen.fill(pygame.Color(COLOR_BG))
        self.screen.blit(self.board_surface, BOARD_POS)
        draw_pieces(self.screen, self.position)
        pygame.display.flip()

    