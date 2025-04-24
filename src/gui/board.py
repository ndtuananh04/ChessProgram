# import tkinter as tk

# class ChessBoard(tk.Canvas):
#     def __init__(self, parent, board_size=8, square_size=60):
#         super().__init__(parent, width=board_size * square_size, height=board_size * square_size)
#         self.board_size = board_size
#         self.square_size = square_size
#         self.draw_board()

#     def draw_board(self):
#         for row in range(self.board_size):
#             for col in range(self.board_size):
#                 color = "white" if (row + col) % 2 == 0 else "gray"
#                 x1 = col * self.square_size
#                 y1 = row * self.square_size
#                 x2 = x1 + self.square_size
#                 y2 = y1 + self.square_size
#                 self.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

#     def draw_piece(self, row, col, piece):
#         x = col * self.square_size + self.square_size // 2
#         y = row * self.square_size + self.square_size // 2
#         self.create_text(x, y, text=piece, font=("Arial", 24), fill="black")