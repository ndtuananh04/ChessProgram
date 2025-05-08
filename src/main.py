import pygame
import sys
from gui.human_engine import GameWindow
from gui.engine_engine import EngineBattle

def main_menu():
    pygame.init()
    screen = pygame.display.set_mode((500, 300))
    pygame.display.set_caption("Chess Program")
    
    font_title = pygame.font.Font(None, 48)
    font_options = pygame.font.Font(None, 36)
    
    title = font_title.render("Chess Program", True, (255, 255, 255))
    option1 = font_options.render("1. Human vs Engine", True, (255, 255, 255))
    option2 = font_options.render("2. Engine vs Engine", True, (255, 255, 255))
    exit_text = font_options.render("3. Exit", True, (255, 255, 255))
    
    running = True
    while running:
        screen.fill((22, 21, 18))  # Same bg color as the game
        
        # Get the center of the screen
        center_x = screen.get_width() // 2
        
        # Draw title and options
        title_rect = title.get_rect(center=(center_x, 50))
        screen.blit(title, title_rect)
        
        option1_rect = option1.get_rect(center=(center_x, 120))
        screen.blit(option1, option1_rect)
        
        option2_rect = option2.get_rect(center=(center_x, 170))
        screen.blit(option2, option2_rect)
        
        exit_rect = exit_text.get_rect(center=(center_x, 220))
        screen.blit(exit_text, exit_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "human_vs_engine"
                elif event.key == pygame.K_2:
                    return "engine_vs_engine"
                elif event.key == pygame.K_3 or event.key == pygame.K_ESCAPE:
                    return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if option1_rect.collidepoint(mouse_pos):
                    return "human_vs_engine"
                elif option2_rect.collidepoint(mouse_pos):
                    return "engine_vs_engine"
                elif exit_rect.collidepoint(mouse_pos):
                    return None

def main():
    while True:
        choice = main_menu()
        
        if choice is None:
            break  # Exit the program
        elif choice == "human_vs_engine":
            game = GameWindow()
            game.run()
        elif choice == "engine_vs_engine":
            battle = EngineBattle()
            battle.run()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()