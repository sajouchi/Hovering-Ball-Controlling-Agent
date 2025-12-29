import pygame
import imageio
import numpy as np

class ball_GUI():
    
    def __init__(self):
        pass
    
    def ball_gui(self): # makes the ball graphics
        pygame.init()
        self.WIDTH, self.HEIGHT = 600, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.WHITE = (255,255,255)
        self.BLACK = (0,0,0)
        self.RED = (200,0,0)

        self.MAX_HEIGHT = 20.0
        self.GROUND_Y = self.HEIGHT - 50
        self.TOP_Y = 50

        self.font = pygame.font.SysFont("consolas", 18)
        
    def setting_balls_height(self,h,v,action,fps):
        
        self.h=h
        self.v=v
        
        self.fps = fps
        
        self.screen.fill(self.WHITE)

        # ----- DRAW LIMITS -----
        pygame.draw.line(self.screen, self.BLACK, (0, self.GROUND_Y),
                        (self.WIDTH, self.GROUND_Y), 2)
        pygame.draw.line(self.screen, self.BLACK, (0, self.TOP_Y),
                        (self.WIDTH, self.TOP_Y), 2)

        # ----- DRAW BALL -----
        self.ball_y = self.HEIGHT - (self.h / self.MAX_HEIGHT) * self.HEIGHT
        pygame.draw.circle(
            self.screen, self.RED,
            (self.WIDTH // 2, int(self.ball_y)), 15
        )

        # TEXT OVERLAY (BOTTOM-RIGHT, ABOVE GROUND)
        self.padding = 10
        self.line_gap = 5

        self.action_map = {0: "↓", 1: "○", 2: "↑"}

        self.text_h = self.font.render(f"h : {h:.2f}", True, self.BLACK)
        self.text_v = self.font.render(f"v : {v:.2f}", True, self.BLACK)
        self.text_a = self.font.render(f"action : {self.action_map[action]}", True, self.BLACK)

        self.texts = [self.text_a, self.text_v, self.text_h]

        self.y = self.GROUND_Y - self.padding   # local variable different(not self.h)

        for text in self.texts:
            self.rect = text.get_rect()
            self.y -= self.rect.height
            self.rect.topleft = (
                self.WIDTH - self.rect.width - self.padding,
                self.y
            )
            self.screen.blit(text, self.rect)
            self.y -= self.line_gap

        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def renderer(self):
        pygame.event.pump()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
    
    def capture_gif(self):
        
        self.frame = pygame.surfarray.array3d(self.screen)
        self.frame = np.transpose(self.frame,(1,0,2)) # pygames gives (width,height,color)/(1,0,2) for gif we need (height,width,color)/(1,0,2)

        return self.frame