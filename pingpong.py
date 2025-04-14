import pygame
import sys
import threading
import socket
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Global Constants and Settings
# ------------------------------

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100
BALL_SIZE = 15

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CHAT_BG = (50, 50, 50)
CHAT_TEXT_COLOR = (200, 200, 200)

# Global list to store chat messages
chat_messages = []
chat_lock = threading.Lock()

# ------------------------------
# Chat Server Code
# ------------------------------

class ChatServer(threading.Thread):
    """
    A simple chat server that listens for client connections, then broadcasts any received message
    to all connected clients.
    """
    def __init__(self, host="127.0.0.1", port=12345):
        super().__init__()
        self.host = host
        self.port = port
        self.clients = []
        self.clients_lock = threading.Lock()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.daemon = True

    def broadcast(self, message, source=None):
        with self.clients_lock:
            for client in self.clients:
                # Send the message to all clients (including the sender)
                try:
                    client.sendall(message.encode('utf-8'))
                except Exception:
                    self.clients.remove(client)

    def run(self):
        print(f"[ChatServer] Running on {self.host}:{self.port}")
        while True:
            try:
                client_socket, addr = self.server_socket.accept()
                with self.clients_lock:
                    self.clients.append(client_socket)
                threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()
            except Exception as e:
                print("[ChatServer] Error accepting connection:", e)
                break

    def handle_client(self, client_socket):
        while True:
            try:
                data = client_socket.recv(1024)
                if data:
                    message = data.decode('utf-8')
                    # Append to global chat messages list
                    with chat_lock:
                        chat_messages.append(message)
                    # Broadcast the message to all clients
                    self.broadcast(message)
                else:
                    with self.clients_lock:
                        if client_socket in self.clients:
                            self.clients.remove(client_socket)
                    break
            except Exception as e:
                with self.clients_lock:
                    if client_socket in self.clients:
                        self.clients.remove(client_socket)
                break

# ------------------------------
# Chat Client Code
# ------------------------------

class ChatClient(threading.Thread):
    """
    A chat client that connects to the ChatServer and constantly listens for new messages.
    """
    def __init__(self, host="127.0.0.1", port=12345):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.daemon = True

    def send_message(self, message):
        try:
            self.sock.sendall(message.encode('utf-8'))
        except Exception as e:
            print("[ChatClient] Send error:", e)

    def run(self):
        while True:
            try:
                data = self.sock.recv(1024)
                if data:
                    message = data.decode('utf-8')
                    with chat_lock:
                        if message not in chat_messages:
                            chat_messages.append(message)
                else:
                    break
            except Exception as e:
                break

# ------------------------------
# AI Agent using PyTorch
# ------------------------------

class PingPongAI(nn.Module):
    """
    A simple feed-forward network to simulate a trained AI agent for Ping Pong.
    The network receives 4 inputs and outputs a decision (3 possible actions: up, down, or stay).
    """
    def __init__(self):
        super(PingPongAI, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)  # outputs: 0=up, 1=down, 2=stay

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the AI and set to eval mode (simulating that it is trained)
ai_agent = PingPongAI()
ai_agent.eval()

# For demonstration, we are not loading any pretrained weights. In an actual project,
# you would load a trained model using: ai_agent.load_state_dict(torch.load('model.pth'))

# ------------------------------
# Game Code (PyGame)
# ------------------------------

def draw_chat(screen, font, chat_input, chat_rect):
    """ Draws the chat messages and the input box on the screen. """
    # Define the chat display area (we use the bottom 150 pixels of the screen)
    chat_area_height = 150
    chat_area = pygame.Rect(0, SCREEN_HEIGHT - chat_area_height, SCREEN_WIDTH, chat_area_height)
    pygame.draw.rect(screen, CHAT_BG, chat_area)

    # Display last few messages (limit to 6 messages)
    with chat_lock:
        recent_msgs = chat_messages[-6:]
    offset_y = SCREEN_HEIGHT - chat_area_height + 5
    for msg in recent_msgs:
        msg_surface = font.render(msg, True, CHAT_TEXT_COLOR)
        screen.blit(msg_surface, (5, offset_y))
        offset_y += 20

    # Draw input box
    input_box = pygame.Rect(5, SCREEN_HEIGHT - 30, SCREEN_WIDTH - 10, 25)
    pygame.draw.rect(screen, WHITE, input_box, 2)
    input_surface = font.render(chat_input, True, WHITE)
    screen.blit(input_surface, (input_box.x + 5, input_box.y + 3))

def game_loop(two_player=True, chat_client=None):
    """ Main game loop for a match. If two_player is False, one human plays versus the AI agent. """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Ping Pong Game")
    clock = pygame.time.Clock()

    # Fonts for text and chat
    font = pygame.font.SysFont("consolas", 20)
    large_font = pygame.font.SysFont("consolas", 50)

    # Paddle positions
    paddle1_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
    paddle2_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2

    # Paddle speeds
    paddle_speed = 7

    # Ball position and velocity
    ball_x = SCREEN_WIDTH // 2 - BALL_SIZE // 2
    ball_y = SCREEN_HEIGHT // 2 - BALL_SIZE // 2
    ball_vel_x = 5 * (1 if torch.rand(1).item() > 0.5 else -1)
    ball_vel_y = 5 * (1 if torch.rand(1).item() > 0.5 else -1)

    # Scores
    score1 = 0
    score2 = 0

    # For chat input
    chat_input = ""
    chat_active = False

    run_game = True
    win_message = ""
    while run_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Process key presses for chat input and game commands
            if event.type == pygame.KEYDOWN:
                if chat_active:
                    if event.key == pygame.K_RETURN:
                        # Send chat message (prepends a tag so we know which mode)
                        full_message = f"[Player]: {chat_input}"
                        if chat_client is not None:
                            chat_client.send_message(full_message)
                        else:
                            with chat_lock:
                                chat_messages.append(full_message)
                        chat_input = ""
                        chat_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        chat_input = chat_input[:-1]
                    else:
                        chat_input += event.unicode
                else:
                    # Toggle chat input if TAB is pressed
                    if event.key == pygame.K_TAB:
                        chat_active = True

                    # Restart or Quit if match ended
                    if win_message:
                        if event.key == pygame.K_r:
                            run_game = False
                            return "restart"
                        if event.key == pygame.K_q:
                            pygame.quit()
                            sys.exit()

        keys = pygame.key.get_pressed()
        # Handle paddle movement for player 1 (W/S)
        if keys[pygame.K_w]:
            paddle1_y -= paddle_speed
        if keys[pygame.K_s]:
            paddle1_y += paddle_speed

        # Boundaries for paddle1
        paddle1_y = max(0, min(SCREEN_HEIGHT - PADDLE_HEIGHT, paddle1_y))

        if two_player:
            # For player 2, use UP/DOWN arrow keys.
            if keys[pygame.K_UP]:
                paddle2_y -= paddle_speed
            if keys[pygame.K_DOWN]:
                paddle2_y += paddle_speed
        else:
            # Use AI for paddle2 movement.
            # Create a simple state vector: [ball_y, ball_x, paddle2_y, ball_vel_y]
            state = torch.tensor([ball_y/SCREEN_HEIGHT, ball_x/SCREEN_WIDTH, paddle2_y/SCREEN_HEIGHT, ball_vel_y/10.0])
            # Run through the AI model
            with torch.no_grad():
                output = ai_agent(state)
            action = torch.argmax(output).item()
            # Map action to movement: 0 -> up, 1 -> down, 2 -> stay.
            if action == 0:
                paddle2_y -= paddle_speed
            elif action == 1:
                paddle2_y += paddle_speed

        # Boundaries for paddle2
        paddle2_y = max(0, min(SCREEN_HEIGHT - PADDLE_HEIGHT, paddle2_y))

        # Update ball position
        ball_x += ball_vel_x
        ball_y += ball_vel_y

        # Ball collision with top/bottom walls
        if ball_y <= 0 or ball_y >= SCREEN_HEIGHT - BALL_SIZE:
            ball_vel_y *= -1

        # Ball collision with paddles
        # Left paddle collision
        if ball_x <= PADDLE_WIDTH:
            if paddle1_y <= ball_y <= paddle1_y + PADDLE_HEIGHT:
                ball_vel_x *= -1
            else:
                score2 += 1
                # Reset ball position
                ball_x, ball_y = SCREEN_WIDTH // 2 - BALL_SIZE // 2, SCREEN_HEIGHT // 2 - BALL_SIZE // 2
                ball_vel_x = 5 * (1 if torch.rand(1).item() > 0.5 else -1)
                ball_vel_y = 5 * (1 if torch.rand(1).item() > 0.5 else -1)

        # Right paddle collision
        if ball_x >= SCREEN_WIDTH - PADDLE_WIDTH - BALL_SIZE:
            if paddle2_y <= ball_y <= paddle2_y + PADDLE_HEIGHT:
                ball_vel_x *= -1
            else:
                score1 += 1
                # Reset ball
                ball_x, ball_y = SCREEN_WIDTH // 2 - BALL_SIZE // 2, SCREEN_HEIGHT // 2 - BALL_SIZE // 2
                ball_vel_x = 5 * (1 if torch.rand(1).item() > 0.5 else -1)
                ball_vel_y = 5 * (1 if torch.rand(1).item() > 0.5 else -1)

        # Check if any player reaches 10 points
        if score1 >= 10 or score2 >= 10:
            if two_player:
                win_message = "Player 1 Wins!" if score1 >= 10 else "Player 2 Wins!"
            else:
                win_message = "Player 1 Wins!" if score1 >= 10 else "AI Agent Wins!"
        
        # Draw Everything
        screen.fill(BLACK)

        # Draw paddles and ball
        paddle1 = pygame.Rect(5, paddle1_y, PADDLE_WIDTH, PADDLE_HEIGHT)
        paddle2 = pygame.Rect(SCREEN_WIDTH - 5 - PADDLE_WIDTH, paddle2_y, PADDLE_WIDTH, PADDLE_HEIGHT)
        pygame.draw.rect(screen, WHITE, paddle1)
        pygame.draw.rect(screen, WHITE, paddle2)
        ball = pygame.Rect(ball_x, ball_y, BALL_SIZE, BALL_SIZE)
        pygame.draw.rect(screen, WHITE, ball)

        # Draw scores
        score_text = font.render(f"{score1}  :  {score2}", True, WHITE)
        screen.blit(score_text, (SCREEN_WIDTH//2 - score_text.get_width()//2, 20))

        # Draw win message if applicable
        if win_message:
            win_surface = large_font.render(win_message, True, WHITE)
            screen.blit(win_surface, (SCREEN_WIDTH//2 - win_surface.get_width()//2, SCREEN_HEIGHT//2 - win_surface.get_height()//2))
            prompt_surface = font.render("Press Q to Quit or R to Restart", True, WHITE)
            screen.blit(prompt_surface, (SCREEN_WIDTH//2 - prompt_surface.get_width()//2, SCREEN_HEIGHT//2 + win_surface.get_height()))
        
        # Draw Chat area and input box
        draw_chat(screen, font, chat_input, None)

        pygame.display.flip()
        clock.tick(FPS)

    return "quit"


def main_menu(chat_client):
    """ Displays the main menu and processes user input for game mode selection or quit. """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Ping Pong Main Menu")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 30)
    
    menu_active = True
    while menu_active:
        screen.fill(BLACK)
        title = font.render("Ping Pong Game", True, WHITE)
        option1 = font.render("1 - Two Player", True, WHITE)
        option2 = font.render("2 - Play vs AI Agent", True, WHITE)
        optionQ = font.render("Q - Quit", True, WHITE)
        instruction = font.render("Type TAB to open chat input", True, CHAT_TEXT_COLOR)

        screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 150))
        screen.blit(option1, (SCREEN_WIDTH//2 - option1.get_width()//2, 220))
        screen.blit(option2, (SCREEN_WIDTH//2 - option2.get_width()//2, 260))
        screen.blit(optionQ, (SCREEN_WIDTH//2 - optionQ.get_width()//2, 300))
        screen.blit(instruction, (10, SCREEN_HEIGHT - 40))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                menu_active = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    mode = game_loop(two_player=True, chat_client=chat_client)
                    if mode == "restart":
                        continue
                elif event.key == pygame.K_2:
                    mode = game_loop(two_player=False, chat_client=chat_client)
                    if mode == "restart":
                        continue
                elif event.key == pygame.K_q:
                    menu_active = False
                    pygame.quit()
                    sys.exit()
        clock.tick(30)

# ------------------------------
# Main Execution: Start Server, Client, and Menu
# ------------------------------

def main():
    # Start the chat server in a background thread.
    chat_server = ChatServer()
    chat_server.start()
    time.sleep(0.5)  # slight delay to ensure the server is running

    # Start the chat client.
    chat_client = ChatClient()
    chat_client.start()

    # Enter the main menu.
    main_menu(chat_client)

if __name__ == '__main__':
    main()

