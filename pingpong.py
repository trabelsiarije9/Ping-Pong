import pygame
import socket
import threading
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------------
# Configuration and Constants
# -------------------------------------------------------
WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 10
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100

# Colors
WHITE      = (255, 255, 255)
BLACK      = (0, 0, 0)
LILAC      = (200, 162, 200)    #Pour Joueur 1
BABY_GREEN = (144, 238, 144)    #Pour Joueur 2
CHAT_BTN_COLOR = (100, 100, 255)
CHAT_BG    = (50, 50, 50)

# Chat network settings 
SERVER_HOST = 'localhost'
SERVER_PORT = 12345
is_host = False  

# -------------------------------------------------------
# Chat Functions 
# -------------------------------------------------------
def start_chat_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind((SERVER_HOST, SERVER_PORT))
    except Exception as e:
        print("Error binding server:", e)
        return
    server.listen(2)
    clients = []
    print("Chat server started on port", SERVER_PORT)
    def handle_client(client_sock):
        while True:
            try:
                msg = client_sock.recv(1024)
                if not msg:
                    break
                print("Server received:", msg.decode())
                for c in clients:
                    if c != client_sock:
                        try:
                            c.send(msg)
                        except Exception as e:
                            print("Error sending to client:", e)
            except Exception as e:
                print("Error handling client:", e)
                break
        client_sock.close()
        if client_sock in clients:
            clients.remove(client_sock)
        print("Client disconnected.")
    while True:
        try:
            client_sock, addr = server.accept()
            print("Client connected from", addr)
            clients.append(client_sock)
            client_thread = threading.Thread(target=handle_client, args=(client_sock,))
            client_thread.daemon = True
            client_thread.start()
        except Exception as e:
            print("Server accept error:", e)
            break

def start_chat_client():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_HOST, SERVER_PORT))
    print("Connected to chat server at {}:{}".format(SERVER_HOST, SERVER_PORT))
    return sock

def chat_receive_thread(sock):
    global chat_messages
    while True:
        try:
            msg = sock.recv(1024).decode()
            if msg:
                print("Received chat message:", msg)
                chat_messages.append(msg)
        except Exception as e:
            print("Error in chat receiving thread:", e)
            break

# -------------------------------------------------------
# DQN Agent Definition using PyTorch
# -------------------------------------------------------
class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class DQNAgentWrapper:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQNAgent(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
    
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))

# -------------------------------------------------------
# Environment Simulation for Training the AI Agent
# -------------------------------------------------------
def simulate_episode(agent):
    # Initialize environment state
    ball_x = WIDTH // 2
    ball_y = HEIGHT // 2
    ball_dx = 5
    ball_dy = 5
    paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2  # AI controls the right paddle
    done = False
    total_reward = 0
    steps = 0
    max_speed = 10.0
    while not done and steps < 500:
        # State: normalized [ball_x, ball_y, ball_dx, ball_dy, paddle_y]
        state = np.array([ball_x / WIDTH, ball_y / HEIGHT, ball_dx / max_speed, ball_dy / max_speed, paddle_y / HEIGHT])
        action = agent.act(state)
        # Actions: 0 = do nothing, 1 = move up, 2 = move down
        if action == 1:
            paddle_y -= 7
        elif action == 2:
            paddle_y += 7
        paddle_y = max(0, min(HEIGHT - PADDLE_HEIGHT, paddle_y))
        # Update ball position
        ball_x += ball_dx
        ball_y += ball_dy
        if ball_y - BALL_RADIUS <= 0 or ball_y + BALL_RADIUS >= HEIGHT:
            ball_dy = -ball_dy
        # When the ball reaches the AI paddle side:
        if ball_x >= WIDTH - PADDLE_WIDTH:
            if paddle_y < ball_y < paddle_y + PADDLE_HEIGHT:
                reward = 1.0
                ball_dx = -ball_dx  # reflect the ball
            else:
                reward = -1.0
                done = True
        else:
            reward = 0.0
        total_reward += reward
        next_state = np.array([ball_x / WIDTH, ball_y / HEIGHT, ball_dx / max_speed, ball_dy / max_speed, paddle_y / HEIGHT])
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        steps += 1
    return total_reward

def train_ai_agent(episodes=1000):
    state_size = 5
    action_size = 3
    agent = DQNAgentWrapper(state_size, action_size)
    for episode in range(episodes):
        total_reward = simulate_episode(agent)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
    agent.save("dqn_pong.pth")
    print("Training complete, model saved as dqn_pong.pth")
    return agent

# -------------------------------------------------------
# Pygame Initialization and Global Variables
# -------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ping Pong with Chat and AI")
clock = pygame.time.Clock()

# Chat variables
chat_visible = False
chat_input = ""
chat_messages = []
chat_font = pygame.font.Font(None, 24)
chat_overlay_rect = pygame.Rect(50, HEIGHT - 150, WIDTH - 100, 100)
chat_button_rect = pygame.Rect(WIDTH - 100, 10, 80, 30)
client_socket = None

# Start chat networking (optional)
if is_host:
    server_thread = threading.Thread(target=start_chat_server)
    server_thread.daemon = True
    server_thread.start()
try:
    client_socket = start_chat_client()
    recv_thread = threading.Thread(target=chat_receive_thread, args=(client_socket,))
    recv_thread.daemon = True
    recv_thread.start()
except Exception as e:
    print("Could not connect to chat server:", e)

# Game variables and reset function
def reset_game():
    global ball_x, ball_y, ball_dx, ball_dy
    global left_paddle_y, right_paddle_y, left_score, right_score, winner
    ball_x = WIDTH // 2
    ball_y = HEIGHT // 2
    ball_dx, ball_dy = 5, 5
    left_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
    right_paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
    left_score = 0
    right_score = 0
    winner = None

reset_game()
winning_score = 10
font = pygame.font.Font(None, 36)
title_font = pygame.font.Font(None, 40)
win_font = pygame.font.Font(None, 50)

# -------------------------------------------------------
# Main Menu (Mode Selection)
# -------------------------------------------------------
def main_menu():
    menu = True
    selected_mode = None  # Options: "two_player", "pvc", "train"
    while menu:
        screen.fill(BLACK)
        title = title_font.render("Ping Pong - Main Menu", True, WHITE)
        option1 = font.render("1. Two Players", True, WHITE)
        option2 = font.render("2. Player vs Computer", True, WHITE)
        option3 = font.render("3. Train AI Agent", True, WHITE)
        option4 = font.render("Q. Quit", True, WHITE)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
        screen.blit(option1, (WIDTH//2 - option1.get_width()//2, 200))
        screen.blit(option2, (WIDTH//2 - option2.get_width()//2, 250))
        screen.blit(option3, (WIDTH//2 - option3.get_width()//2, 300))
        screen.blit(option4, (WIDTH//2 - option4.get_width()//2, 350))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    selected_mode = "two_player"
                    menu = False
                elif event.key == pygame.K_2:
                    selected_mode = "pvc"
                    menu = False
                elif event.key == pygame.K_3:
                    selected_mode = "train"
                    menu = False
                elif event.key == pygame.K_q:
                    pygame.quit()
                    exit()
        clock.tick(15)
    return selected_mode

mode = main_menu()
ai_agent = None
if mode == "train":
    ai_agent = train_ai_agent(episodes=500)  # Adjust episode count as desired
    mode = "pvc"  # After training, switch to player vs. computer mode
elif mode == "pvc":
    # Try to load a pre-trained model if available
    state_size = 5
    action_size = 3
    ai_agent = DQNAgentWrapper(state_size, action_size)
    try:
        ai_agent.load("dqn_pong.pth")
        print("Loaded trained model.")
    except Exception as e:
        print("No trained model found, using untrained agent.")

# -------------------------------------------------------
# Main Game Loop
# -------------------------------------------------------
running = True
while running:
    screen.fill(BLACK)
    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Toggle chat overlay when clicking the Chat button.
        if event.type == pygame.MOUSEBUTTONDOWN:
            if chat_button_rect.collidepoint(event.pos):
                chat_visible = not chat_visible
                print("Chat overlay toggled:", chat_visible)
        # If chat overlay is visible, handle text input.
        if chat_visible:
            if event.type == pygame.KEYDOWN:
                print("Chat key pressed:", event.key)
                if event.key == pygame.K_RETURN:
                    if chat_input.strip() != "":
                        if client_socket:
                            try:
                                print("Sending message:", chat_input)
                                client_socket.send(chat_input.encode())
                            except Exception as e:
                                print("Error sending message:", e)
                        chat_messages.append("Me: " + chat_input)
                        chat_input = ""
                elif event.key == pygame.K_BACKSPACE:
                    chat_input = chat_input[:-1]
                else:
                    chat_input += event.unicode
        else:
            # Game controls (when chat is not active)
            if winner is None:
                keys = pygame.key.get_pressed()
                # Left paddle (player 1) controls
                if keys[pygame.K_w] and left_paddle_y > 0:
                    left_paddle_y -= 7
                if keys[pygame.K_s] and left_paddle_y < HEIGHT - PADDLE_HEIGHT:
                    left_paddle_y += 7
                # Right paddle: In two-player mode, use arrow keys.
                if mode == "two_player":
                    if keys[pygame.K_UP] and right_paddle_y > 0:
                        right_paddle_y -= 7
                    if keys[pygame.K_DOWN] and right_paddle_y < HEIGHT - PADDLE_HEIGHT:
                        right_paddle_y += 7
            # After game over, allow restart or quit.
            if winner is not None and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_game()
                elif event.key == pygame.K_q:
                    running = False
    # Game Logic avant gagner
    if winner is None:
        ball_x += ball_dx
        ball_y += ball_dy
        if ball_y - BALL_RADIUS <= 0 or ball_y + BALL_RADIUS >= HEIGHT:
            ball_dy = -ball_dy
        # Collision with left paddle
        if ball_x - BALL_RADIUS <= PADDLE_WIDTH and left_paddle_y < ball_y < left_paddle_y + PADDLE_HEIGHT:
            ball_dx = -ball_dx
        # Collision with right paddle
        if ball_x + BALL_RADIUS >= WIDTH - PADDLE_WIDTH and right_paddle_y < ball_y < right_paddle_y + PADDLE_HEIGHT:
            ball_dx = -ball_dx
        # Scoring
        if ball_x < 0:
            right_score += 1
            ball_x, ball_y = WIDTH // 2, HEIGHT // 2
            ball_dx = -ball_dx
        if ball_x > WIDTH:
            left_score += 1
            ball_x, ball_y = WIDTH // 2, HEIGHT // 2
            ball_dx = -ball_dx
        if left_score >= winning_score:
            winner = "Joueur 1"
        if right_score >= winning_score:
            winner = "Joueur 2"
        # In player vs. computer mode, let the AI control the right paddle.
        if mode == "pvc" and ai_agent is not None:
            state_ai = np.array([ball_x / WIDTH, ball_y / HEIGHT, ball_dx/10, ball_dy/10, right_paddle_y/HEIGHT])
            action = ai_agent.act(state_ai)
            if action == 1:  # move up
                right_paddle_y -= 7
            elif action == 2:  # move down
                right_paddle_y += 7
            right_paddle_y = max(0, min(HEIGHT - PADDLE_HEIGHT, right_paddle_y))
    # Drawing Game Elements
    pygame.draw.rect(screen, WHITE, (0, left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, WHITE, (WIDTH - PADDLE_WIDTH, right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.circle(screen, WHITE, (int(ball_x), int(ball_y)), BALL_RADIUS)
    score_text = font.render(f"{left_score} - {right_score}", True, WHITE)
    screen.blit(score_text, (WIDTH // 2 - 20, 20))
    title_left = title_font.render("Joueur Numéro 1", True, LILAC)
    title_right = title_font.render("Joueur Numéro 2", True, BABY_GREEN)
    screen.blit(title_left, (50, 20))
    screen.blit(title_right, (WIDTH - 250, 20))
    if winner:
        win_text = win_font.render(f"{winner} a gagné!", True, LILAC if winner == "Joueur 1" else BABY_GREEN)
        screen.blit(win_text, (WIDTH // 2 - 150, HEIGHT // 2 - 30))
        instruction_text = font.render("Press R to Restart or Q to Quit", True, WHITE)
        screen.blit(instruction_text, (WIDTH // 2 - 170, HEIGHT // 2 + 30))
    # Drawing Chat Interface
    pygame.draw.rect(screen, CHAT_BTN_COLOR, chat_button_rect)
    chat_btn_text = chat_font.render("Chat", True, WHITE)
    screen.blit(chat_btn_text, (chat_button_rect.x + 10, chat_button_rect.y + 5))
    if chat_visible:
        pygame.draw.rect(screen, CHAT_BG, chat_overlay_rect)
        input_surface = chat_font.render(">" + chat_input, True, WHITE)
        screen.blit(input_surface, (chat_overlay_rect.x + 5, chat_overlay_rect.y + chat_overlay_rect.height - 25))
        y_offset = chat_overlay_rect.y + 5
        for msg in chat_messages[-5:]:
            msg_surface = chat_font.render(msg, True, WHITE)
            screen.blit(msg_surface, (chat_overlay_rect.x + 5, y_offset))
            y_offset += 20
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
