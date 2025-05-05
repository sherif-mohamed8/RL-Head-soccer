import pygame
import time
import numpy as np
from settings import *
from objects import Player, Ball
from ai import ai_move
from q_learning import QLearningAgent, q_move


class SoccerGame:
    def __init__(self, use_q_learning=True, training_mode=True):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Soccer Game with Q-Learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 36)
        self.small_font = pygame.font.SysFont('Arial', 18)
        self.large_font = pygame.font.SysFont('Arial', 72)
        self.field_rect = pygame.Rect(50, 50, WIDTH - 100, HEIGHT - 100)

        # متغيرات التحكم في حجم الملعب
        self.is_large_field = False
        self.field_changed = False

        # initialize game state on team-select screen
        self.game_state = STATE_TEAM_SELECT
        self.p3_team   = None  # 1 = red, 2 = blue

        # Q-learning configuration
        self.use_q_learning = use_q_learning
        self.training_mode  = training_mode
        if self.use_q_learning:
            self.q_agent = QLearningAgent()
            self.auto_train = False
            self.max_episodes = 1000
            self.current_frame = 0
            self.max_frames_per_episode = 1800  # ~30 seconds at 60 FPS

        # now set up players, ball, etc.
        self.reset()

    def toggle_field_size(self):
        """تغيير حجم الملعب بين الحجم الطبيعي والكبير"""
        global WIDTH, HEIGHT
        
        if self.is_large_field:
            WIDTH, HEIGHT = NORMAL_WIDTH, NORMAL_HEIGHT
        else:
            WIDTH, HEIGHT = LARGE_WIDTH, LARGE_HEIGHT
            
        self.is_large_field = not self.is_large_field
        self.field_changed = True
        
        # تحديث حجم النافذة
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.field_rect = pygame.Rect(50, 50, WIDTH-100, HEIGHT-100)

    def reset(self):
        # تغيير حجم الملعب في بداية كل دورة
        self.toggle_field_size()
        
        # إعادة تعيين مواقع اللاعبين الأحمر والأزرق
        self.p1 = Player(WIDTH//4, HEIGHT//2, RED)
        self.p2 = Player(3*WIDTH//4, HEIGHT//2, BLUE, ai=True)

        # إنشاء اللاعب الثالث (الأصفر)
        # إنشاء اللاعب الثالث
        if self.p3_team is None:
            # قبل اختيار الفريق: أعلى المركز قليلاً
            spawn_x = WIDTH // 2
            spawn_y = HEIGHT // 2 - 100
        else:
            # بعد اختيار الفريق: الجانب الأعلى للملعب
            spawn_x = WIDTH//4 if self.p3_team == 1 else 3*WIDTH//4
            spawn_y = HEIGHT // 2 - 100
        self.p3 = Player(spawn_x, spawn_y, YELLOW)

        
        # إعادة تعيين الكرة والنتيجة وبقية الحالة
        self.ball  = Ball(WIDTH//2, HEIGHT//2, self.field_rect, is_large_field=self.is_large_field)
        self.score         = [0, 0]
        self.game_active   = False
        self.countdown     = 3
        self.last_count    = time.time()
        self.winner        = None
        self.current_frame = 0
        
        # إنهاء حلقة تدريب Q-learning السابقة إذا لزم الأمر
        if hasattr(self, 'q_agent') and self.use_q_learning:
            self.q_agent.end_episode()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # ← اختيار الفريق بالأسهم ←
            if self.game_state == STATE_TEAM_SELECT and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    # انضمام إلى الفريق الأحمر
                    self.p3_team = 1
                    self.game_state = STATE_PLAYING
                    self.reset()
                    return True
                elif event.key == pygame.K_RIGHT:
                    # انضمام إلى الفريق الأزرق
                    self.p3_team = 2
                    self.game_state = STATE_PLAYING
                    self.reset()
                    return True

            # معالجة بقية المفاتيح بعد بدء اللعب
            if event.type == pygame.KEYDOWN and self.game_state != STATE_TEAM_SELECT:
                if event.key == pygame.K_r and self.winner:
                    self.reset()
                elif event.key == pygame.K_q and self.use_q_learning:
                    # Toggle Q-learning
                    self.use_q_learning = not self.use_q_learning
                    print(f"Q-learning: {'ON' if self.use_q_learning else 'OFF'}")
                elif event.key == pygame.K_t and self.use_q_learning:
                    # Toggle training mode
                    self.training_mode = not self.training_mode
                    print(f"Training mode: {'ON' if self.training_mode else 'OFF'}")
                elif event.key == pygame.K_y and self.use_q_learning:
                    # Toggle auto-training
                    self.auto_train = not self.auto_train
                    if self.auto_train:
                        self.training_mode = True
                    print(f"Auto-training: {'ON' if self.auto_train else 'OFF'}")
                elif event.key == pygame.K_e and self.use_q_learning:
                    # Save Q-table
                    self.q_agent.save()
                elif event.key == pygame.K_l and self.use_q_learning:
                    # Load Q-table
                    self.q_agent.load()

        return True

    def update(self):
        # تأكد من انتهاء العد التنازلي قبل بدء اللعب
        if not self._countdown_logic():
            return

        keys = pygame.key.get_pressed()

        # 1. حركة اللاعب الأحمر (WASD)
        self.p1.move(
            keys, self.field_rect,
            up=PLAYER1_CONTROLS['up'],    down=PLAYER1_CONTROLS['down'],
            left=PLAYER1_CONTROLS['left'], right=PLAYER1_CONTROLS['right']
        )

        # 2. حركة اللاعب الأصفر (أسهم)
        self.p3.move(
            keys, self.field_rect,
            up=PLAYER3_CONTROLS['up'],    down=PLAYER3_CONTROLS['down'],
            left=PLAYER3_CONTROLS['left'], right=PLAYER3_CONTROLS['right']
        )

        # 3. حركة اللاعب الآلي (أزرق)
        if self.use_q_learning:
            q_move(self.p2, self.ball, self.p1, self.q_agent, training=self.training_mode)
        else:
            ai_move(self.p2, self.ball)

        # 4. تحديث الكرة وتصادمها مع الجميع
        self.ball.update()
        self.ball.collide_with_player(self.p1)
        self.ball.collide_with_player(self.p2)
        self.ball.collide_with_player(self.p3)   # ← تصادم مع اللاعب الثالث

        # 5. فحص تسجيل الأهداف
        self._check_goal()
        
        # 6. منطق التدريب التلقائي
        if self.auto_train and self.use_q_learning:
            self.current_frame += 1
            
            # إعادة تعيين إذا طالت الحلقة أكثر من الحد
            if self.current_frame >= self.max_frames_per_episode:
                self.reset()
                
            # إنهاء التدريب إذا وصلنا للحد الأقصى
            if self.q_agent.episode_count >= self.max_episodes:
                self.auto_train = False
                self.training_mode = False
                self.q_agent.save()
                print(f"Training complete after {self.max_episodes} episodes")

    def _countdown_logic(self):
        if not self.game_active:
            if time.time() - self.last_count > 1:
                self.countdown -= 1
                self.last_count = time.time()
                if self.countdown <= 0:
                    self.game_active = True
            return False
        return True

    def _check_goal(self):
        goal = self.ball.check_goal()
        if goal == 1:
            self.score[1] += 1
            self._after_goal()
        elif goal == 2:
            self.score[0] += 1
            self._after_goal()

        if self.score[0] >= WINNING_SCORE:
            self.winner = "Player 1"
            self.game_active = False
        elif self.score[1] >= WINNING_SCORE:
            self.winner = "Player 2"
            self.game_active = False

    def _after_goal(self):
        # إعادة ضبط مواضع اللاعبين بعد تسجيل هدف
        self.p1.reset(WIDTH//4, HEIGHT//2)
        self.p2.reset(3*WIDTH//4, HEIGHT//2)

        # اللاعب الثالث (أصفر): استخدم الموضع الابتدائي المُنزاح عن المنتصف
        spawn_x = WIDTH//4 if self.p3_team == 1 else 3*WIDTH//4
        spawn_y = HEIGHT//2 - 100   # نفس الارتفاع الذي وضعناه في reset()
        self.p3.reset(spawn_x, spawn_y)

        # إعادة ضبط الكرة والعدادات
        self.ball.reset()
        self.countdown   = 2
        self.last_count  = time.time()
        self.game_active = False

    
    def _render_team_select(self):
        # 1. خلفية الملعب
        self.screen.fill(GREEN)
        pygame.draw.rect(self.screen, WHITE, self.field_rect, 2)
        pygame.draw.line(self.screen, WHITE, (WIDTH//2, 50), (WIDTH//2, HEIGHT-50), 2)
        pygame.draw.circle(self.screen, WHITE, (WIDTH//2, HEIGHT//2), 70, 2)
        pygame.draw.rect(self.screen, WHITE, (0, HEIGHT//2-GOAL_WIDTH//2, 20, GOAL_WIDTH), 2)
        pygame.draw.rect(self.screen, WHITE, (WIDTH-20, HEIGHT//2-GOAL_WIDTH//2, 20, GOAL_WIDTH), 2)

        # 2. رسم مثال للاعبين (مربعات حمراء وزرقاء) لتوضيح الاختيار
        pygame.draw.rect(self.screen, RED,  (WIDTH//4 - PLAYER_SIZE//2, HEIGHT//2 - PLAYER_SIZE//2, PLAYER_SIZE, PLAYER_SIZE))
        pygame.draw.rect(self.screen, BLUE, (3*WIDTH//4 - PLAYER_SIZE//2, HEIGHT//2 - PLAYER_SIZE//2, PLAYER_SIZE, PLAYER_SIZE))

        # 3. نصّ العنوان والتعليمات
        title_surf = self.large_font.render("Choose Team", True, WHITE)
        instr_surf = self.small_font.render("Right Or Left", True, WHITE)
        self.screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, 50))
        self.screen.blit(instr_surf, (WIDTH//2 - instr_surf.get_width()//2, HEIGHT - 50))

        # 4. أزرار الاختيار (مربعات قابلة للنقر)
        # نرسم مستطيل حول كل مربع لدلالة إمكانية النقر
        pygame.draw.rect(self.screen, WHITE, 
                         (WIDTH//4 - PLAYER_SIZE//2 - 5, HEIGHT//2 - PLAYER_SIZE//2 - 5,
                          PLAYER_SIZE+10, PLAYER_SIZE+10), 2)
        pygame.draw.rect(self.screen, WHITE, 
                         (3*WIDTH//4 - PLAYER_SIZE//2 - 5, HEIGHT//2 - PLAYER_SIZE//2 - 5,
                          PLAYER_SIZE+10, PLAYER_SIZE+10), 2)

    def render(self):
        # 1. إذا كنا في وضع اختيار الفريق، ارسم شاشة الاختيار فقط
        if self.game_state == STATE_TEAM_SELECT:
            self._render_team_select()
            pygame.display.flip()
            self.clock.tick(FPS)
            return

        # 2. الرسم العادي للمباراة
        self.screen.fill(GREEN)
        pygame.draw.rect(self.screen, WHITE, self.field_rect, 2)
        pygame.draw.line(self.screen, WHITE, (WIDTH//2, 50), (WIDTH//2, HEIGHT-50), 2)
        pygame.draw.circle(self.screen, WHITE, (WIDTH//2, HEIGHT//2), 70, 2)
        pygame.draw.rect(self.screen, WHITE, (0, HEIGHT//2-GOAL_WIDTH//2, 20, GOAL_WIDTH), 2)
        pygame.draw.rect(self.screen, WHITE, (WIDTH-20, HEIGHT//2-GOAL_WIDTH//2, 20, GOAL_WIDTH), 2)

        # 3. رسم اللاعبين الثلاثة والكرة
        self.p1.draw(self.screen)
        self.p2.draw(self.screen)
        self.p3.draw(self.screen)    # ← اللاعب الثالث
        self.ball.draw(self.screen)

        # 4. عرض النتيجة
        score_text = f"{self.score[0]} - {self.score[1]}"
        self.screen.blit(self.font.render(score_text, True, WHITE), (WIDTH//2 - 40, 20))

        # 5. عرض فرقة اللاعب الأصفر
        if self.p3_team is not None:
            team_name = "Red Team" if self.p3_team == 1 else "Blue Team"
            info_text = f"Yellow Player: {team_name}"
            self.screen.blit(self.small_font.render(info_text, True, YELLOW), (10, HEIGHT - 30))

        # 6. العد التنازلي وبداية اللعب
        if not self.game_active and self.countdown > 0:
            count_surface = self.large_font.render(str(self.countdown), True, YELLOW)
            self.screen.blit(count_surface, (WIDTH//2 - 20, HEIGHT//2 - 50))

        # 7. إعلان الفائز وإعادة التشغيل
        if self.winner:
            win_text = self.large_font.render(f"{self.winner} Wins!", True, YELLOW)
            self.screen.blit(win_text, (WIDTH//2 - 200, HEIGHT//2 - 80))
            restart_text = self.font.render("Press R to restart", True, WHITE)
            self.screen.blit(restart_text, (WIDTH//2 - 100, HEIGHT//2 + 20))

        # 8. معلومات Q-learning (إذا مفعّل)
        if self.use_q_learning:
            q_text     = f"Q-Learning: {'ON' if self.use_q_learning else 'OFF'}"
            mode_text  = f"Training: {'ON' if self.training_mode else 'OFF'}"
            auto_text  = f"Auto-Train: {'ON' if self.auto_train else 'OFF'}"
            episode_text = f"Episode: {self.q_agent.episode_count}"
            explore_text = f"Exploration: {self.q_agent.exploration_rate:.3f}"

            # إحصائيات
            if self.q_agent.rewards_history:
                avg_reward = sum(self.q_agent.rewards_history[-10:]) / min(10, len(self.q_agent.rewards_history))
                reward_text = f"Avg Reward: {avg_reward:.1f}"
            else:
                reward_text = "Avg Reward: N/A"

            # عرض النصوص
            y = 10
            for txt in (q_text, mode_text, auto_text, episode_text, explore_text, reward_text):
                self.screen.blit(self.small_font.render(txt, True, WHITE), (10, y))
                y += 20

            controls = ["Q: Toggle Q-learning", "T: Toggle training", 
                        "Y: Toggle auto-train", "E: Save Q-table", "L: Load Q-table"]
            y = 10
            for ctl in controls:
                self.screen.blit(self.small_font.render(ctl, True, WHITE), (WIDTH - 150, y))
                y += 20

        pygame.display.flip()
        self.clock.tick(FPS)

    def quit(self):
        if self.use_q_learning:
            self.q_agent.save()
        pygame.quit()