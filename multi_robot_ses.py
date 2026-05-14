# ================================
# Multi-Robot Path Planning (SES)
# VS Code Compatible Version
# ================================

import matplotlib
matplotlib.use("TkAgg")   # IMPORTANT for VS Code

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import trange
import time


class MultiRobotPathPlanner:

    def __init__(self,
                 num_robots=8,
                 workspace_size=40,
                 max_velocity=3.0,
                 dt=0.1,
                 influence_distance=8.0,
                 population_size=30,
                 generations=100):

        # Environment
        self.num_robots = num_robots
        self.workspace_size = workspace_size
        self.max_velocity = max_velocity
        self.dt = dt
        self.influence_distance = influence_distance
        self.population_size = population_size
        self.generations = generations

        # ANN structure
        self.input_size = 3
        self.hidden_size = 10
        self.output_size = 1

        # Swarm parameters
        self.ksi_att = 0.5
        self.ksi_rep = 0.3

        np.random.seed(42)

        # Initialize robots
        self.robot_positions = np.random.uniform(
            5, workspace_size - 5, (num_robots, 2))

        self.robot_goals = np.random.uniform(
            5, workspace_size - 5, (num_robots, 2))

        self.initialize_ann_structure()
        self.generate_training_data()
        self.train_ann_with_ses()

    # =====================================
    # Utility Functions
    # =====================================

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def angle_diff(self, a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    # =====================================
    # ANN Setup
    # =====================================

    def initialize_ann_structure(self):
        self.w1_size = self.input_size * self.hidden_size
        self.b1_size = self.hidden_size
        self.w2_size = self.hidden_size * self.output_size
        self.b2_size = self.output_size
        self.vector_size = (self.w1_size +
                            self.b1_size +
                            self.w2_size +
                            self.b2_size)

    def decode_ann(self, particle):
        idx = 0
        W1 = particle[idx:idx+self.w1_size].reshape(
            self.input_size, self.hidden_size)
        idx += self.w1_size
        b1 = particle[idx:idx+self.b1_size]
        idx += self.b1_size
        W2 = particle[idx:idx+self.w2_size].reshape(
            self.hidden_size, self.output_size)
        idx += self.w2_size
        b2 = particle[idx:idx+self.b2_size]
        return W1, b1, W2, b2

    def ann_forward(self, X, W1, b1, W2, b2):
        hidden = self.sigmoid(np.dot(X, W1) + b1)
        output = self.tanh(np.dot(hidden, W2) + b2) * np.pi
        return output

    # =====================================
    # Generate Training Data
    # =====================================

    def generate_training_data(self):
        samples = 2000
        self.X_train = np.zeros((samples, 3))
        self.y_train = np.zeros(samples)

        for i in range(samples):

            goal_angle = np.random.uniform(-np.pi, np.pi)
            crit_angle = np.random.uniform(-np.pi, np.pi)
            crit_dist = np.random.uniform(
                1.0, self.influence_distance + 2)

            F_att = self.ksi_att * crit_dist

            if crit_dist < self.influence_distance:
                F_rep = self.ksi_rep * (
                    (1/crit_dist -
                     1/self.influence_distance) /
                    (crit_dist**2))
            else:
                F_rep = 0

            deviation = np.arctan2(
                F_rep*np.sin(crit_angle) +
                F_att*np.sin(goal_angle),
                F_rep*np.cos(crit_angle) +
                F_att*np.cos(goal_angle)
            )

            self.X_train[i] = [goal_angle,
                               crit_angle,
                               crit_dist]

            self.y_train[i] = deviation

    # =====================================
    # SES Training
    # =====================================

    def diversity_penalty(self, population):
        centroid = np.mean(population, axis=0)
        diversity = np.mean(
            np.linalg.norm(population - centroid, axis=1))
        return 1 / (diversity + 1e-6)

    def train_ann_with_ses(self):

        population = np.random.uniform(
            -1, 1,
            (self.population_size, self.vector_size))

        print("Training using SES...")

        for gen in trange(self.generations):

            fitness_scores = []

            for i in range(self.population_size):
                W1, b1, W2, b2 = self.decode_ann(
                    population[i])

                y_pred = self.ann_forward(
                    self.X_train, W1, b1, W2, b2).flatten()

                mse = np.mean(
                    self.angle_diff(
                        self.y_train,
                        y_pred)**2)

                fitness_scores.append(mse)

            fitness_scores = np.array(fitness_scores)

            penalty = self.diversity_penalty(population)
            total_fitness = fitness_scores + 0.1 * penalty

            elite_idx = np.argmin(total_fitness)
            elite = population[elite_idx]

            new_population = [elite.copy()]

            while len(new_population) < self.population_size:
                mutation = np.random.normal(
                    0, 0.1, size=self.vector_size)
                child = elite + mutation
                new_population.append(
                    np.clip(child, -5, 5))

            population = np.array(new_population)

        # Select final best
        final_scores = []

        for i in range(self.population_size):
            W1, b1, W2, b2 = self.decode_ann(population[i])
            y_pred = self.ann_forward(
                self.X_train, W1, b1, W2, b2).flatten()

            mse = np.mean(
                self.angle_diff(
                    self.y_train, y_pred)**2)

            final_scores.append(mse)

        best_idx = np.argmin(final_scores)
        self.best_weights = self.decode_ann(
            population[best_idx])

        print("Training completed.")
        self.animate_robot_movement()

    # =====================================
    # Robot Simulation
    # =====================================

    def animate_robot_movement(self):

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.workspace_size)
        ax.set_ylim(0, self.workspace_size)
        ax.set_title("Multi-Robot Path Planning (SES)")

        robots_plot = [
            ax.plot([], [], 'o')[0]
            for _ in range(self.num_robots)
        ]

        reached = [False] * self.num_robots

        def update(frame):

            W1, b1, W2, b2 = self.best_weights

            for i in range(self.num_robots):

                if reached[i]:
                    continue

                goal_vec = (self.robot_goals[i] -
                            self.robot_positions[i])

                dist_goal = np.linalg.norm(goal_vec)

                if dist_goal < 0.5:
                    reached[i] = True
                    continue

                goal_dir = goal_vec / (dist_goal + 1e-6)
                angle_to_goal = np.arctan2(
                    goal_dir[1], goal_dir[0])

                moved = False

                for j in range(self.num_robots):

                    if i != j:

                        dist = np.linalg.norm(
                            self.robot_positions[i] -
                            self.robot_positions[j])

                        if dist < self.influence_distance:

                            crit_angle = np.arctan2(
                                self.robot_positions[j][1] -
                                self.robot_positions[i][1],
                                self.robot_positions[j][0] -
                                self.robot_positions[i][0])

                            inp = np.array(
                                [[angle_to_goal,
                                  crit_angle,
                                  dist]])

                            dev = self.ann_forward(
                                inp, W1, b1, W2, b2)[0, 0]

                            rot = np.array([
                                [np.cos(dev),
                                 -np.sin(dev)],
                                [np.sin(dev),
                                 np.cos(dev)]
                            ])

                            new_dir = np.dot(rot,
                                             goal_dir)

                            self.robot_positions[i] += (
                                new_dir *
                                self.max_velocity *
                                self.dt)

                            moved = True
                            break

                if not moved:
                    self.robot_positions[i] += (
                        goal_dir *
                        self.max_velocity *
                        self.dt)

                robots_plot[i].set_data(
                    [self.robot_positions[i][0]],
                    [self.robot_positions[i][1]])

            return robots_plot

        self.ani = FuncAnimation(
            fig, update,
            frames=400,
            interval=50)

        plt.grid()
        plt.show(block=True)


# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    print("Starting Multi-Robot Path Planning...")
    planner = MultiRobotPathPlanner()