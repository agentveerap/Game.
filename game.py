import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import simpson


# ------------------ Optimizer Classes ------------------

class GoldCollectionOptimizer:
    def __init__(self, r, w_per_gold, time_limit, weight_limit):
        self.r = r
        self.w_per_gold = w_per_gold
        self.time_limit = time_limit
        self.weight_limit = weight_limit

    def lagrangian(self, x, v, lam_t, lam_w):
        reward = np.sqrt(x)
        time_constraint = (self.r / max(v, 0.1)) - self.time_limit
        weight_constraint = (x * self.w_per_gold) - self.weight_limit
        return reward - lam_t * time_constraint - lam_w * weight_constraint

    def optimize(self, x_vals, v_vals, lam_t=1.0, lam_w=1.0):
        best_score, best_x, best_v = -1e9, None, None
        for x in x_vals:
            for v in v_vals:
                score = self.lagrangian(x, v, lam_t, lam_w)
                if score > best_score:
                    best_score, best_x, best_v = score, x, v
        return best_x, best_v, best_score


class ShopBalancer:
    def __init__(self, prices, budget, time_limit):
        self.prices = prices
        self.budget = budget
        self.time_limit = time_limit

    def lagrangian(self, quantities, lam_b, lam_t):
        total_value = np.dot(self.prices, quantities)
        budget_constraint = np.dot(self.prices, quantities) - self.budget
        time_constraint = np.sum(quantities) - self.time_limit
        return total_value - lam_b * budget_constraint - lam_t * time_constraint

    def evaluate(self, q, lam_b=1.0, lam_t=1.0):
        return self.lagrangian(np.array(q), lam_b, lam_t)


class CraftingOptimizer:
    def __init__(self, resource_limits, recipe_matrix, values):
        self.resource_limits = resource_limits
        self.recipe_matrix = recipe_matrix
        self.values = values

    def lagrangian(self, q, lam):
        total_value = np.dot(self.values, q)
        resource_use = np.dot(self.recipe_matrix, q)
        penalty = np.dot(lam, (resource_use - self.resource_limits))
        return total_value - penalty

    def evaluate(self, q, lam):
        return self.lagrangian(np.array(q), np.array(lam))


class LevelDesigner:
    def __init__(self, difficulty_curve, reward_curve, x_range):
        self.difficulty_curve = difficulty_curve
        self.reward_curve = reward_curve
        self.x_range = x_range

    def evaluate_balance(self):
        difficulty_area = simpson(self.difficulty_curve(self.x_range), self.x_range)
        reward_area = simpson(self.reward_curve(self.x_range), self.x_range)
        return reward_area / difficulty_area if difficulty_area > 0 else float("inf")

    def plot_curves(self):
        fig, ax = plt.subplots()
        ax.plot(self.x_range, self.difficulty_curve(self.x_range), label="Difficulty")
        ax.plot(self.x_range, self.reward_curve(self.x_range), label="Reward")
        ax.legend()
        st.pyplot(fig)


# ------------------ Streamlit UI ------------------

st.title("🎮 Game Optimizer Playground")

mode = st.sidebar.selectbox("Choose a Game Mode", 
    ["Gold Collection", "Shop Balancing", "Crafting", "Level Designer"]
)

if mode == "Gold Collection":
    st.header("⛏ Gold Collection Optimizer")
    r = st.slider("Distance (r)", 5, 50, 10)
    w_per_gold = st.slider("Weight per gold", 1, 5, 2)
    time_limit = st.slider("Time limit", 1, 20, 5)
    weight_limit = st.slider("Weight limit", 5, 50, 15)

    gco = GoldCollectionOptimizer(r, w_per_gold, time_limit, weight_limit)
    best_x, best_v, best_score = gco.optimize(
        x_vals=range(1, 21),
        v_vals=np.linspace(1, 10, 10)
    )
    st.write(f"Best gold: {best_x}, Best velocity: {best_v:.2f}, Score: {best_score:.2f}")

elif mode == "Shop Balancing":
    st.header("🛒 Shop Balancer")
    prices = np.array([3, 5])
    budget = st.slider("Budget", 10, 50, 20)
    time_limit = st.slider("Max items you can buy", 1, 20, 5)

    sb = ShopBalancer(prices, budget, time_limit)
    q1 = st.number_input("Quantity of Item 1", 0, 20, 2)
    q2 = st.number_input("Quantity of Item 2", 0, 20, 2)

    score = sb.evaluate([q1, q2])
    st.write(f"Your choice → Item1={q1}, Item2={q2}, Score={score:.2f}")

elif mode == "Crafting":
    st.header("⚒ Crafting Optimizer")
    co = CraftingOptimizer(
        resource_limits=np.array([10, 12]),
        recipe_matrix=np.array([[2, 1], [1, 3]]),
        values=np.array([5, 8])
    )
    q1 = st.number_input("Quantity of Recipe 1", 0, 10, 2)
    q2 = st.number_input("Quantity of Recipe 2", 0, 10, 2)

    lam1 = st.slider("Penalty λ1", 0, 5, 1)
    lam2 = st.slider("Penalty λ2", 0, 5, 1)

    score = co.evaluate([q1, q2], [lam1, lam2])
    st.write(f"Your crafting → Recipe1={q1}, Recipe2={q2}, Score={score:.2f}")

elif mode == "Level Designer":
    st.header("🎲 Level Balance Designer")
    ld = LevelDesigner(
        difficulty_curve=lambda x: x**2,
        reward_curve=lambda x: np.sqrt(x),
        x_range=np.linspace(0.1, 5, 100)
    )
    balance = ld.evaluate_balance()
    st.write(f"Balance Ratio (Reward/Difficulty) = {balance:.2f}")
    ld.plot_curves()

            
