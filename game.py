import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import simpson

# ------------------ Optimizer Classes ------------------

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


class FarmingOptimizer:
    def __init__(self, land_limit, water_limit, crop_requirements, profits):
        self.land_limit = land_limit
        self.water_limit = water_limit
        self.crop_requirements = crop_requirements  # matrix [ [land, water], ... ]
        self.profits = profits

    def lagrangian(self, q, lam):
        total_profit = np.dot(self.profits, q)
        land_used = np.dot(self.crop_requirements[:, 0], q)
        water_used = np.dot(self.crop_requirements[:, 1], q)
        penalty = lam[0] * (land_used - self.land_limit) + lam[1] * (water_used - self.water_limit)
        return total_profit - penalty

    def evaluate(self, q, lam):
        return self.lagrangian(np.array(q), np.array(lam))


class TradingOptimizer:
    def __init__(self, buy_prices, sell_prices, budget):
        self.buy_prices = buy_prices
        self.sell_prices = sell_prices
        self.budget = budget

    def lagrangian(self, q, lam):
        cost = np.dot(self.buy_prices, q)
        revenue = np.dot(self.sell_prices, q)
        budget_constraint = cost - self.budget
        return revenue - lam * budget_constraint

    def evaluate(self, q, lam=1.0):
        return self.lagrangian(np.array(q), lam)


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
        ax.plot(self.x_range, self.difficulty_curve(self.x_range), label="Difficulty", color="red")
        ax.plot(self.x_range, self.reward_curve(self.x_range), label="Reward", color="green")
        ax.legend()
        st.pyplot(fig)


# ------------------ Streamlit UI ------------------

st.title("🎮 Game Optimizer Playground")

mode = st.sidebar.selectbox("Choose a Game Mode", 
    ["Shop Balancing", "Farming Optimizer", "Resource Trading", "Level Designer"]
)

if mode == "Shop Balancing":
    st.header("🛒 Shop Balancer")
    prices = np.array([3, 5])
    budget = st.slider("Budget", 10, 50, 20)
    time_limit = st.slider("Max items you can buy", 1, 20, 5)

    sb = ShopBalancer(prices, budget, time_limit)
    q1 = st.number_input("Quantity of Item 1", 0, 20, 2)
    q2 = st.number_input("Quantity of Item 2", 0, 20, 2)

    score = sb.evaluate([q1, q2])
    st.write(f"Your choice → Item1={q1}, Item2={q2}, Score={score:.2f}")

elif mode == "Farming Optimizer":
    st.header("🌾 Farming Business Optimizer")
    fo = FarmingOptimizer(
        land_limit=20,
        water_limit=30,
        crop_requirements=np.array([[2, 3], [4, 5]]),  # Crop1: 2 land,3 water | Crop2: 4 land,5 water
        profits=np.array([10, 18])
    )
    c1 = st.number_input("Acres of Crop 1", 0, 10, 2)
    c2 = st.number_input("Acres of Crop 2", 0, 10, 2)

    lam1 = st.slider("Penalty λ_land", 0, 5, 1)
    lam2 = st.slider("Penalty λ_water", 0, 5, 1)

    score = fo.evaluate([c1, c2], [lam1, lam2])
    st.write(f"Your farming → Crop1={c1}, Crop2={c2}, Profit Score={score:.2f}")

elif mode == "Resource Trading":
    st.header("💱 Resource Trading Optimizer")
    to = TradingOptimizer(
        buy_prices=np.array([4, 6]),
        sell_prices=np.array([7, 10]),
        budget=40
    )
    t1 = st.number_input("Quantity of Resource 1 to Buy", 0, 10, 2)
    t2 = st.number_input("Quantity of Resource 2 to Buy", 0, 10, 2)

    lam = st.slider("Penalty λ_budget", 0, 5, 1)

    score = to.evaluate([t1, t2], lam)
    st.write(f"Your trading → Res1={t1}, Res2={t2}, Profit Score={score:.2f}")

elif mode == "Level Designer":
    st.header("🎲 Level Balance Designer")
    ld = LevelDesigner(
        difficulty_curve=lambda x: x**1.5,  # easier difficulty curve
        reward_curve=lambda x: np.sqrt(x),
        x_range=np.linspace(0.1, 5, 100)
    )
    balance = ld.evaluate_balance()
    st.write(f"Balance Ratio (Reward/Difficulty) = {balance:.2f}")
    ld.plot_curves()
