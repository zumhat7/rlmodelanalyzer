# F&O RL Trading Bot

This project automates trading strategies for Futures & Options (F&O/VIOP) markets using Reinforcement Learning (RL) in Python.

## 📋 Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Installation & Running](#installation--running)  
- [Usage](#usage)  
- [Model Training](#model-training)  
- [Evaluation](#evaluation)  
- [Development Guide](#development-guide)  
- [License](#license)  

---

## 🔍 Project Overview

1. **Data Collection**: Fetch real price and technical indicator data from AlphaVantage API.  
2. **Environment**: Simulate F&O contract properties and portfolio management in a Gym environment (`FxOptionEnv`).  
3. **Agent**: Train a Proximal Policy Optimization (PPO) agent using Stable-Baselines3.  
4. **Evaluation**: Compute Sharpe ratio, maximum drawdown, and total PnL to assess performance.

---

## ⚙️ Features

- Support for both **real** and **dummy** price series  
- Built-in technical indicators: RSI, MACD, SMA, etc.  
- Margin, transaction costs, and drawdown penalty for risk management  
- Single and vectorized Gym environments  
- Easily extendable state & action spaces  

---

## 🛠 Technologies Used

| Technology               | Purpose                                      | Recommended Version       |
|--------------------------|----------------------------------------------|---------------------------|
| **Python 3.8+**          | Core programming language                    | 3.8, 3.9, 3.10            |
| **gymnasium / gym**      | RL environment interface                     | gymnasium ≥0.28.x         |
| **stable-baselines3**    | PPO, A2C, DQN and other RL algorithms        | ≥1.8.0                    |
| **pandas**               | Time series & DataFrame manipulation         | ≥1.4.0                    |
| **numpy**                | Numerical computations                       | ≥1.22.0                   |
| **requests**             | HTTP requests (AlphaVantage API)             | ≥2.27.0                   |
| **TA-Lib**               | Technical indicator calculations             | ≥0.4.24                   |
| **matplotlib**           | Plotting performance charts (optional)       | ≥3.5.0                    |
| **Docker** (optional)    | Containerized environment                    | 20.x                      |
| **GitHub Actions** (CI)  | Automated training/evaluation pipelines      | —                         |

