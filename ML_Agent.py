import yfinance as yf

# Get the stock data for AAPL
stock_data = yf.Ticker("AAPL")

# Get the historical prices
hist = stock_data.history(period="max")

# Select the "Close" column as the training data
X = hist["Close"].values

# Reshape the data into a 2D array
X = X.reshape(-1, 1)

# Train the reinforcement learning agent on the data
agent = Agent(state_size=1, action_size=1)
agent.fit(X)

# Use the agent to make a prediction about the next stock price for AAPL
next_state = [X[-1]]  # Use the last price in the sequence as the current state
next_price = agent.act(next_state)
print(f"Predicted next price for AAPL: {next_price}")
