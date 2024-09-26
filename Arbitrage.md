# What is Arbitrage?
It is the simultaneous buying and selling of the same or similar asset in different markets. Usually done to profit from differences in listed prices on different exchanges.

## What is the goal?
We want to parse in some stocks and their ticker symbols and have the program determine if there are any opportunities for arbitrage on different exchanges. This will be a simple example and as a result will have a dataframe that contains stocks and their corresponding exchanges.

## How to start

First, we need to break down the arbitrage process and focus on comparing stock pricess across different markets or time intervals. What we are trying to do is to exploit the price differences.

### Outline

1. **Data source:** We could use APIs to gather real-time stock price data from multiple exchanges. But in this simple version we will be using AAPL,AAPL.L(on the London Stock Exchange), MSFT, MSFT.L etc.
2. **Price Discrepancies:** Identify price changes across different markets.
3. **Arbitrage conditions:** Figure out an appropriate threshold to identify significant price differences that would offer a profitable arbitrage opportunity.
4. **Decision Making:** Calculate potential profit
5. **Monitoring:** And finally our program should continuously monitor price changes to allert potential opportunities

### Key Libraries Needed

| Libraries | What they Do |
|-----|-----|
| pandas |  used for working with data sets. |
| numpy | for numerical calculations |
| yfinance | for fetching stock data from yahoo finance |
| matplotlib | to vizualize data |
