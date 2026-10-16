# Project 1: A marketing agent that learns from its customers

You build an AI marketing agent that writes the first line ("hook") of marketing emails for five customer segments and improves it with **OPRO** (Optimization by PROmpting). The customers are simulated: for each segment, a language model plays a customer with a hidden description of what this customer likes. You only see how many customers visit the shop.

## How to start
1. Open `project1.ipynb` in Google Colab.
2. **Runtime → Change runtime type → T4 GPU.**
3. **File → Save a copy in Drive**, then run the cells from top to bottom.

## Files
| File | What it is |
|---|---|
| `project1.ipynb` | The notebook you work in (10 🎯 TODOs) |
| `marketing.py` | Helper functions: language model, hidden customers, campaign simulation, OPRO loop, plots. Read it if you like; you do not need to change it. |
| `checks.py` | The ✅/❌ checks that run after the TODOs |
| `data/customers.csv` | 700 customers |
| `data/segments.csv` | The five segments and their descriptions |
| `data/customer_feedback.csv` | Quotes from customer surveys and support tickets |
| `data/personas.json` | The hidden customer descriptions, encoded. **Please don't decode them**: finding out what customers want is the point of the project. |

## Language model
The notebook uses `google/gemma-4-E2B-it` for both writing hooks and playing the customers.
