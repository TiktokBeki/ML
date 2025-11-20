# ML
machine learning coding with bereket
Core Data Processing

pandas → Used to create, read, modify, and analyze tabular data (rows & columns).

numpy → Helps with mathematical operations, random number generation, arrays.

Visualization

matplotlib.pyplot → Basic graphs & plots.

seaborn → Pretty statistical visualizations (heatmaps, correlations, etc.).

Scikit-Learn (Traditional ML)

train_test_split → Splits data into training & testing sets.

StandardScaler → Normalizes numbers so models train better.

ColumnTransformer → Applies transformations to selected columns.

Pipeline → Chains preprocessing + model steps into one clean object.

RandomForestRegressor → Ensemble ML model for regression.

GradientBoostingRegressor → Another powerful boosting-based regressor.

Metrics → MAE, RMSE, R² to evaluate accuracy.

Joblib

joblib → Saves the trained model to a file.

PyTorch

torch → Main PyTorch library.

torch.nn → Layers (Linear, ReLU, etc.) for building neural networks.

torch.optim → Optimizers like Adam.

TensorDataset → Converts X & y into PyTorch training-ready datasets.

DataLoader → Loads data in batches for efficient training.






python -m pip install pandas matplotlib seaborn joblib torch





1. What This Program Is

This program is basically a smart calculator that tries to guess how much electricity a house will use in one day.

You tell it things like:

how hot it is

how much AC you used

how many people live in the house

how many hours the TV or computer was used

the day of the week

the month

And the program gives you a number, like:

18.52 kWh


which means:
“This is about how much electricity your house would use that day.”

2. What the Program Does When You Run It

When you run it, it does a lot of work on its own:

It creates 50,000 fake days of household activity
(but realistic, like actual human behavior)

It studies the patterns in that fake data
(for example: hotter days → more AC → more electricity)

It teaches three different “smart systems” to understand those patterns
(you don’t need to know what they are, they just learn)

It checks how good those smart systems are
(it prints numbers telling you how accurate they are)

It saves the best one

It makes a test prediction just to show you it works

It asks you for your own household values and gives you your prediction

You do NOT need to give any input unless the program asks you at the end.

3. How It Thinks (Simple Version)

Think of the program like this:

You show it 50,000 examples of days:

temperature

humidity

number of people

how long AC was used

how long heater was used

how long TV/computer was used

laundry cycles

and the electricity used (the “answer”)

The program looks at all of that and learns:

what causes electricity to go up

what causes it to go down

how much each action affects the total

what combinations matter

and what doesn’t matter

It keeps adjusting itself until it becomes good at predicting the final number.

After enough training, it becomes good at guessing electricity usage for days it has never seen before.

4. How the Program Calculates Electricity Usage (Simple)

Here's the simple idea of what it does:

It takes your input (temperature, AC time, etc.)

It compares it to what it learned from 50,000 examples

It looks for patterns it learned, like:

“Hot days usually mean more AC.”

“More people means more electricity.”

“Laundry adds a lot.”

“Humidity can slightly change usage.”

It mixes all those patterns together

It gives you the final number: electricity used in kWh

That’s it.

There is no magic — just learning from examples.

5. How to Use the Program (Super Simple Instructions)
Step 1 — Run the file
python Scores.py

Step 2 — Wait

It needs time to train itself.

This part prints messages like:

“Training…”

“Loss: …”

“Prediction: …”

This is completely normal.

Step 3 — Look for the questions

At the end it will ask you:

Temperature (°C):
Humidity (%):
Number of people home:
Hours AC used:
Hours heater used:
TV hours:
Computer hours:
Laundry cycles:
Day of week (0=Monday):
Month (1–12):


Just type numbers and press ENTER each time.

Step 4 — The program gives you your electricity usage

It will print something like:

Estimated Electricity Usage (kWh): 19.52


You can run it again and enter different values if you want.

6. What the Program’s “Smart Brain” Actually Does (Simple Explanation)

The “smart brain” is the part that learned all the patterns.

Here’s what it does behind the scenes:

It takes your numbers

It pushes them through a bunch of math layers
(you can think of them like filters)

Each layer changes the numbers a bit
based on what it learned from the 50,000 examples

Finally, it spits out one single number
→ how much electricity it thinks your house used

It's like a calculator that taught itself how electricity usage works.

7. Conclusion

This program:

creates a huge amount of realistic household data

teaches itself how electricity usage works

asks you for your household details

predicts your electricity use for the day

gives you a clear number you can understand

You don’t need to understand machine learning to use it.
You just give it simple numbers and it does all the thinking for you.

It is basically a smart electricity guesser built using thousands of examples and a trained digital brain.
