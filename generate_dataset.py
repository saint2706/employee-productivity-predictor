"""
Generates a synthetic Employee Productivity dataset for a linear regression demo.
Clean (no missing values, no extreme outliers) but with realistic correlation
structure, department/job-level effects, and noise so the regression isn't trivial.

"Synthetic" just means: nobody is a real employee. Every number in the final
file is created by code, not collected from a company. We build it this way
on purpose so that the columns relate to each other the way real HR data
would (e.g. more experienced people tend to finish tasks faster), which
makes it a good, realistic-feeling dataset to practice linear regression on.
"""

# --- Imports: bring in code libraries other people wrote, so we don't have
# to write everything from scratch. --------------------------------------

# numpy ("np" for short) gives us fast tools for working with lists of
# numbers (called "arrays") and for generating random numbers.
import numpy as np

# pandas ("pd" for short) gives us the DataFrame, which is basically a
# spreadsheet/table object living in Python. We'll build our final table
# with it and save it as a CSV file (a plain-text spreadsheet format).
import pandas as pd

# --- Randomness setup ------------------------------------------------------

# A "random number generator" (rng) is an object that produces random
# numbers whenever we ask it to (e.g. "give me a random age").
# We pass it a fixed "seed" (42) so that every time this script runs, it
# produces the EXACT SAME "random" numbers. This is called making the
# results "reproducible" — useful so you and I see the same dataset.
# If you want a different dataset each run, change 42 to any other number,
# or remove the seed entirely.
rng = np.random.default_rng(42)

# N is the number of rows (employees) we want in the final dataset.
# 5000 is large enough to give a linear regression model plenty of examples
# to learn from, which produces a more stable, realistic demo.
N = 5000

# --- Department setup -------------------------------------------------------

# This is a plain Python list of the possible department names. Each
# simulated employee will be assigned to exactly one of these.
departments = [
    "Sales",
    "Engineering",
    "HR",
    "Marketing",
    "Finance",
    "Operations",
    "Customer_Support",
]

# dept_weights controls how LIKELY each department is to be picked, in the
# same order as the "departments" list above (so 0.18 is the chance of
# "Sales", 0.22 is the chance of "Engineering", and so on). These numbers
# must add up to 1.0 (i.e. 100%). We're saying Engineering is the biggest
# department (22% of employees) and HR/Customer_Support are the smallest
# (10% each) — just to make the company feel realistic, not perfectly even.
dept_weights = [0.18, 0.22, 0.10, 0.14, 0.12, 0.14, 0.10]

# dept_productivity_offset is a "dictionary" (a lookup table of key -> value
# pairs). It says: "if you work in this department, add this many points to
# your productivity score." This simulates the real-world idea that some
# departments/teams tend to score a bit higher or lower than others on
# whatever "productivity" measure the company uses, for reasons unrelated
# to any individual's skill (e.g. different tools, different workflows).
dept_productivity_offset = {
    "Sales": -1.0,
    "Engineering": 3.5,
    "HR": -0.5,
    "Marketing": 0.5,
    "Finance": 1.5,
    "Operations": -1.5,
    "Customer_Support": -2.0,
}

# --- Job level setup ---------------------------------------------------------

# The five possible seniority levels, from least to most senior.
job_levels = ["Junior", "Mid", "Senior", "Lead", "Manager"]

# job_level_ord converts each level's NAME into a NUMBER representing its
# rank ("ord" is short for "ordinal", meaning "position in an order").
# We need this because some of the math below (like "more senior people
# tend to have more experience") is much easier to write with numbers
# (1, 2, 3, 4, 5) than with text ("Junior", "Mid", ...).
job_level_ord = {"Junior": 1, "Mid": 2, "Senior": 3, "Lead": 4, "Manager": 5}

# job_level_weights: same idea as dept_weights above, but for how common
# each job level is. Most companies have lots of Junior/Mid employees and
# progressively fewer people the higher up the ladder you go — that's why
# these numbers shrink from 0.30 down to 0.08. They also add up to 1.0.
job_level_weights = [0.30, 0.30, 0.20, 0.12, 0.08]

# job_level_productivity_offset: like dept_productivity_offset, but for job
# level. More senior employees get a bigger bonus added to their eventual
# productivity score, on the (realistic) assumption that seniority tends to
# come with more efficient, higher-output work.
job_level_productivity_offset = {
    "Junior": 0,
    "Mid": 3,
    "Senior": 6,
    "Lead": 9,
    "Manager": 11,
}

# --- Years of experience setup ----------------------------------------------

# exp_params tells us, for each job level, what a "typical" number of years
# of experience looks like. Each entry is a tuple (a small fixed-size group
# of values) of 4 numbers: (mu, sigma, lo, hi) where:
#   mu    = the average ("mean") years of experience for that level
#   sigma = how spread out the values are around that average (standard
#           deviation — a bigger number means more variety between people)
#   lo    = the lowest years of experience we'll allow for that level
#   hi    = the highest years of experience we'll allow for that level
# For example, a "Junior" employee averages 1.5 years of experience, with
# some spread of 1.2 years, and we won't let a Junior's experience go below
# 0 or above 4 years (a Junior with 20 years of experience wouldn't be
# realistic).
exp_params = {
    "Junior": (1.5, 1.2, 0, 4),
    "Mid": (4.5, 2.0, 2, 8),
    "Senior": (9.0, 3.0, 5, 16),
    "Lead": (14.0, 4.0, 8, 23),
    "Manager": (19.0, 5.0, 10, 35),
}

# --- Column 1: Employee_ID ---------------------------------------------------

# A "list comprehension" is a compact way to build a list by repeating an
# expression for every value in a range. Here, for every whole number i
# from 1 to N (5000), we build a string like "EMP00001", "EMP00002", ...
# f"..." is an f-string: anything inside {curly braces} gets replaced with
# its actual value. str(i).zfill(5) turns the number i into text and pads
# it with leading zeros until it's 5 digits long (so 1 becomes "00001").
Employee_ID = [f"EMP{str(i).zfill(5)}" for i in range(1, N + 1)]

# --- Column 2: Department ----------------------------------------------------

# rng.choice randomly picks N values from the "departments" list. The `p=`
# argument tells it to use our dept_weights as the probability of picking
# each option, instead of picking every department equally often.
# The result is an array of N department names, one per simulated employee.
Department = rng.choice(departments, size=N, p=dept_weights)

# --- Column 3: Job_Level ------------------------------------------------------

# Same idea as Department, but picking from job_levels using job_level_weights.
Job_Level = rng.choice(job_levels, size=N, p=job_level_weights)

# --- Column 4: Years_of_Experience -------------------------------------------

# We start with an array of N zeros (one slot per employee) and then fill
# each slot in with a real value below. np.zeros(N) just reserves the space.
Years_of_Experience = np.zeros(N)

# We loop over each job level and its matching (mu, sigma, lo, hi) numbers
# from exp_params (defined above). For each level:
for lvl, (mu, sigma, lo, hi) in exp_params.items():
    # mask is a list of True/False values, one per employee: True means
    # "this employee's Job_Level equals the level we're currently handling".
    # This lets us select just the employees at this level.
    mask = Job_Level == lvl

    # mask.sum() counts how many True values are in mask (True counts as 1,
    # False as 0), i.e. how many employees are at this level.
    # rng.normal(mu, sigma, count) draws that many random numbers from a
    # "normal distribution" (the classic bell-curve shape) centered on mu
    # with spread sigma. Most values land close to mu, fewer land far away.
    vals = rng.normal(mu, sigma, mask.sum())

    # np.clip(vals, lo, hi) forces every value to stay between lo and hi —
    # if a random draw came out lower than lo or higher than hi, it gets
    # pulled back to that boundary. This keeps experience realistic for
    # each level (e.g. no Junior with 20 years of experience).
    # We then store these values into Years_of_Experience, but ONLY in the
    # slots where mask is True (i.e. only for employees at this level).
    Years_of_Experience[mask] = np.clip(vals, lo, hi)

# Round every value to 1 decimal place (e.g. 4.83217 becomes 4.8) so the
# numbers look like something a human would actually write down.
Years_of_Experience = np.round(Years_of_Experience, 1)

# level_ord converts every employee's Job_Level text into its numeric rank
# (using the job_level_ord dictionary from above), so we can use it in math
# formulas later (you can't multiply a formula by the word "Senior", but
# you can multiply it by the number 3).
level_ord = np.array([job_level_ord[j] for j in Job_Level])

# --- Column 5: Training_Hours -------------------------------------------------

# Training hours: newer/junior employees tend to get more formal training
# (onboarding), while senior employees get less. We model this with a
# straight-line formula: start at 55 hours and subtract 4 hours for every
# step up in seniority (level_ord). So Juniors (level 1) average around
# 55 - 4*1 = 51 hours, while Managers (level 5) average 55 - 4*5 = 35 hours.
# rng.normal(..., 15, N) then adds random person-to-person variation with a
# spread of 15 hours around that average, for all N employees at once.
Training_Hours = rng.normal(55 - 4 * level_ord, 15, N)

# Clip so nobody has negative training hours or an unrealistically huge
# number, then round to 1 decimal place.
Training_Hours = np.clip(Training_Hours, 4, 120).round(1)

# --- Column 6: Monthly_Working_Hours ------------------------------------------

# Most employees work around 175 hours a month (roughly 40 hours/week),
# with some natural variation (sigma of 14 hours) between people/months.
Monthly_Working_Hours = rng.normal(175, 14, N)

# Clip to a believable range (130 to 215 hours/month) and round.
Monthly_Working_Hours = np.clip(Monthly_Working_Hours, 130, 215).round(1)

# --- Column 7 (built early, used later): Engagement_Score --------------------
# We build this before Absence_Days because absences below depend on it.

# dept_engagement_base: like dept_productivity_offset earlier, but this
# time it's the AVERAGE engagement score (0-100 scale, like a survey
# result) typical for each department, rather than an offset added to
# something else.
dept_engagement_base = {
    "Sales": 62,
    "Engineering": 70,
    "HR": 66,
    "Marketing": 64,
    "Finance": 63,
    "Operations": 60,
    "Customer_Support": 58,
}

# For every employee, look up their department's base engagement score
# (using a list comprehension, same trick as Employee_ID above), turn that
# list into a numpy array, and then add random per-person variation
# (rng.normal(0, 12, N) draws N random "noise" values averaging 0 with a
# spread of 12 points) to represent that people in the same department
# still don't all feel equally engaged.
Engagement_Score = np.array([dept_engagement_base[d] for d in Department]) + rng.normal(
    0, 12, N
)

# Engagement scores don't make sense below 5 or above 100, so we clip them
# into that range, then round to 1 decimal place.
Engagement_Score = np.clip(Engagement_Score, 5, 100).round(1)

# --- Column 8: Absence_Days ---------------------------------------------------

# We assume less engaged employees tend to take more days off. absence_lambda
# is the AVERAGE number of absence days we expect for each employee, computed
# from a formula: start at 6 days and subtract a small amount (0.045) for
# every point of Engagement_Score. So a highly engaged employee (score 100)
# averages 6 - 0.045*100 = 1.5 absence days, while a low-engagement employee
# (score 20) averages 6 - 0.045*20 = 5.1 days.
# np.clip(..., 0.3, None) makes sure this average never drops below 0.3
# (the "None" means "no upper limit needed here").
absence_lambda = np.clip(6.0 - 0.045 * Engagement_Score, 0.3, None)

# rng.poisson(lambda, N) draws random WHOLE NUMBERS (you can't take "half a
# day off") from a "Poisson distribution" — a standard way statisticians
# model random COUNTS of events (like "how many days off did someone take
# this month"), where absence_lambda is the average count for each person.
Absence_Days = rng.poisson(absence_lambda, N)

# Cap absences at a believable maximum of 18 days in a month, and don't
# allow negative absences (Poisson never produces negative numbers anyway,
# but we clip for safety/clarity).
Absence_Days = np.clip(Absence_Days, 0, 18)

# --- Column 9: Projects_Completed ---------------------------------------------

# More senior and more experienced employees tend to complete more projects.
# proj_lambda is the average number of projects we expect per employee:
# start at 2, add 1.1 projects for every step up in seniority (level_ord),
# and add a small bonus (0.05 per year) for years of experience.
proj_lambda = np.clip(2.0 + 1.1 * level_ord + 0.05 * Years_of_Experience, 0.5, None)

# Again use a Poisson distribution to turn that average into random whole
# numbers of completed projects per employee.
Projects_Completed = rng.poisson(proj_lambda, N)

# Cap at a believable maximum of 25 projects in a month.
Projects_Completed = np.clip(Projects_Completed, 0, 25)

# --- Column 10: Average_Task_Completion_Time ----------------------------------

# This is how many hours, on average, it takes this employee to finish one
# task. We assume more experienced/senior employees are FASTER (a LOWER
# number of hours), so the formula SUBTRACTS a bit of time for every year
# of experience (0.14 hours) and every level of seniority (0.35 hours),
# starting from a baseline of 10.5 hours. rng.normal(..., 1.4, N) then adds
# natural person-to-person variation with a spread of 1.4 hours.
Average_Task_Completion_Time = rng.normal(
    10.5 - 0.14 * Years_of_Experience - 0.35 * level_ord, 1.4, N
)

# Clip to a believable range (nobody finishes a task in under 1.5 hours on
# average, or takes more than 18 hours on average) and round to 2 decimals.
Average_Task_Completion_Time = np.clip(Average_Task_Completion_Time, 1.5, 18).round(2)

# --- Column 11: Monthly_Productivity_Score (this is our TARGET column) -------
# "Target column" means: this is the value a linear regression model will
# try to predict using all the other columns as clues. Everything above was
# building the "clues"; now we combine them into the answer.

# Turn each employee's Department and Job_Level into their matching
# productivity offsets (defined near the top of the script) as numpy arrays,
# so we can add them into a single big math formula below.
dept_offset = np.array([dept_productivity_offset[d] for d in Department])
level_offset = np.array([job_level_productivity_offset[j] for j in Job_Level])

# "noise" represents random, unpredictable variation in productivity that
# isn't explained by any of our columns — exactly like real life, where two
# people with identical experience/training/etc. still don't perform
# IDENTICALLY. Without this, the data would be too perfectly predictable to
# feel like a realistic regression exercise. It averages 0 with a spread of
# 4.5 points, so about as often it nudges the score up as it nudges it down.
noise = rng.normal(0, 4.5, N)

# This is the heart of the whole script: a straight-line ("linear") formula
# that combines every input column into a single productivity score. Each
# column is multiplied by a fixed number (its "weight" or "coefficient")
# that controls how strongly it pushes the score up or down, then everything
# is added together. This is precisely the kind of relationship a linear
# regression model is designed to learn back out from the data.
Monthly_Productivity_Score = (
    32  # baseline score everyone starts with
    + 0.55 * Years_of_Experience  # more experience -> higher score
    + 0.10 * Training_Hours  # more training -> slightly higher score
    + 0.04
    * (
        Monthly_Working_Hours - 175
    )  # working more than the 175hr norm -> slightly higher score
    + 1.6 * Projects_Completed  # more finished projects -> higher score
    - 1.1
    * Average_Task_Completion_Time  # SLOWER task completion -> lower score (that's why it's subtracted)
    - 0.55 * Absence_Days  # more absences -> lower score (also subtracted)
    + 0.30 * Engagement_Score  # more engaged -> higher score
    + dept_offset  # each department's fixed bonus/penalty
    + level_offset  # each job level's fixed bonus
    + noise  # random unexplained variation
)

# Productivity scores don't make sense below 0 or above 100 on our chosen
# scale, so clip into that range, then round to 2 decimal places.
Monthly_Productivity_Score = np.clip(Monthly_Productivity_Score, 0, 100).round(2)

# --- Assemble everything into one table (DataFrame) ---------------------------

# pd.DataFrame(...) builds a table where each key below becomes a column
# name, and each value (one of our arrays/lists built above) becomes that
# column's data, matched up row-by-row (row 0 of every column belongs to
# the same simulated employee, row 1 to the next employee, and so on).
df = pd.DataFrame(
    {
        "Employee_ID": Employee_ID,
        "Department": Department,
        "Job_Level": Job_Level,
        "Years_of_Experience": Years_of_Experience,
        "Training_Hours": Training_Hours,
        "Monthly_Working_Hours": Monthly_Working_Hours,
        "Projects_Completed": Projects_Completed,
        "Average_Task_Completion_Time": Average_Task_Completion_Time,
        "Absence_Days": Absence_Days,
        "Engagement_Score": Engagement_Score,
        "Monthly_Productivity_Score": Monthly_Productivity_Score,
    }
)

# --- Save the table to a file and print a quick summary -----------------------

# The file name/path where we'll save our table as a CSV (Comma-Separated
# Values) file — a plain text format that Excel, Google Sheets, and pandas
# can all open.
out_path = "employee_productivity_dataset.csv"

# Write the DataFrame out to that CSV file. index=False means "don't add an
# extra column of row numbers (0, 1, 2, ...) to the file" — we already have
# Employee_ID to identify each row, so we don't need pandas' automatic one.
df.to_csv(out_path, index=False)

# Print a short confirmation message so you know the script worked, using
# an f-string to insert the row count and file name into the sentence.
print(f"Wrote {len(df)} rows to {out_path}")

# df.isna() checks every cell in the table for missing values, giving back
# True/False for each; .sum().sum() adds up all the Trues (first per
# column, then across all columns) to get one total count. We expect this
# to print 0, confirming the dataset has no missing values.
print(df.isna().sum().sum(), "missing values")

# df.describe(...) computes summary statistics (count, average, min, max,
# etc.) for every column; include='all' makes it also summarize the text
# columns (Department, Job_Level, Employee_ID), not just the numeric ones.
# .T "transposes" the result (swaps rows and columns) purely so it's easier
# to read when printed to the terminal.
print(df.describe(include="all").T)
