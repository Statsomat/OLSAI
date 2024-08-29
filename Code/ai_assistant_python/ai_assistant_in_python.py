# ====== Getting / Creating Assistent from OpenAI ======


import os #import os to get the windows environment
import openai #import openai to use chatgpt

openai.api_key = os.environ.get("OPENAI_API_KEY") #get api key from windows environment
client = openai.OpenAI() #setting up the openai client

assistant_id = "asst_hy4QCYQSm82ugjY98k3mxgYq" #assistant id of olsai
thread_id = None #thread id

import requests #for sending http requests
from pathlib import Path #for filesystem paths

url = "https://raw.githubusercontent.com/reyar/Statsomat/master/cacao.csv" #url of our dataset
response = requests.get(url) 
filename = 'dataset.csv'

with open(filename, 'wb') as f: #download the dataset from the internet
    f.write(response.content)

file = client.files.create( #upload the dataset to the assistent
    file=Path(filename),
    purpose='assistants'
)
file_id = file.id #id of the file

if not assistant_id: #creating the new ai assistent if the assistent is non existing
    olsai = client.beta.assistants.create(
        name = "OLSAI",
        instructions = "You are the best teacher who can explain things the most clearly. You know everything about exploratory data analysis, multiple linear regression, and regression diagnostics. Adress the user as a student, who has no knowledge about data science.",
        tools = [{"type":"code_interpreter"}],
        tool_resources={"code_interpreter": {"file_ids": [file.id]}},
        model = "gpt-4o-2024-08-06",
        temperatur = 0.1
    )
    print(olsai.id)

    thread = client.beta.threads.create()
    print(thread.id)
else:
    olsai = client.beta.assistants.retrieve(assistant_id = assistant_id)
    print(f"The assistant with the ID '{assistant_id}' was loaded.")
    
if not thread_id:
    thread = client.beta.threads.create()
    thread_id = thread.id
    print(thread.id)
else:
    thread = client.beta.threads.retrieve(thread_id = thread_id)
    print(f"The thread with the ID '{thread_id}' was loaded.")


# ====== Part 1: Descriptive Statistics ======


# === What are Descriptive Statistics? ===

message = client.beta.threads.messages.create(
    thread_id = thread_id,
    role = "user",
    content="Explain what descriptive statistics are.",
)

run = client.beta.threads.runs.create_and_poll(
  thread_id = thread.id,
  assistant_id = olsai.id,
  instructions = "Adress the user as a student, who has no knowledge about data science."
)

if run.status == 'completed':
  response = client.beta.threads.messages.list(thread_id=thread.id)
  response_descriptive_statistics = response.data[0].content[0].text.value
else:
  print(run.status)
  
# === Table of Descriptive Statistics ===

message = client.beta.threads.messages.create(
    thread_id = thread_id,
    role = "user",
    content="Display the descriptive statistics of this dataset as a Markdown table with no other sentences.",
    attachments=[
        {
          "file_id": file_id,
          "tools": [{"type": "code_interpreter"}]
        }]
)

run = client.beta.threads.runs.create_and_poll(
    thread_id = thread.id,
    assistant_id = olsai.id,
    instructions = "Adress the user as a student, who has no knowledge about data science."
)

if run.status == 'completed':
  response = client.beta.threads.messages.list(thread_id=thread.id)
  response_descriptive_stat_table = response.data[0].content[0].text.value
else:
  print(run.status)


# === Summary Statistics ===

message = client.beta.threads.messages.create(
    thread_id = thread_id,
    role = "user",
    content="Explain the summary statistics",
    attachments=[
        {
          "file_id": file_id,
          "tools": [{"type": "code_interpreter"}]
        }]
)

run = client.beta.threads.runs.create_and_poll(
  thread_id = thread.id,
  assistant_id = olsai.id,
  instructions = "Adress the user as a student, who has no knowledge about data science."
)


if run.status == 'completed':
  response = client.beta.threads.messages.list(thread_id=thread.id)
  response_summary_stat = response.data[0].content[0].text.value
else:
  print(run.status)

# === Histograms ===

message = client.beta.threads.messages.create(
    thread_id = thread_id,
    role = "user",
    content="I want to know more about histograms: 1. What are histograms? 2. What are the components of a histogram? 3. How do i interpret a histogram? 4. Provide histograms for every variable of the dataset. Use 'sns.histplot(kde=True,color='gray')' to display the histograms in a grid format.",
    attachments=[
        {
          "file_id": file_id,
          "tools": [{"type": "code_interpreter"}]
        }]
)

run = client.beta.threads.runs.create_and_poll(
    thread_id = thread.id,
    assistant_id = olsai.id,
    instructions = "Adress the user as a student, who has no knowledge about data science."
)

if run.status == 'completed':
  response = client.beta.threads.messages.list(thread_id=thread.id)
  response_before_hist = response.data[1].content[0].text.value
  response_hist = response.data[0].content[1].text.value
  image_file_id = response.data[0].content[0].image_file.file_id
  image_data = client.files.content(image_file_id)
  image_data_bytes = image_data.read()

  with open("desc_stat_hist.png", "wb") as file:
    file.write(image_data_bytes)
else:
  print(run.status)

# === Boxplots ===

message = client.beta.threads.messages.create(
    thread_id = thread_id,
    role = "user",
    content= "I want to know more about boxplots: 1. what are boxplots? 2. what are the components of a boxplots? 3. how do i interpret a boxplots? 4. Provide boxplots for every variable of the dataset. Use 'sns.boxplot(color='gray')' to display the boxplots in a grid format",
    attachments=[
        {
          "file_id": file_id,
          "tools": [{"type": "code_interpreter"}]
        }]
)

run = client.beta.threads.runs.create_and_poll(
    thread_id = thread.id,
    assistant_id = olsai.id,
    instructions = "Adress the user as a student, who has no knowledge about data science."
)

if run.status == 'completed':
  response = client.beta.threads.messages.list(thread_id=thread.id)
  response_before_box = response.data[1].content[0].text.value
  response_box = response.data[0].content[1].text.value
  image_file_id = response.data[0].content[0].image_file.file_id
  image_data = client.files.content(image_file_id)
  image_data_bytes = image_data.read()

  with open("desc_stat_box.png", "wb") as file:
    file.write(image_data_bytes)
else:
  print(run.status)

# === ECDF Plots ===

message = client.beta.threads.messages.create(
    thread_id = thread_id,
    role = "user",
    content= "I want to know more about ecdf plots: 1. what are ecdf plots? 2. what are the components of a ecdf plots? 3. how do i interpret a ecdf plots? 4. Provide ecdf plots for every variable of the dataset. Use 'sns.ecdfplot(color = 'black')' to display the ecdf plots in a grid format",
    attachments=[
        {
          "file_id": file_id,
          "tools": [{"type": "code_interpreter"}]
        }]
)

run = client.beta.threads.runs.create_and_poll(
    thread_id = thread.id,
    assistant_id = olsai.id,
    instructions = "Adress the user as a student, who has no knowledge about data science."
)

if run.status == 'completed':
  response = client.beta.threads.messages.list(thread_id=thread.id)
  response_before_ecdf = response.data[1].content[0].text.value
  response_ecdf = response.data[0].content[1].text.value
  image_file_id = response.data[0].content[0].image_file.file_id
  image_data = client.files.content(image_file_id)
  image_data_bytes = image_data.read()

  with open("desc_stat_ecdf.png", "wb") as file:
    file.write(image_data_bytes)
else:
  print(run.status)

# === QQ Plots ===

message = client.beta.threads.messages.create(
    thread_id = thread_id,
    role = "user",
    content= "I want to know more about qq plots: 1. what are qq plots? 2. what are the components of a qq plots? 3. how do i interpret a qq plots? 4. Provide qq plots for every variable of the dataset. Use 'stats.probplot(dist='norm')' to display the qq plots in a grid format",
    attachments=[
        {
          "file_id": file_id,
          "tools": [{"type": "code_interpreter"}]
        }]
)

run = client.beta.threads.runs.create_and_poll(
    thread_id = thread.id,
    assistant_id = olsai.id,
    instructions = "Adress the user as a student, who has no knowledge about data science."
)

if run.status == 'completed':
  response = client.beta.threads.messages.list(thread_id=thread.id)
  response_before_qq = response.data[1].content[0].text.value
  response_qq = response.data[0].content[1].text.value
  image_file_id = response.data[0].content[0].image_file.file_id
  image_data = client.files.content(image_file_id)
  image_data_bytes = image_data.read()

  with open("desc_stat_qq.png", "wb") as file:
    file.write(image_data_bytes)
else:
  print(run.status)


# ====== Part 2: Multiple Linear Regressino ======


# === What is Multiple Linear Regression? ===

message = client.beta.threads.messages.create(
    thread_id = thread_id,
    role = "user",
    content="What is multiple linear regression? Use x_{ij} for independent variables. Explain the ranges of i and j. Explain the assumptions of a (classical) linear regression model in detail and simple, including mathematical equations. Do not provide additional considerations or methods for checking the assumptions. Summarize the assumptions in mathematical form.",
)

run = client.beta.threads.runs.create_and_poll(
  thread_id = thread.id,
  assistant_id = olsai.id,
  instructions = "Adress the user as a student, who has no knowledge about data science."
)


if run.status == 'completed':
  response = client.beta.threads.messages.list(thread_id=thread.id)
  response_multiple_linear_model = response.data[0].content[0].text.value
else:
  print(run.status)

# === Building the Model ===

message = client.beta.threads.messages.create(
    thread_id = thread_id,
    role = "user",
    content="Build an OLS regression model using stem_diameter as the dependent variable and all remaining variables as independent variables. Do not display the regression model summary or parameters.",
)

run = client.beta.threads.runs.create_and_poll(
  thread_id = thread.id,
  assistant_id = olsai.id,
  instructions = "Adress the user as a student, who has no knowledge about data science."
)


if run.status == 'completed':
  response = client.beta.threads.messages.list(thread_id=thread.id)
  response_build_model = response.data[0].content[0].text.value
else:
  print(run.status)

# === What are Regression Diagnostics? ===

message = client.beta.threads.messages.create(
    thread_id = thread_id,
    role = "user",
    content="Explain what Regression Diagnostics are without listing specific diagnostics or methods.",
)

run = client.beta.threads.runs.create_and_poll(
  thread_id = thread.id,
  assistant_id = olsai.id,
  instructions = "Adress the user as a student, who has no knowledge about data science."
)


if run.status == 'completed':
  response = client.beta.threads.messages.list(thread_id=thread.id)
  response_reg_diag_expl = response.data[0].content[0].text.value
else:
  print(run.status)


# ====== Part 3: Regression Diagnostics ======


# === Outliers ===

prompt = "Explain outliers to me. Then, explain studentized residuals to me and provide mathematical equations. I want to understand the basic idea of studentized residuals. Afterwards, tell me which observation is an outlier by using the plot of studentized residuals vs index and 3 as threshold. Where does this threshold come from? Explain and interpret the plot. Should regression diagnostics be repeated after removing potential outliers? Additional infos on the plot: - annotate he outlier in the plot using only the index - use a red dashed line to show the threshold"

for i in range(2):
    message = client.beta.threads.messages.create(
        thread_id = thread_id,
        role = "user",
        content= prompt,
        attachments=[
            {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}]
            }] 
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id = thread.id,
        assistant_id = olsai.id,
        instructions = "Adress the user as a student, who has no knowledge about data science."
    )

    if run.status == 'completed':
        response = client.beta.threads.messages.list(thread_id=thread.id)
        response_before_out = response.data[1].content[0].text.value
        response_out = response.data[0].content[1].text.value
        image_file_id = response.data[0].content[0].image_file.file_id
        image_data = client.files.content(image_file_id)
        image_data_bytes = image_data.read()

        with open("out.png", "wb") as file:
            file.write(image_data_bytes)
    else:
        print(run.status)

# === High-Leverage Points ===

prompt = "Explain high-leverage points to me. Use 2p/n as threshold. Then, explain cooks distance to me and provide mathematical equations. I want to understand the basic idea of cooks distance.  Afterwards, tell me which observation is an high-leverage point by using the plot of leverage vs index using 2p/n as threshold and the plot of cooks distance vs index using 4/n as threshold. Do not make subplots. Explain and interpret the plots. Do not answer the question whether regression diagnostics be repeated after removing high-leverage points? Additional infos on the plot: - use black stemlines - use red stemlines for observations above the threshold and annotate them only using the index - use a red dashed line to show the threshold"

for i in range(2):
    message = client.beta.threads.messages.create(
        thread_id = thread_id,
        role = "user",
        content= prompt,
        attachments=[
            {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}]
            }]
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id = thread.id,
        assistant_id = olsai.id,
        instructions = "Adress the user as a student, who has no knowledge about data science."
    )

    if run.status == 'completed':
        response = client.beta.threads.messages.list(thread_id=thread.id)
        response_before_hlev = response.data[1].content[0].text.value
        response_hlev = response.data[0].content[2].text.value
        image_file_id1 = response.data[0].content[0].image_file.file_id
        image_file_id2 = response.data[0].content[1].image_file.file_id
        image_data1 = client.files.content(image_file_id1)
        image_data_bytes1 = image_data1.read()

        with open("hlev1.png", "wb") as file:
            file.write(image_data_bytes1)
  
        image_data2 = client.files.content(image_file_id2)
        image_data_bytes2 = image_data2.read()

        with open("hlev2.png", "wb") as file:
            file.write(image_data_bytes2)
    
    else:
        print(run.status)

# === Non-Linearity ===

prompt = "Explain non-linearity to me. Then, explain the rainbow test to me and provide mathematical equations. I want to understand the basic idea of the rainbow test. Afterwards, tell me if non linearity if violated by using the rainbow test and the plot of residuals vs fitted values. Explain and interpret the plot. Additional infos on the plot: - use sns.residplot(lowess=True) and plt.scatter(predictions, residuals, color='black')"

for i in range(2):
    message = client.beta.threads.messages.create(
        thread_id = thread_id,
        role = "user",
        content= prompt,
        attachments=[
            {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}]
            }]
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id = thread.id,
        assistant_id = olsai.id,
        instructions = "Adress the user as a student, who has no knowledge about data science."
    )

    if run.status == 'completed':
        response = client.beta.threads.messages.list(thread_id=thread.id)
        response_before_nonlin = response.data[1].content[0].text.value
        response_nonlin = response.data[0].content[1].text.value
        image_file_id = response.data[0].content[0].image_file.file_id
        image_data = client.files.content(image_file_id)
        image_data_bytes = image_data.read()

        with open("nonlin.png", "wb") as file:
            file.write(image_data_bytes)
    else:
        print(run.status)

# === Hetersoscedasticity ===

prompt = "Explain heteroscedasticity to me in detail. Then, explain the breusch pangan test to me and provide mathematical equations. I want to understand the basic idea of the breusch pangan test. Afterwards, tell me if heteroscedasticity is violated by using the breusch pangan test and the scale location plot. Explain in detail and interpret the plot in detail. Additional infos on the plot: - use sns.regplot(lowess=True) and plt.scatter(predictions, sqrt_standardized_residuals, color='black') - no dashed line at 0"

for i in range(2):
    message = client.beta.threads.messages.create(
        thread_id = thread_id,
        role = "user",
        content= prompt,
        attachments=[
            {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}]
            }]
        )

    run = client.beta.threads.runs.create_and_poll(
        thread_id = thread.id,
        assistant_id = olsai.id,
        instructions = "Adress the user as a student, who has no knowledge about data science."
    )

    if run.status == 'completed':
        response = client.beta.threads.messages.list(thread_id=thread.id)
        response_before_hetero = response.data[1].content[0].text.value
        response_hetero = response.data[0].content[1].text.value
        image_file_id = response.data[0].content[0].image_file.file_id
        image_data = client.files.content(image_file_id)
        image_data_bytes = image_data.read()

        with open("hetero.png", "wb") as file:
            file.write(image_data_bytes)
    else:
        print(run.status)

# === Correlation of Error Terms ===

prompt = "Explain correlation of error terms to me. Then, explain the durbin watson test to me and provide mathematical equations. I want to understand the basic idea of the durbin watson test. Afterwards, tell me if correlation of error terms is violated by using the durbin watson test with 1.5 - 2.5 as recommended range and the plot of residuals over time. Explain and interpret the plot.  Additional infos on the plot: - use plt.plot(studentized_residuals)"

for i in range(2):
    message = client.beta.threads.messages.create(
        thread_id = thread_id,
        role = "user",
        content= prompt,
        attachments=[
        {
          "file_id": file_id,
          "tools": [{"type": "code_interpreter"}]
        }]
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id = thread.id,
        assistant_id = olsai.id,
        instructions = "Adress the user as a student, who has no knowledge about data science."
    )

    if run.status == 'completed':
        response = client.beta.threads.messages.list(thread_id=thread.id)
        response_before_cor = response.data[1].content[0].text.value
        response_cor = response.data[0].content[1].text.value
        image_file_id = response.data[0].content[0].image_file.file_id
        image_data = client.files.content(image_file_id)
        image_data_bytes = image_data.read()

        with open("cor.png", "wb") as file:
            file.write(image_data_bytes)
    else:
        print(run.status)

# === Normality of Residuals ===

prompt = "Explain normality of residuals to me. Then, explain the shapiro wilk test to me and provide mathematical equations. I want to understand the basic idea of the shapiro wilk test. Afterwards, tell me if non normality of residuals is violated by using the shapiro wilk test and the qq plot of standardized residuals. Explain and interpret the plot. Additional infos on the plot: - use stats.probplot(standardized_residuals, dist='norm', plot=plt)"

for i in range(2):
    message = client.beta.threads.messages.create(
        thread_id = thread_id,
        role = "user",
        content= prompt,
        attachments=[
            {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}]
            }]
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id = thread.id,
        assistant_id = olsai.id,
        instructions = "Adress the user as a student, who has no knowledge about data science."
    )

    if run.status == 'completed':
        response = client.beta.threads.messages.list(thread_id=thread.id)
        response_before_normality = response.data[1].content[0].text.value
        response_normality = response.data[0].content[1].text.value
        image_file_id = response.data[0].content[0].image_file.file_id
        image_data = client.files.content(image_file_id)
        image_data_bytes = image_data.read()

        with open("normality.png", "wb") as file:
            file.write(image_data_bytes)
    else:
        print(run.status)

# === Collinearity of Predictors ===

prompt = "Explain collinearity of predictors to me. Then, explain the variance inflation factor to me and provide mathematical equations. I want to understand the basic idea of the variance inflation factor. Afterwards, tell me if collinearity of predictors is violated by using the variance inflation factor with 10 as threshold and plot of the correlation matrix. Explain and interpret the plot. Explain how to read the plot. Additional infos on the plot: - use sns.heatmap(annot=True, cmap='coolwarm')"

for i in range(2):
    message = client.beta.threads.messages.create(
        thread_id = thread_id,
        role = "user",
        content= prompt,
        attachments=[
            {
            "file_id": file_id,
            "tools": [{"type": "code_interpreter"}]
            }]
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id = thread.id,
        assistant_id = olsai.id,
        instructions = "Adress the user as a student, who has no knowledge about data science."
    )

    if run.status == 'completed':
        response = client.beta.threads.messages.list(thread_id=thread.id)
        response_before_coll = response.data[1].content[0].text.value
        response_coll = response.data[0].content[1].text.value
        image_file_id = response.data[0].content[0].image_file.file_id
        image_data = client.files.content(image_file_id)
        image_data_bytes = image_data.read()

        with open("coll.png", "wb") as file:
            file.write(image_data_bytes)
    else:
        print(run.status)

# === Summary ===

for i in range(2):
    message = client.beta.threads.messages.create(
        thread_id = thread_id,
        role = "user",
        content="Summarize the results of outliers, high-leverage points, non-linearity, heteroscedasticity, correlation of error terms, normality of residuals and collinearity of predictors",
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id = thread.id,
        assistant_id = olsai.id,
        instructions = "Adress the user as a student, who has no knowledge about data science."
    )

    if run.status == 'completed':
        response = client.beta.threads.messages.list(thread_id=thread.id)
        response_summary = response.data[0].content[0].text.value
    else:
        print(run.status)


# ====== Formatting Answers ======


import markdown2
import pdfkit
from tabulate import tabulate
import re

# === Formatting Text Output to Markdown ===

text1 = response_descriptive_statistics.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text2 = response_summary_stat.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text3 = response_before_hist.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text4 = response_hist.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text5 = response_before_box.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text6 = response_box.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text7 = response_before_ecdf.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text8 = response_ecdf.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text9 = response_before_qq.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text10 = response_qq.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text11 = response_multiple_linear_model.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text12 = response_reg_diag_expl.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text13 = response_build_model.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text14 = response_before_out.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text15 = response_out.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text16 = response_before_hlev.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text17 = response_hlev.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text18 = response_before_nonlin.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')

text20 = response_nonlin.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text21 = response_before_hetero.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text22 = response_hetero.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text23 = response_before_cor.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text24 = response_cor.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text25 = response_before_normality.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text26 = response_normality.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text27 = response_before_coll.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text28 = response_coll.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
text29 = response_summary.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')

# === Convert Markdown to HTML ===

html_text1 = markdown2.markdown(text1, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text2 = markdown2.markdown(text2, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text3 = markdown2.markdown(text3, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text4 = markdown2.markdown(text4, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text5 = markdown2.markdown(text5, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text6 = markdown2.markdown(text6, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text7 = markdown2.markdown(text7, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text8 = markdown2.markdown(text8, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text9 = markdown2.markdown(text9, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text10 = markdown2.markdown(text10, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text11 = markdown2.markdown(text11, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text12 = markdown2.markdown(text12, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text13 = markdown2.markdown(text13, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text14 = markdown2.markdown(text14, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text15 = markdown2.markdown(text15, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text16 = markdown2.markdown(text16, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text17 = markdown2.markdown(text17, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text18 = markdown2.markdown(text18, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])

html_text20 = markdown2.markdown(text20, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text21 = markdown2.markdown(text21, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text22 = markdown2.markdown(text22, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text23 = markdown2.markdown(text23, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text24 = markdown2.markdown(text24, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text25 = markdown2.markdown(text25, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text26 = markdown2.markdown(text26, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text27 = markdown2.markdown(text27, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text28 = markdown2.markdown(text28, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
html_text29 = markdown2.markdown(text29, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])

# === Formatting Table ===

html_table_raw = response_descriptive_stat_table.replace("```markdown", "").replace("```", "").strip()
lines = html_table_raw.split('\n')
html_table1 = "<table>\n"
# Process each line
for line in lines:
    # Skip any empty lines
    if line.strip():
        # Replace the Markdown table delimiters with HTML table tags
        if '---' in line:
            continue  # Skip the line with "---"
        elif '|' in line:
            line = line.replace('|', '</td><td>').strip('<td>').strip('</td>')
            html_table1 += f"  <tr><td>{line}</td></tr>\n"
# Close the table tag
html_table1 += "</table>"


# ====== Creating HTML File ======


html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Learning Regression Diagnostics</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        table, th, td {{
            border: 1px solid black;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
        }}
        .math {{
            text-align: center;
            margin: 1em 0;
        }}
    </style>
</head>
<body>
    <h1>Part 1: Exploratory Data Analysis</h1>
    <br>
    <h2>What are Descriptive Statistics?</h2>
    <p>{html_text1}</p>
    <br>
    <h2>The Table of Descriptive Statistics</h2>

    <div>
        {html_table1}
    </div>
    <p>{html_text2}</p>
    <br>
    <h2>Visual Representation of the Data</h2>
    <h3>Histograms</h3>
    <p>{html_text3}</p>
    <img src="{"desc_stat_hist.png"}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    <p>{html_text4}</p>
    <br>
    <h3>Boxplots</h3>
    <p>{html_text5}</p>
    <img src="{"desc_stat_box.png"}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    <p>{html_text6}</p>
    <br>
    <h3>ECDF Plots</h3>
    <p>{html_text7}</p>
    <img src="{"desc_stat_ecdf.png"}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    <p>{html_text8}</p>
    <br>
    <h3>QQ-Plots</h3>
    <p>{html_text9}</p>
    <img src="{"desc_stat_qq.png"}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    <p>{html_text10}</p>
    <br>
    <h1>Part 2: The Multiple Linear Regression Model</h1>
    <br>
    <h2>What is Multiple Linear Regression?</h2>
    {html_text11}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>What are Regression Diagnostics?</h2>
    <p>{html_text12}</p>
    <br>
    <p>{html_text13}</p>
    <br>
    <h1>Part 3: Regression Diagnostics</h1>
    <br>
    <h2>Outliers</h2>
    {html_text14}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{"out.png"}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {html_text15}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>High-Leverage Points</h2>
    {html_text16}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{"hlev1.png"}" alt="Descriptive Statistics" style="width:100%;;max-width: 1000px;">
    <br>
    <img src="{"hlev2.png"}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {html_text17}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Non-Linearity</h2>
    {html_text18}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{"nonlin.png"}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {html_text20}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Heteroscedasticity</h2>
    {html_text21}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{"hetero.png"}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {html_text22}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Correlation of Error Terms</h2>
    {html_text23}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{"cor.png"}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {html_text24}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Normality of Residuals</h2>
    {html_text25}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{"normality.png"}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {html_text26}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Collinearity of Predictors</h2>
    {html_text27}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{"coll.png"}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {html_text28}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Summary of Results</h2>
    {html_text29}
</body>
</html>
"""

# Write the content to an HTML file
with open("ai_output.html", "w") as file:
    file.write(html_template)
