#####################################################################
#                                                                   #
#                   Information for Users                           #
#                                                                   #
#        - You must have an OpenAI Account and have an API key      #
#        - You have to store this key in your windows environment   #
#          or in your Google Colab as "OPENAI_API_KEY"              #
#        - Without this key you won't be able to run this code      #
#        - You have to install all necessary packages               #
#                                                                   #
#####################################################################





#########################  Importing Packages ######################### 

import os
import openai
import requests
from pathlib import Path
import markdown2
from tabulate import tabulate
import re

######################### Reading in Your OpenAI Key ######################### 

openai.api_key = os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI()

### If you use Google Colab, please use this code to read in your key:

# from google.colab import userdata
# key = userdata.get('OLSAI_API')
# client = OpenAI(api_key = key)


######################### Uploading the Dataset cacao.csv to the Client ######################### 

url = "https://raw.githubusercontent.com/reyar/Statsomat/master/cacao.csv"
response = requests.get(url) 
filename = 'dataset.csv'

with open(filename, 'wb') as f:
    f.write(response.content)

file = client.files.create(file=Path(filename),purpose='assistants')
file_id = file.id

#########################  Retrieving the OLSAI Assistent and Creating a Thread ######################### 

assistant_id = "asst_hy4QCYQSm82ugjY98k3mxgYq"
olsai = client.beta.assistants.retrieve(assistant_id = assistant_id)
thread = client.beta.threads.create()
thread_id = thread.id

#########################  Prompts, Messages and Runs #########################  

prompts = [
    "Explain what descriptive statistics are. Do not refer to the dataset that i gave you.",
    "Display the descriptive statistics of this dataset as a Markdown table with no other sentences.",
    "Explain the summary statistics.",
    "I want to know more about histograms: 1. What are histograms? 2. What are the components of a histogram? 3. How do i interpret a histogram? 4. Provide histograms for every variable of the dataset. Use 'sns.histplot(kde=True,color='gray')' to display the histograms in a grid format.",
    "I want to know more about boxplots: 1. what are boxplots? 2. what are the components of a boxplots? 3. how do i interpret a boxplots? 4. Provide boxplots for every variable of the dataset. Use 'sns.boxplot(color='gray')' to display the boxplots in a grid format",
    "I want to know more about ecdf plots: 1. what are ecdf plots? 2. what are the components of a ecdf plots? 3. how do i interpret a ecdf plots? 4. Provide ecdf plots for every variable of the dataset. Use 'sns.ecdfplot(color = 'black')' to display the ecdf plots in a grid format",
    "I want to know more about qq plots: 1. what are qq plots? 2. what are the components of a qq plots? 3. how do i interpret a qq plots? 4. Provide qq plots for every variable of the dataset. Use 'stats.probplot(dist='norm')' to display the qq plots in a grid format",
    "What is multiple linear regression? Use x_{ij} for independent variables. Explain the ranges of i and j. Explain the assumptions of a (classical) linear regression model in detail and simple, including mathematical equations. Do not provide additional considerations or methods for checking the assumptions. Summarize the assumptions in mathematical form.",
    "Build an OLS regression model using stem_diameter as the dependent variable and all remaining variables as independent variables. Do not display the regression model summary or parameters.",
    "Explain what Regression Diagnostics are without listing specific diagnostics or methods.",
    "Explain outliers to me. Then, explain studentized residuals to me and provide mathematical equations. I want to understand the basic idea of studentized residuals. Afterwards, tell me which observation is an outlier by using the plot of studentized residuals vs index and 3 as threshold. Where does this threshold come from? Explain and interpret the plot. Should regression diagnostics be repeated after removing potential outliers? Additional infos on the plot: - annotate he outlier in the plot using only the index - use a red dashed line to show the threshold",
    "Explain high-leverage points to me. Use 2p/n as threshold. Then, explain cooks distance to me and provide mathematical equations. I want to understand the basic idea of cooks distance.  Afterwards, tell me which observation is an high-leverage point by using the plot of leverage vs index using 2p/n as threshold and the plot of cooks distance vs index using 4/n as threshold. Do not make subplots. Explain and interpret the plots. Do not answer the question whether regression diagnostics be repeated after removing high-leverage points? Additional infos on the plot: - use black stemlines - use red stemlines for observations above the threshold and annotate them only using the index - use a red dashed line to show the threshold",
    "Explain non-linearity to me. Then, explain the rainbow test to me and provide mathematical equations. I want to understand the basic idea of the rainbow test. Afterwards, tell me if non linearity if violated by using the rainbow test and the plot of residuals vs fitted values. Explain and interpret the plot. Additional infos on the plot: - use sns.residplot(lowess=True) and plt.scatter(predictions, residuals, color='black')",
    "Explain heteroscedasticity to me in detail. Then, explain the breusch pangan test to me and provide mathematical equations. I want to understand the basic idea of the breusch pangan test. Afterwards, tell me if heteroscedasticity is violated by using the breusch pangan test and the scale location plot. Explain in detail and interpret the plot in detail. Additional infos on the plot: - use sns.regplot(lowess=True) and plt.scatter(predictions, sqrt_standardized_residuals, color='black') - no dashed line at 0",
    "Explain correlation of error terms to me. Then, explain the durbin watson test to me and provide mathematical equations. I want to understand the basic idea of the durbin watson test. Afterwards, tell me if correlation of error terms is violated by using the durbin watson test with 1.5 - 2.5 as recommended range and the plot of residuals over time. Explain and interpret the plot.  Additional infos on the plot: - use plt.plot(studentized_residuals)",
    "Explain normality of residuals to me. Then, explain the shapiro wilk test to me and provide mathematical equations. I want to understand the basic idea of the shapiro wilk test. Afterwards, tell me if non normality of residuals is violated by using the shapiro wilk test and the qq plot of standardized residuals. Explain and interpret the plot. Additional infos on the plot: - use stats.probplot(standardized_residuals, dist='norm', plot=plt)",
    "Explain collinearity of predictors to me. Then, explain the variance inflation factor to me and provide mathematical equations. I want to understand the basic idea of the variance inflation factor. Afterwards, tell me if collinearity of predictors is violated by using the variance inflation factor with 10 as threshold and plot of the correlation matrix. Explain and interpret the plot. Explain how to read the plot. Additional infos on the plot: - use sns.heatmap(annot=True, cmap='coolwarm')",
    "Summarize the results of outliers, high-leverage points, non-linearity, heteroscedasticity, correlation of error terms, normality of residuals and collinearity of predictors"
]

for prompt in prompts:
    status = "incompleted"  
    while status != "completed": 
        message = client.beta.threads.messages.create( thread_id = thread_id, role = "user", content=prompt, attachments=[{"file_id": file_id, "tools": [{"type": "code_interpreter"}]}])
        run = client.beta.threads.runs.create_and_poll(thread_id = thread.id, assistant_id = olsai.id, instructions = "Adress the user as a student, who has no knowledge about data science.")
        status = run.status # NOCH EINBAUEN: Wenn Status incomplete, failed, ..., was dann?
        
response = client.beta.threads.messages.list(thread_id=thread.id, limit= 100)

#########################  Formatting ChatGPTs Output ######################### 

output_chatgpt = []

for i in range(len(response.data)):
    if response.data[i].assistant_id:
        output_chatgpt.append(response.data[i].content)
        
# output_chatgpt Muss Länge 29 haben, sonst ist ein Fehler aufgetreten ---> noch einbauen!

full_output_raw = output_chatgpt[::-1]
full_output = {}

for i in range(len(full_output_raw)):
    if len(full_output_raw[i]) == 1 and i != 1:
        text = full_output_raw[i][0].text.value.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
        full_output[f'text{i+1}'] = markdown2.markdown(text, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
    if len(full_output_raw[i]) == 1 and i == 1:
        if i == 1:
            # Formatting Table
            html_table_raw = full_output_raw[i][0].text.value.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
            html_table_raw = html_table_raw.replace("```markdown", "").replace("```", "").strip()
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
            full_output[f'text{i+1}'] = html_table1
    if len(full_output_raw[i]) == 2:
        img_id = full_output_raw[i][0].image_file.file_id
        image_data = client.files.content(img_id)
        image_data_bytes = image_data.read()
        with open(f"image{i+1}.png", "wb") as file:
            file.write(image_data_bytes)
        full_output[f'img{i+1}'] = f"image{i+1}.png"
        text = full_output_raw[i][1].text.value.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
        full_output[f'text{i+1}'] = markdown2.markdown(text, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])
    if len(full_output_raw[i]) == 3:
        # first image
        img_id = full_output_raw[i][0].image_file.file_id
        image_data = client.files.content(img_id)
        image_data_bytes = image_data.read()
        with open(f"image{i+1}.png", "wb") as file:
            file.write(image_data_bytes)
        # secon image
        img_id = full_output_raw[i][1].image_file.file_id
        image_data = client.files.content(img_id)
        image_data_bytes = image_data.read()
        with open(f"image{i+1}_2.png", "wb") as file:
            file.write(image_data_bytes)
        full_output[f'img{i+1}'] = f"image{i+1}.png"
        full_output[f'img{i+1}_2'] = f"image{i+1}_2.png"
        text = full_output_raw[i][2].text.value.replace(r'\[', r'\\(').replace(r'\]', r'\\)').replace(r'\(', r'\\(').replace(r'\)', r'\\)')
        full_output[f'text{i+1}'] = markdown2.markdown(text, extras=["code-friendly", "fenced-code-blocks", "cuddled-lists"])

#########################  Creating HTML-Document ######################### 
     
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
    <p>{full_output['text1']}</p>
    <br>
    <h2>The Table of Descriptive Statistics</h2>

    <div>
        {full_output['text2']}
    </div>
    <p>{full_output['text3']}</p>
    <br>
    <h2>Visual Representation of the Data</h2>
    <h3>Histograms</h3>
    <p>{full_output['text4']}</p>
    <img src="{full_output['img5']}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    <p>{full_output['text5']}</p>
    <br>
    <h3>Boxplots</h3>
    <p>{full_output['text6']}</p>
    <img src="{full_output['img7']}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    <p>{full_output['text7']}</p>
    <br>
    <h3>ECDF Plots</h3>
    <p>{full_output['text8']}</p>
    <img src="{full_output['text9']}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    <p>{full_output['text9']}</p>
    <br>
    <h3>QQ-Plots</h3>
    <p>{full_output['text10']}</p>
    <img src="{full_output['img11']}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    <p>{full_output['text11']}</p>
    <br>
    <h1>Part 2: The Multiple Linear Regression Model</h1>
    <br>
    <h2>What is Multiple Linear Regression?</h2>
    {full_output['text12']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>What are Regression Diagnostics?</h2>
    <p>{full_output['text13']}</p>
    <br>
    <p>{full_output['text14']}</p>
    <br>
    <h1>Part 3: Regression Diagnostics</h1>
    <br>
    <h2>Outliers</h2>
    {full_output['text15']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{full_output['img16']}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {full_output['text16']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>High-Leverage Points</h2>
    {full_output['text17']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{full_output['img18']}" alt="Descriptive Statistics" style="width:100%;;max-width: 1000px;">
    <br>
    <img src="{full_output['img18_2']}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {full_output['text18']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Non-Linearity</h2>
    {full_output['text19']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{full_output['img20']}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {full_output['text20']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Heteroscedasticity</h2>
    {full_output['text21']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{full_output['img22']}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {full_output['text22']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Correlation of Error Terms</h2>
    {full_output['text23']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{full_output['img24']}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {full_output['text24']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Normality of Residuals</h2>
    {full_output['text25']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{full_output['img26']}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {full_output['text26']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Collinearity of Predictors</h2>
    {full_output['text27']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <img src="{full_output['img28']}" alt="Descriptive Statistics" style="width:100%;max-width: 1000px;">
    {full_output['text28']}
    <script>
        MathJax.typeset();  // Renders the LaTeX after the page loads
    </script>
    <br>
    <h2>Summary of Results</h2>
    {full_output['text29']}
</body>
</html>
"""

# Write the content to an HTML file
with open("ai_output.html", "w") as file:
    file.write(html_template)