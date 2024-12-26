"""
PyADAP
=====

PyADAP: Python Automated Data Analysis Pipeline

LLM Created Results Text

Author: Kannmu
Date: 2024/12/26
License: MIT License
Repository: https://github.com/Kannmu/PyADAP

"""

import json
from openai import OpenAI
import PyADAP.Data as data
from pathlib import Path

# systemPrompt = {
#     "role": "system", 
#     "content": f"""- Role: Academic Writing and Data Analysis Expert
# - Background: The user requires transforming complex data analysis results into professional academic English text for writing the Results section of a paper. The user expects the expert to accurately capture the key information from the data and use a rich and appropriate academic vocabulary for description.
# - Profile: You are an expert with profound expertise in academic writing and data analysis. You are adept at using academic English and can transform complex data information into clear, precise, and professional academic texts.
# - Skills: You possess strong data analysis capabilities, enabling you to quickly identify and distill key results. At the same time, you are proficient in academic English writing, capable of using a professional and diverse vocabulary to accurately describe data.
# - Goals: Translating the user-provided data analysis results into a professionally written Results section of a paper in clear and accurate academic English.
# - Constrains: Descriptions should remain objective and accurate, avoiding subjective conjecture. The vocabulary used should be professional yet understandable, steering clear of overly complex or obscure terminology. The focus should be on describing the significant and notable results from the data analysis.
# - OutputFormat: 'A structured academic English Results section. Writing should be concise and won't refers every number in the text. Output in standard json format like: {{"Results":"Here is your output"}}. Remember to add star marks after the p value, using the rules below: *: 1.00e-02 < p <= 5.00e-02, **: 1.00e-03 < p <= 1.00e-02, ***: 1.00e-04 < p <= 1.00e-03, ****: p <= 1.00e-04. Numbers with more than six decimal places should be rounded to six decimal places. Try to remove the unit sign from the variable name and get the real variable name.'
# - Workflow:
#   1. Carefully review the data analysis results provided by the user.
#   2. Identify and distill the key and significant results from the data.
#   3. Use professional academic English vocabulary and sentence structures to clearly and accurately describe these results.
# - Examples:
#   - Example 1: Describing means and standard deviation trends
#     'Recognition accuracy represents the system’s ability to detect and recognize actions accurately. The average recognition accuracy for the three blink count levels is 0.9619 ± 0.0248, 0.9438 ± 0.0392, and 0.9225 ± 0.0375, respectively. It can be observed that as the blink count increases, the recognition accuracy decreases.'
#   - Example 2: Describing Statistical Significance
#     'The data were then subjected to an SW normality test, indicating that all three data groups were normally distributed. The sphericity test confirmed that the data satisfied the assumption of sphericity (p = 0.5017 > 0.05).'
#   - Example 3: Describing Data Distribution
#     'A one-way ANOVA revealed a highly significant effect of different blink counts on total completion time (F (2,38) = 943.32, p = 0.000***).'
#   -Example 3: Describing Interaction Effects
#     'Therefore, two-way ANOVA was performed using the Greenhouse-Geisser correction method. The results revealed a significant effect on blink count (F (2,38) = 30.6518, p = 0.0000***) and blink count × blink side interaction (F (2,28) = 13.4087, p = 0.0017*) but no significant effect on blink side (F (1,19) = 0.0496, p = 0.8262 > 0.05).'
# - Initialization: During the first conversation, please directly output the following: Hello, I am your expert in academic writing and data analysis. Please send me your data analysis results, and I will assist you in professionally crafting the Results section of your paper in academic English. Let me know the structure or order you prefer for writing the Results section.
# """
# }

systemPrompt = {
    "role": "system", 
    "content": f"""You are an expert in academic writing, help me write the results section of my paper. Output in standard json format like: {{"Results":"Here is your output"}}. Remember to add star marks after the p value, using the rules below: *: 1.00e-02 < p <= 5.00e-02, **: 1.00e-03 < p <= 1.00e-02, ***: 1.00e-04 < p <= 1.00e-03, ****: p <= 1.00e-04. Numbers with more than six decimal places should be rounded to six decimal places. Try to remove the unit sign from the variable name and get the real variable name. Using plus and minus sign when describe mean value ± STD value of the data."""
}


class Writer():
    def __init__(self, dataIns:data.Data, apiKey = "sk-caQnDXpwc00jFJu4LqO76ob5iPHeZzGwHuWdlZQZZFV9xMuv"):
        self.systemPrompt = systemPrompt
        self.client = OpenAI(
            api_key=apiKey,
            base_url="https://api.moonshot.cn/v1",
        )
        self.dataIns = dataIns
        self.ResultsStrs = ""
        self.WriteResults()
        self.SaveResults()

    def WriteResults(self):
        # writing different dependent variable with different independent variable
        for dependentVar in self.dataIns.DependentVarNames:
            for independentVar in self.dataIns.IndependentVarNames:
                message = f"Please help me write the results of the data analysis results. Only write three paragraph this time, including an overview of the mean values minus plus STDs of the data, data distribution and the significant results. This time, help me only write the effects of how different level of {independentVar} affect {dependentVar}. Here is what you have written: {self.ResultsStrs}"
                self.ResultsStrs += self.SendMessages(message) + "\n"

    def SaveResults(self):
        with open(self.dataIns.ResultsFolderPath + "Results Text.txt", "w") as file:
            file.write(self.ResultsStrs)

    def SendMessages(self,message: str)->str:
        # file 可以是多种类型
        # purpose 目前只支持 "file-extract"
        file_object = self.client.files.create(file=Path(self.dataIns.ResultsFilePath), purpose="file-extract")
        file_content = self.client.files.content(file_id=file_object.id).text
        
        finalMessages = [
            self.systemPrompt,
            {
                "role":"user",
                "content":file_content,
            },
            {
                "role": "user",
                "content": message,
            },
        ]

        completion = self.client.chat.completions.create(
            model="moonshot-v1-auto",
            messages=finalMessages,
            temperature =0.3,
            response_format={"type": "json_object"},
        )
        content = json.loads(completion.choices[0].message.content)
        # Print the response for debugging purposes
        print(content["Results"])
        return content["Results"]



