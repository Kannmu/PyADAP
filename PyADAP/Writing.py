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
import time
from openai import OpenAI, RateLimitError
import PyADAP.Data as data
from pathlib import Path
# import ollama

systemPrompt = {
    "role": "system", 
    "content": f"""- Role: Academic Writing and Data Analysis Expert
- Background: The user requires transforming complex data analysis results into professional academic English text for writing the Results section of a paper. The user expects the expert to accurately capture the key information from the data and use a rich and appropriate academic vocabulary for description. I am studying the impact of two gaze thresholds and magnify rates in interactive systems on user interaction efficiency. In the designed experiment, the gaze threshold has four levels, and the magnify rate has three levels. Twenty participants performed the experiment under all combinations of the two independent variables, totaling 12 treatments. Each participant repeated each treatment four times. The dependent variable is the Total Completion Time.
- Profile: You are an expert with profound expertise in academic writing and data analysis. You are adept at using academic English and can transform complex data information into clear, precise, and professional academic texts.
- Skills: You possess strong data analysis capabilities, enabling you to quickly identify and distill key results. At the same time, you are proficient in academic English writing, capable of using a professional and diverse vocabulary to accurately describe data.
- Goals: Translating the user-provided data analysis results into a professionally written Results section of a paper in clear and accurate academic English.
- Constrains: Descriptions should remain objective and accurate, avoiding subjective conjecture. The vocabulary used should be professional yet understandable, steering clear of overly complex or obscure terminology. The focus should be on describing the significant and notable results from the data analysis.
- OutputFormat: 'A structured academic English Results section. Writing should be concise and won't refers every number in the text. Output in standard json format like: {{"Results":"Here is your output"}}. Remember to add star marks after the p value, using the rules below: *: 1.00e-02 < p <= 5.00e-02, **: 1.00e-03 < p <= 1.00e-02, ***: 1.00e-04 < p <= 1.00e-03, ****: p <= 1.00e-04. Numbers with more than six decimal places should be rounded to six decimal places. Try to remove the unit sign after the variable name. Use the variable name in lower case.'
- InputDataFormat: 'The data input is the text form of the data analysis results from a Excel file. The left side of '->' sign is one of the independent variables, and the right side of '->' sign is the level of this independent variable. And the right side of the '-->' sign means the dependent variable. 
- Workflow:
  1. Carefully review the data analysis results provided by the user.
  2. Identify and distill the key and significant results from the data.
  3. Use professional academic English vocabulary and sentence structures to clearly and accurately describe these results.
- Examples:
  - Example 1: Describing means and standard deviation trends
    'Recognition accuracy represents the system’s ability to detect and recognize actions accurately. The average recognition accuracy for the three blink count levels is 0.9619 ± 0.0248, 0.9438 ± 0.0392, and 0.9225 ± 0.0375, respectively. It can be observed that as the blink count increases, the recognition accuracy decreases.'
  - Example 2: Describing Statistical Significance
    'The data were then subjected to an SW normality test, indicating that all three data groups were normally distributed. The sphericity test confirmed that the data satisfied the assumption of sphericity (p = 0.5017 > 0.05).'
  - Example 3: Describing Data Distribution
    'A one-way ANOVA revealed a highly significant effect of different blink counts on total completion time (F (2,38) = 943.32, p = 0.000***).'
  -Example 3: Describing Interaction Effects
    'Therefore, two-way ANOVA was performed using the Greenhouse-Geisser correction method. The results revealed a significant effect on blink count (F (2,38) = 30.6518, p = 0.0000***) and blink count × blink side interaction (F (2,28) = 13.4087, p = 0.0017*) but no significant effect on blink side (F (1,19) = 0.0496, p = 0.8262 > 0.05).'
"""
}

# systemPrompt = {
#     "role": "system", 
#     "content": f"""You are an expert in academic writing, help me write the results section of my paper. Output in standard json format like: {{"Results":"Here is your output"}}. Remember to add star marks after the p value, using the rules below: *: 1.00e-02 < p <= 5.00e-02, **: 1.00e-03 < p <= 1.00e-02, ***: 1.00e-04 < p <= 1.00e-03, ****: p <= 1.00e-04. Numbers with more than six decimal places should be rounded to six decimal places. Try to remove the unit sign from the variable name and get the real variable name. Using plus and minus sign when describe mean value ± STD value of the data. Do not narrate your outputs in points."""
# }

class Writer():
    def __init__(self, dataIns:data.Data, apiKey = "sk-1093dcab736946a39f4887dd226e07a8"):
        self.systemPrompt = systemPrompt
        self.chatClient = OpenAI(
            api_key="sk-1093dcab736946a39f4887dd226e07a8",
            base_url="https://api.deepseek.com/v1",
        )
        self.fileClient = OpenAI(
            api_key="sk-caQnDXpwc00jFJu4LqO76ob5iPHeZzGwHuWdlZQZZFV9xMuv",
            base_url="https://api.moonshot.cn/v1",
        )
        self.dataIns = dataIns
        self.ResultsStrs = ""
        self.WriteResults()
        self.SaveResults()

    def WriteResults(self):
        # writing different dependent variable with different independent variable
        for dependentIndex, dependentVar in enumerate(self.dataIns.DependentVarNames):
            self.ResultsStrs += f"{dependentIndex+1}. {dependentVar}:\n"
            for independentIndex, independentVar in enumerate(self.dataIns.IndependentVarNames):
                self.ResultsStrs += f"\n{dependentIndex+1}.{independentIndex+1}. {independentVar}:\n"
                message = f"""Please help me write the results of the data analysis results. Do not narrate your outputs in points. please output in json format like: {{"Results":"Here is your output"}}. This time, help me only write the effects of how different level of {independentVar} affect {dependentVar}. Output in standard json format. Here is what you have written: '{self.ResultsStrs}' """
                self.ResultsStrs += self.SendMessages(message) + "\n\n"
            self.ResultsStrs += f"{dependentIndex+1}.{len(self.dataIns.IndependentVarNames)+2}. Interaction Analysis between {self.dataIns.IndependentVarNames[0]} and {self.dataIns.IndependentVarNames[1]}:\n"
            
            # Interaction effects of independent variable on dependent variable
            message = f"""Please help me write the interaction effect results of the data analysis results. Do not narrate your outputs in points. please output in json format like: {{"Results":"Here is your output"}}.This time, help me only write the interaction effects between {self.dataIns.IndependentVarNames[0]} and {self.dataIns.IndependentVarNames[1]} affect {dependentVar}. Output in standard json format. Here is what you have written: '{self.ResultsStrs}' """
            self.ResultsStrs += self.SendMessages(message) + "\n\n"

    def SaveResults(self):
        with open(self.dataIns.ResultsTextPath, "w", encoding='utf-8') as file:
            file.write(self.ResultsStrs)
        self.dataIns.Print2Log("***************** Auto-Writing-Results *****************")  
        self.dataIns.Print2Log("******* Please Double-Check the Results in the Text File *******")            
        self.dataIns.Print2Log("\n"+self.ResultsStrs)

    # def SendMessagesLocal(self,message: str):
    #     message = "This is my data analysis results: "+self.dataIns.ReadLogs()+ "\n" + message
    #     res = ollama.generate(
    #         model="qwen2.5:7b",
    #         stream=False,
    #         system=systemPrompt["content"],
    #         prompt=message,
    #         format="json",
    #         options={"temperature": 1},
    #     )

    #     res_dict = json.loads(res.response)
    #     print(res_dict)
    #     return res_dict["Results"]

    def SendMessages(self, message: str) -> str:
        max_retries = 3  # 最大重试次数
        retry_delay = 20  # 初始重试延迟（秒）
        retry_factor = 1  # 延迟增长因子

        for attempt in range(max_retries + 1):
            # try:
            # 创建文件对象
            file_object = self.fileClient.files.create(file=Path(self.dataIns.ResultsFilePath), purpose="file-extract")
            # 获取文件内容
            file_content = self.fileClient.files.content(file_id=file_object.id).text

            # 构建消息列表
            finalMessages = [
                systemPrompt,
                {
                    "role": "user",
                    "content": file_content,
                },
                {
                    "role": "user",
                    "content": message,
                },
            ]

            # 调用大模型API
            completion = self.chatClient.chat.completions.create(
                model="deepseek-chat",
                messages=finalMessages,
                temperature=1,
                response_format={"type": "json_object"},
                max_tokens=3000,
            )

            print(completion.choices[0].message)
            # 解析响应内容
            content = json.loads(completion.choices[0].message.content)

            # 打印响应内容用于调试
            # print(content["Results"])

            return content["Results"]

            # except RateLimitError as e:  # 假设API限速异常类为RateLimitError
            #     if attempt == max_retries:
            #         print(f"达到最大重试次数 {max_retries}，仍然失败。")
            #         raise e
            #     else:
            #         print(f"API调用达到速率限制，第 {attempt + 1} 次重试中... 等待 {retry_delay} 秒。")
            #         time.sleep(retry_delay)
            #         retry_delay *= retry_factor  # 增加重试延迟

            # except Exception as e:
            #     print(f"发生其他错误: {e}")
            #     raise e


