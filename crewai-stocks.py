#!/usr/bin/env python
# coding: utf-8

# #LIB INSTALATION
# !pip install yfinance==0.2.41
# !pip install crewai==0.28.8
# !pip install 'crewai[tools]'
# !pip install langchain==0.1.20
# !pip install langchain-openai==0.1.7
# !pip install langchain-community==0.0.38
# !pip install duckduckgo-search==5.3.0
# !pip install python-dotenv
# !pip install streamlit

# In[8]:


import json
import os
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv

import yfinance as yf

import streamlit as st


# In[9]:


#Yahoo Finance Tool
def fetch_stock_price(ticket):
  stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
  return stock

yahoo_finance_tool = Tool(
  name = "Yahoo Finance Tool",
  description = "Fetch Stock prices for {ticket} from the last year about a specific stock from API",
  func = lambda ticket: fetch_stock_price(ticket)
)

#DuckDuck Go Tool
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)


# In[10]:


# IMPORT OPENAI LLM - GPT
load_dotenv()
#os.environ['OPENAI_API_KEY'] == os.getenv('OPENAI_API_KEY') #local
os.environ['OPENAI_API_KEY'] == st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")


# In[11]:


stockPriceAnalyst = Agent(
  role = "Senior stock price analyst",
  goal = "Find the {ticket} stock price and analyze trends",
  backstory = "You're highly exprienced in analyze the price of an specific stock and make predictions about the future price and get some external factors and historical logs study if is worth invest in the moment",
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  allow_delegation = False,
  tools = [yahoo_finance_tool]
)

newsAnalyst = Agent(
  role = "Stock news analyst",
  goal = "Create a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down or sideways with the news context. For each requested stock specify a number between 0 and 100 where 0 is bad and 100 is good",
  backstory = "You're a highly experienced in analyzing the market trends and news tracking assets for more then 10 years, you're also master level analytics in the tradictional markets and have deep understanding of human psychology. You understand news, theris titles and information but you look at those with a health dose of skepticism. You consider also the source of the news and check if is fake",
  verbose = True,
  llm = llm,
  max_iter = 10,
  memory = True,
  allow_delegation = False,
  tools = [search_tool]
)

stockAnalystWritter = Agent(
  role = "Senior Stock Analyst Writer",
  goal = "Analyze the trends price and news and write an insightfull compelling and informative long newsletter based on the stock report and price trend.",
  backstory = "You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories and narratives that resonate with wider audiences. You understand macro factors and combine multiple theories - eg. cycle theory, and fundamental analyses. You're able to hold a multiple opinions when analyzing anything",
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  allow_delegation = True,
)


# In[12]:


getStockPrice = Task(
  description = "Analyze the stock {ticket} price history and create a trend analyzes of up, down or sideways",
  agent = stockPriceAnalyst,
  expected_output = "Specify the current trend stock price - up , down or sideways eg. stock='APPL, price UP' ",
)

getNews = Task(
  description = f"Take the stock and always include BTC to it or others trend criptos (if not request). Use the search tool to search one individually. The current date is {datetime.now()}. Compose the results into a helpful report",
  agent = newsAnalyst,
  expected_output = "A summary of the overall market and return a sentence for each request asset. Include a good/bad score asset based on the news. Use format: <STOCK ASSET> <SUMMARY NEWS> <TREND PREDICTION> <GOOD/BAD SCORE> ",
)

WriteAnalyses = Task(
  description = "Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company. A brief talk and highlight the most important points. focus on the stock price, news and good/bad score. what are the near future considerations? Also, include the previous analyzes of stock trend and news summary.",
  agent = stockAnalystWritter,
  context = [ getStockPrice, getNews ],
  expected_output = "Return a newsletter formatted as markdown in an easy readable manner. It should contain: -3 bullets executive summary, -introduction (set the overall picture and spike up the interest), - main part (provides the meat of the analysis including the news, summary, and good/bad scores), - summary (key facts and concrete future trends prediction about, - up, down or sideways),   ",
)


# In[13]:


crew = Crew(
  agents = [ stockPriceAnalyst, newsAnalyst, stockAnalystWritter ],
  tasks = [ getStockPrice, getNews, WriteAnalyses ],
  verbose = 2,
  process = Process.hierarchical,
  full_output = True,
  share_crew = False,
  manager_llm = llm,
  max_iter = 15,
)


# In[14]:
with st.sidebar:
  st.header('Enter Ticket Stock')

  with st.form(key='research_form'):
    topic = st.text_input("Select the Ticket")
    submit_button = st.form_submit_button(label = "Run Research")

if submit_button:
  if not topic:
    st.error("Please fill the ticket.")
  else:
    results = crew.kickoff(inputs={'ticket': topic})

    st.subheader("Results:")
    st.write(results['final_output'])

