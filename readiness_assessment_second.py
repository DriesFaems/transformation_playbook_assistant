from groq import Groq
import streamlit as st
import os
import tempfile
from crewai import Crew, Agent, Task, Process
import json
import os
import requests
from crewai_tools import tool
from crewai import Crew, Process
import tomllib
from langchain_groq import ChatGroq
import pandas as pd
import datetime

# create title for the streamlit app

st.title('Generative AI Transformation Playbook Assistant')

# create a description

st.write(f"""This assistant is designed to help you analyzing the Generative AI readiness of your company. You will need to answer a number of questions. Based on the answers, you will receive a report on the Generative AI readiness of your company and suggestions on the next steps in terms of implementing generative AI. For more information, contact Dries Faems at https://www.linkedin.com/in/dries-faems-0371569/""")

groq_api_key = st.text_input('Please provide your Groq API Key. You can get a free API key at https://console.groq.com/playground', type="password")
os.environ["GROQ_API_KEY"] = groq_api_key

st.markdown('**Please provide the following information about your company**')

company_description = st.text_area('Please provide a description of your company. The more detail you provided, the better the advice will be.')

st.markdown('**Please answer the following multiple choice questions to assess the strategic readiness of your company for generative AI**')

question1 = 'Our leadership team understands the potential impact of generative AI on our business.'
answer1 = st.radio(question1, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question2 = 'We have a clear strategy for implementing generative AI across different departments'
answer2 = st.radio(question2, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question3 = 'Generative AI initiatives are aligned with our overall business objectives.'
answer3 = st.radio(question3, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question4 = 'Our company has a clear vision for how generative AI will transform our operations.'
answer4 = st.radio(question4, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

st.markdown('**Please answer the following multiple choice questions to assess the readiness of your company in terms of use cases for generative AI**')

question5 = 'We have identified specific use cases for generative AI within our organization.'
answer5 = st.radio(question5, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question6 = 'We have conducted pilot projects or proofs of concept for generative AI applications.'
answer6 = st.radio(question6, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question7 = 'We have metrics in place to measure the ROI of our generative AI projects.'
answer7 = st.radio(question7, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question8 = 'We regularly assess and prioritize new potential use cases for generative AI.'
answer8 = st.radio(question8, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

st.markdown('**Please answer the following multiple choice questions to assess the readiness of your company in terms of digital architecture**')

question9 = 'Our IT infrastructure is capable of supporting generative AI applications.'
answer9 = st.radio(question9, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question10 = 'Our data storage and management systems are well-suited for handling the large datasets required for generative AI.'
answer10 = st.radio(question10, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question11 = 'We have implemented strong data security and privacy measures to protect data used in generative AI applications.'
answer11 = st.radio(question11, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question12 = 'Our IT infrastructure is flexible enough to integrate generative AI tools and applications.'
answer12 = st.radio(question12, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

st.markdown('**Please answer the following multiple choice questions to assess the readiness of your company in terms of talent and skills**')

question13 = 'Our workforce is aware of what generative AI is and its potential applications.'
answer13 = st.radio(question13, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question14 = 'We have employees with the necessary skills to develop and implement generative AI solutions.'
answer14 = st.radio(question14, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question15 = 'There is a positive attitude towards adopting generative AI technologies among our staff.'
answer15 = st.radio(question15, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

question16 = 'Our company culture encourages experimentation and innovation with generative AI.'
answer16 = st.radio(question16, ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'])

if st.button('Start Analysis'):
    client = Groq()
    GROQ_LLM = ChatGroq(
            # api_key=os.getenv("GROQ_API_KEY"),
            api_key=groq_api_key,
            model="openai/gpt-oss-20b"
        )
    strategic_readiness_investigator = Agent(
        role='Investigating strategic readiness',
        goal=f"""Investigate the strategic readiness of the company for generative AI based on the provided company information and the answers to the multiple choice questions.""", 
        backstory=f"""You are a great expert in investigating the strategic readiness of a company for generative AI. You will investigate the strategic readiness 
        based on the provided company information and the answers to the multiple choice questions.""",
        verbose=True,
        llm=GROQ_LLM,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )
    # Create tasks for the agents
    investigate_strategic_readiness = Task(
        description=f"""Investigate the strategic readiness of the company for generative AI based on the provided company information and the answers to the multiple choice questions.
        Here is a description of the company: {company_description}. 
        Here is the answer to the relevant multiple choice questions: (1) {question1}: {answer1}, (2) {question2}: {answer2}, (3) {question3}: {answer3}, (4) {question4}: {answer4}""",
        expected_output='As output, you provide a clear and concise assessment of the strategic readiness of the company to implement generative AI. You should focus on the core strenghts and weaknessess and refrain from providing recommendations',
        agent=strategic_readiness_investigator
    )

    crew = Crew(
        agents=[strategic_readiness_investigator],
        tasks=[investigate_strategic_readiness],
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )
    # Kick off the crew's work and capture results
    results = crew.kickoff()

    strategic_readiness = investigate_strategic_readiness.output.raw_output
    
    usecase_readiness_investigator = Agent(
        role='Investigating use case readiness',
        goal=f"""Investigate the readiness of the company for generative AI use cases based on the provided company information and the answers to the multiple choice questions.""", 
        backstory=f"""You are a great expert in investigating the readiness of a company for generative AI use cases. You will investigate the readiness 
        based on the provided company information and the answers to the multiple choice questions.""",
        verbose=True,
        llm=GROQ_LLM,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )

    # Create tasks for the agents

    investigate_usecase_readiness = Task(
        description=f"""Investigate the readiness of the company for generative AI use cases based on the provided company information and the answers to the multiple choice questions.
        Here is a description of the company: {company_description}. 
        Here is the answer to the relevant multiple choice questions: (5) {question5}: {answer5}, (6) {question6}: {answer6}, (7) {question7}: {answer7}, (8) {question8}: {answer8}""",
        expected_output='As output, you provide a clear and concise assessment of the readiness of the company for generative AI use cases. You should focus on the core strenghts and weaknessess and refrain from providing recommendations',
        agent=usecase_readiness_investigator
    )

    crew = Crew(
        agents=[usecase_readiness_investigator],
        tasks=[investigate_usecase_readiness],
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )
    # Kick off the crew's work and capture results

    results = crew.kickoff()

    usecase_readiness = investigate_usecase_readiness.output.raw_output

    architecture_readiness_investigator = Agent(
        role='Investigating architecture readiness',
        goal=f"""Investigate the readiness of the company for generative AI architecture based on the provided company information and the answers to the multiple choice questions.""", 
        backstory=f"""You are a great expert in investigating the readiness of a company for generative AI architecture. You will investigate the readiness 
        based on the provided company information and the answers to the multiple choice questions.""",
        verbose=True,
        llm=GROQ_LLM,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )

    # Create tasks for the agents

    investigate_architecture_readiness = Task(
        description=f"""Investigate the readiness of the company for generative AI architecture based on the provided company information and the answers to the multiple choice questions.
        Here is a description of the company: {company_description}. 
        Here is the answer to the relevant multiple choice questions: (9) {question9}: {answer9}, (10) {question10}: {answer10}, (11) {question11}: {answer11}, (12) {question12}: {answer12}""",
        expected_output='As output, you provide a clear and concise assessment of the readiness of the company for generative AI architecture. You should focus on the core strenghts and weaknessess and refrain from providing recommendations',
        agent=architecture_readiness_investigator
    )

    crew = Crew(
        agents=[architecture_readiness_investigator],
        tasks=[investigate_architecture_readiness],
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )
    # Kick off the crew's work and capture results

    results = crew.kickoff()

    architecture_readiness = investigate_architecture_readiness.output.raw_output
    

    talent_readiness_investigator = Agent(
        role='Investigating talent readiness',
        goal=f"""Investigate the readiness of the company for generative AI talent based on the provided company information and the answers to the multiple choice questions.""", 
        backstory=f"""You are a great expert in investigating the readiness of a company for generative AI talent. You will investigate the readiness 
        based on the provided company information and the answers to the multiple choice questions.""",
        verbose=True,
        llm=GROQ_LLM,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )

    # Create tasks for the agents

    investigate_talent_readiness = Task(
        description=f"""Investigate the readiness of the company for generative AI talent based on the provided company information and the answers to the multiple choice questions.
        Here is a description of the company: {company_description}. 
        Here is the answer to the relevant multiple choice questions: (13) {question13}: {answer13}, (14) {question14}: {answer14}, (15) {question15}: {answer15}, (16) {question16}: {answer16}""",
        expected_output='As output, you provide a clear and concise assessment of the readiness of the company for generative AI talent. You should focus on the core strenghts and weaknessess and refrain from providing recommendations',
        agent=talent_readiness_investigator
    )

    crew = Crew(
        agents=[talent_readiness_investigator],
        tasks=[investigate_talent_readiness],
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )
    # Kick off the crew's work and capture results

    results = crew.kickoff()

    talent_readiness = investigate_talent_readiness.output.raw_output
    
    readiness_summarizer = Agent(
        role='Summarize generative ai readiness across dfferent dimensions',
        goal=f"""Summarize generative ai readiness across dfferent dimensions.""", 
        backstory=f"""You are a great expert in summarizing generative ai readiness across dfferent dimensions. 
        You will summarize generative ai readiness on dfferent dimensions. The purpose is to provide a clear and concise overview of the readiness 
        of the company for generative AI. It is crucial to avoid overlap and redundancies in observations.""",
        verbose=True,
        llm=GROQ_LLM,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )

    # Create tasks for the agents

    summarize_readiness = Task(
        description=f"""Provide the most important observations on the generative ai readiness based on the following input:
        - Strategic readiness: {strategic_readiness}
        - Use case readiness: {usecase_readiness}   
        - Architecture readiness: {architecture_readiness}
        - Human readiness: {talent_readiness}""",
        expected_output='As output, you provide a clear and concise overview of the generative ai readiness, making an explicit distinction between the dimensions. It is important to avoid overlap and redundancies.',
        agent=readiness_summarizer
    )

    crew = Crew(
        agents=[readiness_summarizer],
        tasks=[summarize_readiness],
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )

    # Kick off the crew's work and capture results

    results = crew.kickoff()

    st.markdown('**Generative AI Readiness Assessment**')
    assessment = summarize_readiness.output.raw_output
    st.write(summarize_readiness.output.raw_output)

    # create a digital vision based on company description, and readiness assessment

    digital_vision_agent = Agent(
        role='Create digital vision',
        goal=f"""Create a digital vision based on the provided company information and the generative ai readiness assessment.""", 
        backstory=f"""You are a great expert in creating a digital vision. 
        You will create a digital vision based on the provided company information and the generative ai readiness assessment.""",
        verbose=True,
        llm=GROQ_LLM,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )

    # Create tasks for the agents

    create_digital_vision = Task(
        description=f"""Create a digital vision based on the provided company information and the generative ai readiness assessment. A high-quality digital vision should be clear, concise, measurable and actionable.
        Here is a description of the company: {company_description}. Here is a description of the current Generative AI readiness: {strategic_readiness}.""",
        expected_output='As output, you provide a clear and concise digital vision based on the provided company information and the generative ai readiness assessment. The digital vision should encompass a maximum of two pillars',
        agent=digital_vision_agent
    )

    usecase_development_agent = Agent(
        role='Develop use cases',
        goal=f"""Develop use cases based on the provided company information and the generative ai readiness assessment.""", 
        backstory=f"""You are a great expert in developing use cases. 
        You will develop use cases based on the provided company information and the generative ai readiness assessment.""",
        verbose=True,
        llm=GROQ_LLM,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )

    # Create tasks for the agents

    develop_use_cases = Task(
        description=f"""Develop two use cases based on the provided company information and the generative ai readiness assessment that can help to realize the digital vision developed by the digital_vision_agent. Choose use cases that can deliver visible and measurable value
                    Start with use cases that are technically achievable with the current capabilities of the company
                    Aim for projects where failure won't have a significant negative impact, but success can make a big difference
                    Focus on projects that can be implemented relatively quickly
                    Select use cases that can scale across the organization once proven successful
                    Here is a description of the company: {company_description}. Here is a description of the current Generative AI readiness: {usecase_readiness}.""",
        expected_output='As output, you provide two use cases based on the provided company information and the generative ai readiness assessment.',
        agent=usecase_development_agent
    )

    digital_architecture_agent = Agent(
        role='Optimize digital architecture',
        goal=f"""Generate specific recommendations on how to optimize the digital architecture based on the provided company information and the generative ai readiness assessment in order to realize the digital vision and implement the proposed use cases.
                Recommendations should deal with technical topics such as data storage, data processing, data security, and system integration.""", 
        backstory=f"""You are a great expert in providing specific recommendations on how to improve the digital architecture. A high quality digital architecture is well structured, well integrated, and only as complex as absolutely necessary""",
        verbose=True,
        llm=GROQ_LLM,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )

    # Create tasks for the agents

    optimize_digital_architecture = Task(
        description=f"""Generate maximum two specific recommendations on how to optimize the digital architecture based on the provided company information and the generative ai readiness assessment
        in order to realize the digital vision, which is proposed by the digital_vision_agent and implement the use cases, which are proposed by the usecase_development_agent. 
        Here is a description of the company: {company_description}. Here is a description of the current Generative AI readiness: {architecture_readiness}.""",
        expected_output='As output, you provide specific recommendations on how to optimize the digital architecture based on the provided company information and the generative ai readiness assessment.',
        agent=digital_architecture_agent
    )

    improve_human_readiness_agent = Agent(
        role='Improve human readiness',
        goal=f"""Generate specific recommendations on how to improve human readiness based on the provided company information and the generative ai readiness assessment in order to realize the digital vision and implement the proposed use cases.""", 
        backstory=f"""You are a great expert in providing specific recommendations on how to improve human readiness. A high quality human readiness is characterized by a positive attitude towards generative AI, a culture of experimentation and innovation, and a workforce with the necessary skills""",
        verbose=True,
        llm=GROQ_LLM,
        allow_delegation=False,
        max_iter=5,
        memory=True,
    )

    # Create tasks for the agents

    improve_human_readiness = Task(
        description=f"""Generate a maximum of two specific recommendations on how to improve **human readiness** for generative AI. Focus strictly on improving employee skills, fostering a positive culture towards generative AI, and creating an environment that encourages experimentation and innovation.
        Here is a description of the company: {company_description}.
        Here is a description of the current Generative AI readiness: {talent_readiness}.""",
        expected_output="""As output, provide two actionable recommendations aimed at improving human readiness. Focus on training, cultural shifts, leadership involvement, and skills development. Avoid mentioning technical infrastructure or digital architecture.""",
        agent=improve_human_readiness_agent
    )

    crew = Crew(
        agents=[digital_vision_agent, usecase_development_agent, digital_architecture_agent, improve_human_readiness_agent],
        tasks=[create_digital_vision, develop_use_cases, optimize_digital_architecture, improve_human_readiness],
        process=Process.sequential,
        full_output=True,
        share_crew=False,
    )

    # Kick off the crew's work and capture results

    results = crew.kickoff()

    st.markdown('**Digital Vision**')

    digital_vision = create_digital_vision.output.raw_output

    st.write(create_digital_vision.output.raw_output)

    st.markdown('**Use Cases**')

    use_cases = develop_use_cases.output.raw_output

    st.write(develop_use_cases.output.raw_output)

    st.markdown('**Digital Architecture Recommendations**')

    digital_architecture_recommendations = optimize_digital_architecture.output.raw_output

    st.write(optimize_digital_architecture.output.raw_output)

    st.markdown('**Human Readiness Recommendations**')

    human_readiness_recommendations = improve_human_readiness.output.raw_output

    st.write(improve_human_readiness.output.raw_output)

else:
    st.write('Please click the button to start the analysis')
