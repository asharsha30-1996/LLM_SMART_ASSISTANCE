This Readme is classified into two parts: First Part is the general intro of the project and the second part is about running the application.

# Part-1

This was developed as part of a course assignment. **LLM_SMART_ASSISTANCE** is a series of simple applications that offer a variety of functionalities, including:

  1. Reading multiple PDF documents and summarizing the results.
  2. Searching the web for real-time information.
  3. Scheduling meetings.
  4. Drafting and sending emails to recipients.
  5. Deploying a local LLM to store private information for future retrieval.

**Important Considerations**
We are not supposed to leak our personal information like email address, password etc to public LLMs

Note: Also planning to host the working of the above cases in the streamlit global cloud community. Currently the streamlit app is working locally.

Now let us look at the steps of each of the above use cases:

# 1. Reading multiple PDF documents and summarizing the results.

Users here can upload multiple PDF documents. There is a limit of 200MB per file while uploading. The text from these PDFs are split into smaller chunks using **RecursiveCharacterTextSplitter**. Further, we use **FAISS vector indexes** using OpenAI’s embeddings to store the processed text for fast similarity search.
We require user to interact with the system after uploading the PDF which might include asking relevant questions from any one of the PDFs, so we use **OpenAI's GPT-3.5 Turbo**. Interestingly, given the cost I tried with Gemini Pro but I ran into couple of issues that the resources exhausted despite trying only for fewer times. In this aspect, OpenAI's APIs are more stable in this aspect. Lastly, for the user to interact with the system seamlessly, the entire workflow is deployed in the streamlit local.

Below are some of the sample screenshots:

Once all files are uploaded hit the **Submit & Process** and wait until all files are processed successfully. I ensured that all PDFs which are uploaded are processed successsfully. Also I found that processing the PDFs sequentially help in retaining the FAISS data index properly else only half information are embedded. Not sure why.So, I made sure that each of the uploaded PDFs are processed one by one storing the embeddings on the FAISS data index store.

<img width="932" alt="image" src="https://github.com/user-attachments/assets/436913ea-2334-45a4-b688-a187242fa3be">

This screenshot contains the result. 

<img width="925" alt="image" src="https://github.com/user-attachments/assets/01153342-f157-44ab-99ae-7334cbe9516c">


# 2. Searching the web for real-time information.

This Streamlit-based Dynamic Search Agent app integrates **OpenAI GPT-3.5, Google Custom Search Engine (CSE), and SerpAPI** to provide real-time search results based on user queries. The agent decides dynamically which search tool to use, enhancing the efficiency and relevance of results.

I have defined two search tools here: **Google CSE Search:** Uses Google’s Custom Search Engine and **SerpAPI Search:** Mostly used for real time search

**How to setup SerpAPI:**

Using any gmail credential we can create a SerpAI but note that we are given over 100 free searches using this API for a month for the users who are on a free trail

**How to setup GOOGLE_API_KEY:**

You can follow the detailed instructions in **https://support.google.com/googleapi/answer/6158862?hl=en**

**How to setup Google Custom Search Engine (CSE):**

1. Go to **https://programmablesearchengine.google.com/about/**. Click on **Get Started**
2. You will get the below window
<img width="763" alt="image" src="https://github.com/user-attachments/assets/16f9f41b-a4ff-496f-9f1d-60af46088711">

3.Rest others are trivial. You can specify the search engine name and also you can customize the API based on the searches you need to have. For example, I created CSE only for the purpose to search the information related to the following websites below:

<img width="488" alt="image" src="https://github.com/user-attachments/assets/d83c3ca5-f75e-4e0c-85c6-27c8a1eaa287">

Once after specifying the above information, you will be shared with the CSE Id. Copy that Id and the window looks something like below:

<img width="740" alt="image" src="https://github.com/user-attachments/assets/9f8bf6f4-866b-4a05-a753-0b5f1e2f389a">

You can find the CSE information here: (Scroll towards right, I covered only the left part to mask sensitive information)

<img width="364" alt="image" src="https://github.com/user-attachments/assets/e2b039d8-e7cd-4bd0-a0c2-0724a41ac323">

Having done with all these steps, we are set with the app to run and execute on streamlit. I used an agent using LangChain's **ZERO_SHOT_REACT_DESCRIPTION** agent type, which dynamically selects the appropriate search tool.

Sample Output is shown below from Streamlit frontend and processing steps from backend:

<img width="929" alt="image" src="https://github.com/user-attachments/assets/21646366-cf2a-4fb1-ac25-e54a11d13d39">

Backend Chain of Thoughts:

<img width="689" alt="image" src="https://github.com/user-attachments/assets/30d3737b-6f36-4753-b756-e3356766d16b">

# 3.Scheduling meetings.

This is personally and by far the toughest of all parts due to certain issues like 1. Handling natural language response from the user 2. Check for potential conflicts
3. Proper parsing of dates and certain date oriented words like today, next week, after two weeks etc....I felt prompting, parsing dates and extracting date oriented words and process them accurately to be more difficult.

I used the **Streamlit-based Interactive Google Calendar Meeting Scheduler app** uses **OpenAI's GPT-3.5** and **Google Calendar API** to help users find available time slots and schedule meetings through natural language commands

**How to create the Google Calendar API?**

Follow this page **https://support.google.com/cloud/answer/6158849?hl=en**. Download the credentials.json at the last.

During the initial execution you will be prompted to login to necessiate access to your gmail account. (Tip if you encounter any issues, try adding test users, your email is fine too). To make it one time process for authentication, a **token.json** is created. If **token.json** is not found (for the first time), it initiates an authentication flow and saves the credentials locally.

Also implemented the logic to retrieve the free time slots within business hours (9:00 AM – 6:00 PM) for a selected day by checking existing events in the user's calendar
Uses **OpenAI GPT-3.5** to parse user commands for scheduling meetings. The response is converted to a structured JSON format containing:
    1. Meeting name
    2. start time
    3. end time
    4. date
However we used Natural Language way of processing the text like if the user specifies " Schedule Meeting in two days or next Monday" it would work well.

Below there are three pairs of screenshots of demo featuring from Streamlit UI and Google Calendar Interface. The demo are categorized into three types:

  1.  To schedule meeting during the conflict. To find if we get the pop out for the "Conflicts"
  2.  To schedule a proper meeting with well defined input prompt containing meeting name, start time, end time and date
  3.  Try creating meeting with user natural language to test if the prompt is able to identify the details

For each of these screenshots above we also have my Google Calendar to show the changes that are getting reflected:

**Demo1: To schedule meeting during the conflict.**

<img width="794" alt="image" src="https://github.com/user-attachments/assets/d770a077-84d7-4d73-9534-0d5c0ecbd771">

**Created Meeting on conflicts:**

<img width="365" alt="image" src="https://github.com/user-attachments/assets/124bfbd1-b1f7-4b96-b2e1-9e8ce9ee1058">

**Google Calendar:**

<img width="655" alt="image" src="https://github.com/user-attachments/assets/d6bda079-0f4c-4b55-a51a-26e0c87f91d0">

**Demo2: To schedule a proper meeting with well defined input prompt containing meeting name, start time, end time and date.**

<img width="484" alt="image" src="https://github.com/user-attachments/assets/cae8a90e-18c5-4d2c-961d-b1fd76b3851a">

**Changes Reflected on Google Calendar**
<img width="595" alt="image" src="https://github.com/user-attachments/assets/b176cb47-deb8-411a-9f03-580695d9b6ed">

**Demo3: Try creating meeting with user natural language to test if the prompt is able to identify the details.**

<img width="416" alt="image" src="https://github.com/user-attachments/assets/64452623-81dd-4e13-946b-4d3ba5d81d82">

**Changes Reflected on Google Calendar**

<img width="744" alt="image" src="https://github.com/user-attachments/assets/af00ab6c-34be-4232-89bd-a073c27291a0">

Given in the prompt to handle all possible date oriented words like 'next monday, next friday, next week, next month etc' (as per testing they are working fine)

**Note: We never exposed the tokens to openAI we used openAI only for prompting.**


# 4.Drafting and sending emails to recipients:

This Streamlit-based Email Drafting App integrates LangChain, Gmail API, and OpenAI GPT-3.5 to help users generate and manage email drafts through natural language input.

The setup of auth remains the same as that of meeting usecase discussed above:

The Langchain configuration contains **GPT-3.5** with deterministic responses (temperature=0) for consistent output. Hosted locally through Streamlit.

Below is the screenshot of demo

**Preparing the Draft of the email**

<img width="427" alt="image" src="https://github.com/user-attachments/assets/628dbc6d-20eb-48c9-80f9-4e7958628b36">

On my Gmail:

<img width="371" alt="image" src="https://github.com/user-attachments/assets/697f0c23-7164-465a-a1c4-20447d7bd5d1">

**Now send the email from the interaction box itself. Like the one below:**

<img width="338" alt="image" src="https://github.com/user-attachments/assets/f4bb4a82-57c2-4904-86d0-a543b7efc334">

**On the Gmail of the Recipient:**

<img width="719" alt="image" src="https://github.com/user-attachments/assets/e028c194-63f3-44d4-85e0-4af823e1d5a5">

**Note: We never exposed the tokens to openAI we used openAI only for prompting.**

# 5.Deploying a local LLM to store private information for future retrieval.

This is more like making use of local LLMs like ollama to store and retrieve personal information. Used sqlite3 as a DB to store information and later retrieved information using ollam's local llama 3.1 B model to interact with the information shared.

Below is the sample screenshot of the output. Also the local model took longer time to run and my system is fronzen for 15 minutes.

<img width="274" alt="image" src="https://github.com/user-attachments/assets/f7f29081-3bda-47d1-933f-3e8542628f14">
<img width="323" alt="image" src="https://github.com/user-attachments/assets/acafecdc-0c42-4856-b611-c01ed17c152d">
<img width="368" alt="image" src="https://github.com/user-attachments/assets/ff62421e-c8cd-4d42-b1aa-afacf62010fd">

Hope I summarized the contents related to the applications and now we shall look at the steps to execute the same

# Part-2

1. Activate .venv by .venv\Scripts\activate after specifying the proper path of the code execution. Note that all installations are done on virtual environment.
2. Run the main.py which acts as a first stage user interface where the user will be prompted to select the type of application to run as shown below:

<img width="413" alt="image" src="https://github.com/user-attachments/assets/f5230155-4c7a-467e-a0ce-aa116a9169bc">

3. Then the particular application will be opened on streamlit.
4. If you are planning to run this project, ensure installing all packages as specified in requirements.txt.
5. Streamlit is not supported in the lates 3.12.0 and anything less than this version no problem.
6. I used FAISS instead of ChromaDB as I could not install successfully in older versions of Pyhton <12.0.

Please share feedback for further improbvement and this is my first time learning of this RAG and Langchain use cases.

# References:

1. https://youtu.be/uus5eLz6smA?si=HyTYXVDheuUNJYoV - Multiple PDFs
2. https://medium.com/@gk_/chatgpt-and-langchain-an-attempt-at-scheduling-automation-part-2-of-3-6e38b3c086d5 - For meeting Schedule
3. https://youtu.be/Jq9Sf68ozk0?si=spinx0Lzkr2Hoz7i - For email Agent






















