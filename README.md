Session-20---Group-3-main/

Pre-requisite installations.
In order for you to run this project, you must install the following:

- Visual Studio Code (Download here: https://code.visualstudio.com/download)
- Node.JS LTS (Download here: https://nodejs.org/en/download) For Windows 
- Python 3.13 or Higher (Download here: https://www.python.org/downloads/release/python-3131/)
- Pip Package Manager (Check to install this in the Python 3.10 Installation Menu)

Instructions For Setup

1. Open Visual Studio Code
2. Load the extracted folder, "Session-20---Group-3-main"
3. Open the terminal 
    - Click "Split terminal" to create a second terminal. You should see two terminals

Installing dependencies:
In order for this project to run, you must install these indepdencies via Pip Package Manager
- On one of the split terminals, type "cd backend/fastapi" and then type "pip install -r requirements.txt" without the quotation marks

This step should install the following packages:
1. fastapi
2. uvicorn
3. pandas
4. Joblib
5. pydantic

Running the program:
In order to run the program, you must first enter the directories of both the Backend and the Frontend
On your previously-setup split terminal, on one, type "cd backend/fastapi", and on the other one write "cd frontend"

On the frontend directory, type "npm install". After it is done, type "npm run dev"
This will start the frontend. You can access this by opening a web-browser and going to the address "http://localhost:3000"

On the Backend directory, type "uvicorn main:app --reload".
This will start the frontend. You can access this by opening a web-browser and going to the address "http://localhost:8000/docs#"
