# Social Labeling API for Appraisal System

Django API for integrating the Machine Learning (NLP) Model to the Appraisal Website.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10 installed
- pip package manager
- [Optional] Virtual environment for isolation

## Installation

1. Clone the repository:

   ```bash
   https://github.com/Anurag-cod4fun/Social_Labeling.git
   
2. Navigate to the project directory:

    ```bash
   cd Social_Labeling
   
3. [Optional] Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use "venv\Scripts\activate"
   
4. Install dependencies:
   ```bash
   pip install -r requirements.txt

5. Apply database migrations:
   ```bash
   python manage.py migrate
6. Start Development Server:
   ```bash
   python manage.py runserver
   
7. Open your web browser and go to http://127.0.0.1:8000/ to view the application and http://127.0.0.1:8000//classify/ for API using POST request.



