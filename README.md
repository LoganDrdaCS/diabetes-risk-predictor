# C964 Computer Science Captsone

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#visualizations--descriptive-methods">Visualizations / Descriptive Methods</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project aims to predict the onset of diabetes using supervised machine learning, specifically the logistic regression algorithm. Developed as part of a broader academic initiative, the application uses real-world health metrics such as glucose levels, blood pressure, BMI, and age to assess the likelihood of diabetes onset.

In this tool, each input factor reflects key features of diabetes risk, with the model trained on a robust dataset to improve its predictive accuracy. This application is framed around a hypothetical company, Glycure, which specializes in diabetes research and treatment development. Glycure's goal is to create a reliable, user-friendly application that empowers healthcare providers with early predictions to make informed decisions on patient management.

Key elements of the project include:
- **Data Processing**: Using `pandas` and `numpy` for data handling, cleaning, and feature preparation.
- **Machine Learning**: Utilizing `scikit-learn` for model building and evaluation to provide meaningful accuracy and classification reports.
- **User Interface**: A graphical interface, built with Python's `tkinter` module, allowing healthcare providers to input patient data and receive predictions, designed to maximize usability.

The project leverages an accessible GUI, which includes fields for data input and dynamically displays the prediction result along with a probability score, enabling a straightforward experience for end-users.


<!-- GETTING STARTED -->
## Getting Started

This project may be run directly from an IDE that support Python.

### Prerequisites

A user requires 3 libraries for the program to operate:

* pandas
  ```sh
  python -m pip install pandas
  ```
* NumPy
  ```sh
  python -m pip install numpy
  ```
* scikit-learn
  ```sh
  python -m pip install scikit-learn
  ```

<!-- USAGE EXAMPLES -->
## Usage

Upon running the program, some relevant data will be output to the terminal. A new window will open, allowing the user to input data and use the trained model to make a prediction. The default values in the GUI are the median values of each feature.

![GUI screenshot](images\demo_screenshot.jpg)

<!-- VISUALIZATIONS / DESCRIPTIVE METHODS -->
## Visualizations / Descriptive Methods

Here I have 3 pictures

<!-- CONTACT -->
## Contact

Logan Drda - logan.drda.cs@gmail.com

Student ID: 011010779