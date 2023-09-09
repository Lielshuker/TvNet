<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="300" height="150">
  </a>
</div>

<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
### About the project

TvNet is a Web application where you can create rooms for video and chat while watching movie with your friends.  
This project is based on MVC architecture with Backend written in Python and Frontend written in React.  
the js code can be found at the following link:  
https://github.com/Lielshuker/Tv-Net-client.git  
Using Firebase for messages, AgoralO platform for video chatting, and MySQL db for saving movies and users data.  
In addition, it contains a Recommender system based on the Movielens model - the data for the model is taken from Movielens db.  

![TvNet-screenshot1][TvNet-screenshot1]

### Features
* Create watch party rooms.
* Watch movies in sync with your friends.
* Chat with your friends while watching movies.
* get recommendations to movies based on movies you wathced.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python]][Python-url]
* [![Flask][Flask]][Flask-url]
* [![React][React.js]][React-url]
* [![MySQL][MySQL]][MySQL-url]
* [![Firebase][Firebase]][Firebase-url]
* [![Agora.io][Agora.io]][Agora.io-url]
* [![Docker][Docker]][Docker-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites: 
#### backend:
* flask:
  ```sh
  pip install flask
  ```
* docker  
#### frontend:
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

#### backend:
to setting up the backend server project locally follow the instructions below:
* clone the project 
* install requirements.txt:
  ```sh
  pip install requirements.txt
  ```
* run MySQL image at the docker.python  
* set your FLASK_CONFIG and FLASK_APP environment variables before trying to run your Flask app:
  ```sh
  # Window Users
  set FLASK_CONFIG=development
  set FLASK_APP=manage.py
  flask run
  ```
* run flask server
  ```sh
  flask run
  ```


#### frontend:
to setting up the backend server project locally follow the instructions below:
* clone this project
* 
  ```sh
  npm start
  ```

* Go to [http://localhost:3000/](http://localhost:3000/)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Usage
first you need to create user at the website
![TvNet-screenshot0][TvNet-screenshot0]

after you signed in, you get your main page with list of movies
![TvNet-screenshot2][TvNet-screenshot2]

when you found movie you have 3 option: watch alone, host a room or join a room
![TvNet-screenshot4][TvNet-screenshot4]

now you can start your watching party:)
![TvNet-screenshot3][TvNet-screenshot3]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[TvNet-screenshot0]: images/first_page.PNG
[TvNet-screenshot1]: images/main.PNG
[TvNet-screenshot2]: images/2.PNG
[TvNet-screenshot3]: images/3.PNG
[TvNet-screenshot4]: images/4.PNG
[Flask]: https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white
[Flask-url]: https://flask.palletsprojects.com/en/2.3.x/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[MySQL]: https://img.shields.io/badge/mysql-%2300f.svg?style=for-the-badge&logo=mysql&logoColor=white
[MySQL-url]: https://www.mysql.com/
[Firebase]: https://img.shields.io/badge/firebase-%23039BE5.svg?style=for-the-badge&logo=firebase
[Firebase-url]:https://firebase.google.com/
[Agora.io]: https://img.shields.io/badge/agora.io-white?style=for-the-badge&logo=agora&logoColor=4285F4
[Agora.io-url]: https://www.agora.io/en/
[Docker]: https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white
[Docker-url]: https://www.docker.com/
