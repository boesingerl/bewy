- # Bewy, a Group Recommender system for your parties

  In this file, we will give you all instructions and pointers to access our work on a robust Group Recommender System using the last.fm dataset.

  ## Introduction to our work

  In order to run our notebook in the best way possible, here are the libraries which you will need :

  ```
  Unidecode == 1.2.0
  dask == 2.30.0
  h5py == 3.1.0
  implicit == 0.4.4
  ipywidgets == 7.5.1
  matplotlib == 3.3.2
  numpy == 1.19.5
  pandas == 1.1.3
  plotly == 4.14.3
  scikit_learn == 0.23.2
  scikit_surprise == 1.1.1
  scipy == 1.4.1
  seaborn == 0.11.0
  spotipy == 2.18.0
  tqdm == 4.43.0
  yellowbrick == 1.3.post1
  ```
  
  And here is a possible way to download them (consider creating a virtual environment) :

  ```
  pip install -r requirements.txt
  ```

  In order to have the clearest project possible, all our work has been compartmentalized into many different notebooks, all of which will be described in the next section.

  ## Per Notebook Description

  ### 01 - Data Exploration.ipynb
  
  This notebook is the first notebook we created, back in the first milestones when we wanted to find out about the data.
  
  ### 02 - Track Data Creation.ipynb
  
  This notebook serves to load and clean the data in the 1K Dataset, which we will use everywhere in the rest of the project.
  
  ### 03 - Track Collaborative Recommender.ipynb 
  
  This notebook contains all things relevant to the Collaborative Recommender Part of the project (selecting the ratings, the model, impact on relevance).
  It has several parts :

  - Creating the Ratings
  - Selecting the Model
  - Validating the Model

  ### 04 - Content Recommender.ipynb

  This notebook contains all things relevant to the Content based Recommender Part of the project.The spotify data contains for each track many features like danceability, energy, loudness, speechiness, acousticness, instrumentalness, as well as the genre of the Artist.The goal is to create a clustering among the Tracks.This will help for two reasons : 

  - It is easy to compute the goodness of fit of a clustering
  - We will be able to sample Tracks from a cluster, to make the user discover new similar songs
    - This adds stochasticity, and thus variety, the playlist will never be exactly the same
    - We should have enough clusters to guarantee that tracks in a same cluster are close enough for them to be recommended
  
  ### 05 - Heuristic Recommender.ipynb
  
  This notebook contains all things relevant to the Heuristic based Recommender Part of the project.The Heuristic Recommender has several parts :

  - Recommending Songs Users have in Common
  - Recommending Songs from Artists Users have in Common
  - Recommending Songs from Popular Artists in Genres User have in Common
  
  In order to determine what users have in Common, we don't simply compute an intersection : this wouldn't scale, for a large amount of users, we would have absolutely no recall.
  
  Instead, we create some voting system, where each user has a vote, and we take all items which have enough votes.
  
  ### 06 - Merged Recommender.ipynb
  
  This notebook contains a working prototype that uses voilà and jupyter widgets to provide a front-end to the application.
  
  ### 07 - Artist Data Creation.ipynb
  
  This notebook closely resembles the "02 - Track Data Creation" but is applied on the 360K Dataset, and saves the Data in HDF5, a format readable for the implicit library, which we will use instead of surprise, as it allows us to use a GPU to train.
  
  ### 08 - Artist Collaborative Recommender.ipynb
  
  In this notebook, we create a Collaborative Recommender based on Artists. 
  Since we are now using the **implicit** python library, we have to make some changes to the code we used before (we are using Scipy CSR matrices instead of our previous pandas DFs).We didn't have the time to completely include this new recommender in the pipeline, but we still have results we would like to share.
  
  


---

  ### Authors
  - Léopaul Boesinger
  - Marc Egli
  - Yassine Khalfi