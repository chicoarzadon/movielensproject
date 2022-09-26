##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

#install package if needed
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

#initializing packages needed
library(tidyverse)
library(caret)
library(data.table)

#initialize dl for file address of future downloaded temp file
dl <- tempfile()

#downloading MovieLens 10M dataset and saving it in dl variable
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

#data wrangling from ratings.dat from the downloaded file, with column names userId, movieId, rating, and timestamp
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

#data wrangling from movies.dat from the downloaded file, with column names movieId, title, and genres
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


#converting movies to a dataframe, making movieId as numeric, title as character, and genres as character
movies <- as.tibble(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

#combining data from movies, and data from ratings by movieId
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
#setup for creating validation set that is comprised of 10% of data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
#creation of edx dataset
edx <- movielens[-test_index,]
#creation of temp dataset for validation
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
#finish creation of validation dataset
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
#create edx dataset combining existing and the removed data
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#remove variables not needed in training
rm(dl, ratings, movies, test_index, temp, movielens, removed)


#creation of train set and test set
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
#creation of train_set dataset used for training
train_set <- edx[-test_index,]
#creation of temp dataset for test_set
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
#finish creation of test_set
test_set <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from test_set set back into train_set
#create train_set dataset used for training combining existing and the removed data
removed <- anti_join(temp, validation)
train_set <- rbind(train_set, removed)

#remove variables not needed in training
rm(test_index, temp, removed)

#RMSE function to test RMSE comparing true ratings from predicted ratings
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


### Recommendation System ###
#create recommendation system for users using train_set dataset
#using ratings, movieId, userId, and genres


## Model 1 ##

#average rating of all movies for all users
m_all <- mean(train_set$rating)
#m_all = mean of the ratings for all movies and all users

#rmse for average of everything
all_mean_rmse <- RMSE(test_set$rating, m_all)

#save rmse to rmse_results tibble
rmse_results <- tibble(method = "Just Average of Everything Model", RMSE = all_mean_rmse)

#display rmse results for models
rmse_results %>% knitr::kable()

## Model 2 ##

#average rating for each movie
movie_mean <- train_set %>% 
  group_by(movieId) %>% 
  summarize(m_movie = mean(rating - m_all))
#m_movie = mean of the differences between mean of the ratings for each movie and m_all (refer above)  

#making rating predictions based on mean rating of everything, and each movies
predicted_ratings <- test_set %>% 
  left_join(movie_mean, by = "movieId") %>%
  mutate(pred = m_all + m_movie) %>%
  .$pred


#get rmse for movie mean predicted ratings
movie_mean_rmse <- RMSE(test_set$rating, predicted_ratings)

#save rmse to rmse results
rmse_results <- bind_rows(rmse_results,
                          tibble(method = "Movie Mean Effect Model",  
                                     RMSE = movie_mean_rmse ))
#display rmse results for models so far
rmse_results %>% knitr::kable()

## Model 3 ##

#mean rating for each user
user_mean <- train_set %>%
  left_join(movie_mean, by = "movieId") %>%
  group_by(userId) %>%
  summarize(m_user = mean(rating - m_all - m_movie))

#predicted ratings based on mean of everything, per movie, and per user
predicted_ratings <- test_set %>% 
  left_join(movie_mean, by = "movieId") %>% left_join(user_mean, by = "userId") %>%
  mutate(pred = m_all + m_movie + m_user) %>%
  .$pred

#rmse computation for the predicted ratings
user_mean_rmse <- RMSE(test_set$rating, predicted_ratings)

#saving rmse result in tibble
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie and User Mean Effect Model",  
                                     RMSE = user_mean_rmse ))

#display rmse results of all models so far
rmse_results %>% knitr::kable()

## Model 4 ##

#average mean for each genre
genre_mean <- train_set %>%
  left_join(movie_mean, by = "movieId") %>% left_join(user_mean, by = "userId") %>%
  group_by(genres) %>%
  summarize(m_genre = mean(rating - m_all - m_movie - m_user))

#predicted ratings based on mean of everything, per movie, per user, and per genre
predicted_ratings <- test_set %>% 
  left_join(movie_mean, by = "movieId") %>% 
  left_join(user_mean, by = "userId") %>% 
  left_join(genre_mean, by = "genres") %>%
  mutate(pred = m_all + m_movie + m_user + m_genre) %>%
  .$pred

#rmse computation for the predicted ratings
genre_mean_rmse <- RMSE(test_set$rating, predicted_ratings)

#saving rmse to rmse results
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie, User, and Genre Mean Effect Model",  
                                     RMSE = genre_mean_rmse ))

#displaying rmse of all models so far
rmse_results %>% knitr::kable()

## Model 5 ##

#sequence of lambdas to test
lambdas <- seq(0, 10, 0.25)

#create function to check rmse for lambdas
rmses <- sapply(lambdas, function(l){
  
  m_all <- mean(train_set$rating)
  
  #movie mean with regularization
  movie_mean <- train_set %>%
    group_by(movieId) %>%
    summarise(m_movie = sum(rating - m_all)/(n()+l))
  
  #user mean 
  user_mean <- train_set %>%
    left_join(movie_mean, by="movieId") %>%
    group_by(userId) %>%
    summarise(m_user = sum(rating - m_movie - m_all)/(n()))
  
  #genre mean
  genre_mean <- train_set %>%
    left_join(movie_mean, by = "movieId") %>% left_join(user_mean, by = "userId") %>%
    group_by(genres) %>%
    summarise(m_genre = sum(rating - m_movie - m_user - m_all)/(n()))
  
  
  #predicted ratings based on everything, movie mean, user mean, and genre mean with regularization
  predicted_ratings <- test_set %>%
    left_join(movie_mean, by = "movieId") %>% left_join(user_mean, by = "userId") %>% left_join(genre_mean, by = "genres") %>%
    mutate(pred = m_all + m_movie + m_user + m_genre) %>%
    pull(pred)
  
  #return rmse
  return(RMSE(predicted_ratings, test_set$rating))
  
})
#plot for lambdas vs rmses
qplot(lambdas, rmses, color = I("blue"))

#get minimum rmse from all result of all lambdas for mean with regularization
rmse_regularisation <- min(rmses)
rmse_regularisation


#add rmse for regularized movie, user, and genre effects model to rmse_results
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User and Genre Effects with Regularized Movies Model",
                                 RMSE = rmse_regularisation))

#display rmse results for all models
rmse_results %>% knitr::kable()


### Final Validation ###

#gettingg the lambda that was used for the least RMSE
l <- lambdas[which.min(rmses)]
m_all <- mean(edx$rating)

#movie mean with regularization
movie_mean_reg <- edx %>%
  group_by(movieId) %>%
  summarise(m_movie = sum(rating - m_all)/(n()+l))

#user mean
user_mean_reg <- edx %>%
  left_join(movie_mean_reg, by="movieId") %>%
  group_by(userId) %>%
  summarise(m_user = sum(rating - m_movie - m_all)/(n()))

#genre mean
genre_mean_reg <- edx %>%
  left_join(movie_mean_reg, by = "movieId") %>% 
  left_join(user_mean_reg, by = "userId") %>%
  group_by(genres) %>%
  summarise(m_genre = sum(rating - m_movie - m_user - m_all)/(n()))

#predicted ratings based on everything, movie mean, user mean, and genre mean with regularization on movie
predicted_ratings <- validation %>%
  left_join(movie_mean_reg, by = "movieId") %>% 
  left_join(user_mean_reg, by = "userId") %>% 
  left_join(genre_mean_reg, by = "genres") %>%
  mutate(pred = m_all + m_movie + m_user + m_genre) %>%
  pull(pred)

final_RMSE <- RMSE(validation$rating, predicted_ratings)
#save rmse to rmse_results tibble
rmse_results_validation <- tibble(method = "Movie, User and Genre Effects with Regularized Movies Model", RMSE = final_RMSE)

#final rmse
rmse_results_validation %>% knitr::kable()

