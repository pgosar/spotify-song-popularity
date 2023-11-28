# spotify-song-popularity
Predicting how popular a song on spotify is based off several characteristics such as danceability, chart position, energy etc.

The python files contains comparisons between various ML techniques.

Built by Pranay Gosar, Adam Nguyen, Ben Gordon, Nicholas Orlowsky

## Outcome & Results
In our data exploration, we found that there wasn't much of a clustering tendency in the dataset. This led us to conclude that most songs are fairly unique, resulting in a somewhat sparse dataset. The sparseness of the data contributed to KNN and Linear Regression performing poorly as these are not optimal conditions for these models.

We also found that certain features had very little correlation with each other, and thus were inappropriate for PCA. The only model where PCA caused an accuracy increase was K nearest neighbors; for all other models it significantly reduced the accuracy and thus we decided not to do it.

One issue with the dataset is that some features have a very low correlation with the number of streams, while others were almost proxies for the number of streams. Features about the song itself such as 'Danceability' and the key it was written in added a lot of dimensionality while not being useful for prediction. What works best at predicting streams is the popularity of the artists and proxy features for song popularity such as being on charts and being in playlists. This aligns with historical trends in music where artists usually gain listeners after becoming popular, and these listeners continue listening even when the artist's musical style evolves. The importance of artist popularity can also be seen whenever a popular band does a cover of a less-popular band's song and receives more streams than the original less-popular band.

To prove this, we tried running our models with different subsets of the features. Running them without the popularity metrics and only the song characteristics led to significant reductions in accuracy relative to when all features were present. Further some models worked about as well with only the popularity metrics (the various decision tree ensemble classifiers + polynomial regression) as they did with all features present.

The best models were generally decision tree ensemble classifiers. The best one was GradientBoostingRegressor, whose mean absolute error was about 28% of the standard deviation of streams when all features were present. One reason why decision tree regressors might work better than linear regression is the skewed distribution of the y/streams values. The right combination of circumstances can lead to songs being breakout hits from the rest of the data set.
