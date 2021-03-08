# NBA-ShorPredictor
Torch Logistic Regression of NBA shots 

Used data off Kaggle, create a model that shot distance, number of dribbles before the shot, Shot clock time remain and closest defender. 
One data source that I wasn't able to find was the relative angle to the net, which I feel is quite influential. 

Ran training with 80% of the collected data and then validated with the remaining 20%
There is an issue with data.loc when creating the  y tensor in the test section, best guess at the moment is that index data from x:end (x:) results in it being a series rather than a dataframe. FIX was to create a second .csv for the validation data and call it with Pandas.

The model was trained with a variety of learning rates, batch sizes and epochs. The most suitable training parameters led to an prediction accuracy of 60%. This was rather expected as shot to basket is both a function of luck and skill. This can be corrobrated by the fact that approx only 46% of all shots results in points in the NBA suggesting a high influence of randomness and diffculty.

Recommendattions:
- Player shot rating, some players are better than shooting than others, as well as defending player rating.
- Shot location, changes in angle with the backboard influence the rebound of the ball ibto the hoop.
- From research into another project they found movement data being a important predictor. I have included dribbling but that does not necceriloy depict the movement of the player at the instance of the shot being taken. An interesting metric would be velocity of the player.

