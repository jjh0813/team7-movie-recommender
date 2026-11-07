Summary
=======

This dataset provides user recommendations, ratings, and elicited “beliefs” (user expected ratings about movies they have not consumed) on MovieLens, a movie recommendation service. 
Over the period of March 1st, 2023 until May 1st, 2024 the second row of the MovieLens homepage asked users to provide information on how they think they would rate movies they have not seen already. We provide the full dataset of which movies were asked about and provide accompanying ratings and recommendation data for users that provided at least one response to the belief elicitation. No demographic information is included. Each user is represented by an id, and no other information is provided.
The data are contained in the files movies.csv, user_rating_history.csv, movie_elicitation_set.csv, user_recommendation_history.csv,  and belief_data.csv. More details about the contents and use of all these files follows.
This and other GroupLens data sets are publicly available for download at http://grouplens.org/datasets/.

UPDATE: On February 8th 2025 we updated the dataset as follows. First, we provided a new set of movies, so that every movie present in the original beliefs dataset is reflected in the movies.csv. Second, we have provided ratings data for an additional set of users present in the dataset, including users that did not originally provide any beliefs data but saw the elicitations.


Usage License
=============

Neither the University of Minnesota nor Northwestern University nor University College London nor any of the researchers involved can guarantee the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:

* The user may not state or imply any endorsement from the University of Minnesota or Northwestern University or University College London or the GroupLens Research Group.
* The user must acknowledge the use of the data set in publications resulting from the use of the data set (see below for citation information).
* The user may redistribute the data set, including transformations, so long as it is distributed under these same license conditions.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from a faculty member of the GroupLens Research Project at the University of Minnesota.
* The executable software scripts are provided "as is" without warranty of any kind, either expressed or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. The entire risk as to the quality and performance of them is with you. Should the program prove defective, you assume the cost of all necessary servicing, repair or correction.

In no event shall the University of Minnesota, Northwestern University, University College London, affiliates or employees of any of these universities be liable to you for any damages arising out of the use or inability to use these programs (including but not limited to loss of data or data being rendered inaccurate).

If you have any further questions or comments, please email guy.aridor@kellogg.northwestern.edu and grouplens-info@umn.edu.

Citation
========

To acknowledge use of the dataset in publications, please cite the following paper:
Guy Aridor and Duarte Goncalves and Ruoyan Kong and Daniel Culver and Joseph A. Konstan. 2024. The MovieLens Beliefs Dataset: Collecting Pre-Choice Data for Online Recommender Systems.

Further Information About GroupLens
===================================

GroupLens is a research group in the Department of Computer Science and Engineering at the University of Minnesota. Since its inception in 1992, GroupLens's research projects have explored a variety of fields including:
* recommender systems
* online communities
* mobile and ubiquitious technologies
* digital libraries
* local geographic information systems

GroupLens Research operates a movie recommender based on collaborative filtering, MovieLens, which is the source of these data. We encourage you to visit http://movielens.org to try it out! If you have exciting ideas for experimental work to conduct on MovieLens, send us an email at grouplens-info@cs.umn.edu - we are always interested in working with external collaborators.

Content and Use of Files
========================

Verifying the Dataset Contents
------------------------------

The following files (with the provided [MD5 checksums](http://en.wikipedia.org/wiki/Md5sum)) should be present in this zip file:


| MD5 | File |
| --- | --- |
|ec2c04e47cc2c2491d1427959a34f329 | belief_data.csv |
|8ce2e6f3414979ca56b3d0cc09670afe | movie_elicitation_set.csv |
|a0ff36b2c08e86316862d30b6d01963d | movies.csv |
|c822b8611e9767d198f812d4a678524c | user_rating_history.csv |
|326b2c0ee4ccb0455b949e4dd7544032 | user_recommendation_history.csv |



We encourage you to verify that the dataset you have on your computer is identical to the ones hosted at [grouplens.org](http://grouplens.org).  This is an important step if you downloaded the dataset from a location other than [grouplens.org](http://grouplens.org), or if you wish to publish research results based on analysis of this dataset.

To verify the dataset (after unzipping):

    # on linux
    md5sum *; cat checksums.txt

    # on OSX
    md5 *; cat checksums.txt

    # windows users can download a tool from Microsoft (or elsewhere) that verifies MD5 checksums

Check that the two lines of output contain the same hash value.


Formatting and Encoding
-----------------------

The dataset files are written as [comma-separated values](http://en.wikipedia.org/wiki/Comma-separated_values) files with a single header row. Columns that contain commas (,) are escaped using double-quotes ("). These files are encoded as UTF-8. If accented characters in movie titles or tag values (e.g. Misérables, Les (1995)) display incorrectly, make sure that any program reading the data, such as a text editor, terminal, or script, is configured for UTF-8.

User Ids
--------

The user ids have been anonymized. User ids are consistent between the different files.

Movie Ids
---------

The movie ids are consistent with those used on the MovieLens web site (e.g., id 1 corresponds to the URL https://movielens.org/movies/1). Movie ids are consistent between the different files.

Movies Data File Structure (movies.csv)
--------------------------------------

Movie information is contained in the file movies.csv. Each line of this file after the header row represents one movie, and has the following format:

movieId,title,genres

Movie titles are entered manually or imported from https://www.themoviedb.org/, and may include the year of release in parentheses. Errors and inconsistencies may exist in these titles.

Genres are a pipe-separated list, and are selected from the following:
* Action
* Adventure
* Animation
* Children's
* Comedy
* Crime
* Documentary
* Drama
* Fantasy
* Film-Noir
* Horror
* Musical
* Mystery
* Romance
* Sci-Fi
* Thriller
* War
* Western
* (no genres listed)

Ratings (user_rating_history.csv)
---------------------------------

All ratings are contained in the file user_rating_history.csv. Each line of this file after the header row represents one rating of one movie by one user, and has the following format:

userId,movieId,rating,timestamp

The lines within this file are ordered first by userId, then, within user, by movieId.

Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).

Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970. 

The structure is identical to the MovieLens core dataset. We provide the full set of ratings for any user that answered at least one of the belief elicitations.

Belief Data (belief_data.csv)
-----------------------------

All elicited beliefs are contained in the file belief_data.csv. Each line of this file after the header row represents one elicitation of one movie by one user, and has the following format:

userId,movieId,isSeen,watchDate,userElicitRating,userPredictRating,userCertainty,tstamp,month_idx,source,systemPredictRating


isSeen takes on the values -1, 0, 1. A value of -1 means that the (userId, movieId) pair was shown and the user did not respond. A value of 0 means that the user marked the movie as not having been watched. A value of 1 means that the user marked the movie as having been watched.

watchDate is null, unless isSeen is equal to 1 and in this case the user provides the approximate watchdate of the movie as specified by the user.

userElicitRating is null, unless isSeen is equal to 1 and in this case the user provides the rating of the movie on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).

userPredictRating is null, unless isSeen is equal to 0 and in this case the user provides their expected rating of the movie on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).

userCertainty is null, unless isSeen is equal to 0 and in this case the user provides how sure they are of their expected rating (userPredictRating) on a 5-point scale, with one-point increments (1.0 - 5.0).

tstamp is a timestamp that represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970. 

month_idx provides an index that indicates the movie was included in the set of movies to be included in the elicitation procedure during this month.

source refers to which of the buckets from the elicitation procedure the movie is chosen from (see Section 2.4 of the focal reference) where (1 - broad sampling, 2 - elicitation with possible recommendation, 3- sample new movies).

systemPredictRating represents the predicted rating of the movieId for the given userId.

Belief Movie Elicitation Set (movie_elicitation_set.csv)
--------------------------------------------------------

The active set of movies from which we elicit beliefs about is provided in movie_elicitation_set.csv. Please see Section 2.3 of the paper for information on how this is constructed.The columns in this dataset are as follows:
movieId,month_idx,source,tstamp

month_idx indexes the months when a movie is part of the set of movies that could be used for belief elicitation. 

tstamp indicates when the set of movies used for belief elicitation was updated.

source indicates which group in the elicitation procedure (documented in Section 2.3 of the paper) the movie was chosen due to (1 - Popularity, 2- Rating, 3 - Popular recent releases, 4 - Trendy releases, 5 - Serendipity).

User Recommendations (user_recommendation_history.csv)
------------------------------------------------------

The set of recommendations observed by the users is provided in user_recommendation_history.csv. The columns in this dataset are as follows:

userId,tstamp,movieId,predictedRating

Each line of this file after the header row represents one recommendation of one movie to one user with the user-specific predicted rating generated by the recommendation system (provided in predictedRating).

We provide the full set of recommendations for any user that answered at least one of the belief elicitations.
