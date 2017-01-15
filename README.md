# spark-recommendation-engine
###Item-based Collaborative Filtering for Recommending Amazon Products Using Apache Spark

**School:** Brandeis University  
**Course:** COSI 129a: Introduction to Big Data Analysis  
**Professors:** Marcus Verhagen, James Pustejovsky, Pengyu Hong, Liuba Shrira  
**Head TA:** Tuan Do  
**Semester:** Fall 2016  

**Team Members:** Dimokritos Stamatakis, Nikolaos Tsikoudis, William Edgecomb, Tyler Lichten 

**Description**: Our task was to implement in Spark an item-based collaborative filtering recommendation engine given a dataset of Amazon reviews. Given a single product, our engine was to recommend ten similar products. We isolated tuples of userID/productID/rating to train our model, and then used matrix factorization to interpolate the feature vectors for all products. We used the dot product of two vectors as our measure of product similarity. Matrix factorization was computed using Spark's machine learning library. For a more complete description and discussion of the project, please refer to our report PDF. See also the assignment instructions.  
