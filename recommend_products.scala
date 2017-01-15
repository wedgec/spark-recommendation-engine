import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS

// specify the data path by exporting it in an environment variable called DATA_PATH
val dataPath = System.getenv("DATA_PATH")
if (dataPath == null || dataPath.isEmpty){
        println("Data path not specified!")
        System.exit(-1)
}
println("Using data "+ dataPath)
val data = sc.textFile(dataPath)

//filters for only those fields our model will be trained with
var reviews_raw = data.filter( x => (x.startsWith("product/productId:") || x.startsWith("review/userId:") || x.startsWith("review/score:")  ) ).map( y => y.split(" ")(1) )
//assigns unique double in order from 0...n to each element in RDD
var reviews_raw_val_index = reviews_raw.zipWithIndex
//assigns prefixes a,b,c, to productID,userID,score respectively--useful for recovering order
//also puts each in a list for use with reduceByKey
val reviews_raw_fixed = reviews_raw_val_index.map({case(value, id) => (id/3, 
if (id %3 == 0) List("a" + value)
else  if (id % 3 == 1) List("b" + value)
else List("c" + value) ) })
//groups productID,userID,score from same original review together in a list
val reviewsConsolidated = reviews_raw_fixed.reduceByKey( (v1, v2)=>(v1 ::: v2))
//removes prefixes and puts in form: (index, ((productId, userId), score) )
//also we could eliminate k (index) in this command instead of the next one
val reviewsRaw = reviewsConsolidated.map { case (k, v) =>  (k, v.sorted) }.map ({ case (k, v) => (k, ( (v(0).substring(1), v(1).substring(1) ), v(2).substring(1) ) ) } )
//Final review format: ((productID, userID), score)
val reviews = reviewsRaw.map { case (k,v) => v }

//isolate productIds and userIDs and create map data structures convert between String and 
//double representation of IDs 
val productIds = reviews.map( { case ((productId, userId), score) =>productId })
val userIds = reviews.map( { case ((productId, userId), score) =>userId })
//get a map <raw product/user id, unique id>. Used in training
val products = productIds.distinct.zipWithIndex.collectAsMap
val users = userIds.distinct.zipWithIndex.collectAsMap
//reverse the previously created map used to produce the final output
val productsReverse = products.map { case (l , r) => (r, l)}

//create ratings RDD from processed reviews
val ratings = reviews.map( { case ((productId, userId), score) => Rating(users(userId).toInt, products(productId).toInt, score.toDouble) })
//create and train model:
val rank = 10
val numIterations = 10
val model = ALS.train(ratings , rank, numIterations, 0.01)

//RDD with the features in an array for all products
val prodFeatures = model.productFeatures;
//map for getting feature vector from Int ID
val prodFeaturesMap = prodFeatures.collectAsMap;

//function that returns  the dot product of 2 vectors given as arguments.  Our arguments are 
//vectors of features of two products and the result is their similarity
def dotProduct(vector1: Array[Double], vector2: Array[Double]): Double = { (0 to (vector1.size - 1)).toArray.map( Idx => vector1(Idx) * vector2(Idx)).sum }

//products to recommend for, as String IDs
val targetProductsRaw = sc.textFile("/shared3/items-medium.txt")
//loop through all target products
var ok = true
for (targetRaw <- targetProductsRaw.collect() ){
	ok = true
	//error handle if model does not recognize target product
	if (! products.contains(targetRaw)) {
		println(" <product " + targetRaw + " not found >")
		ok = false
	}
	if (ok) {
		//targetID of type long
		val targetID = products(targetRaw)
		//gets all (similarity, productID) pairs, where similarity is measured by 
		//dot product of target product feature vector with other product feature vector
		val similarities = prodFeatures.filter({ case (prodID, featureVec) => (prodID != targetID.toInt) } ).map( { case (prodID, featureVec) => ( dotProduct(prodFeaturesMap(targetID.toInt), featureVec), prodID.toInt ) } )
		//gets product IDs with 10 greatest dot products--makes use of default tuple ordering, which orders by key
		val bestTen = similarities.takeOrdered(10) (Ordering[(Double, Int)].reverse) .map({ case (similarity, prodID) => prodID } )
		//loop writes IDs of top ten recommended products  to file
		var i=0
		print(targetRaw + " ")
		for (productID <- bestTen) {
			val rawID = productsReverse(productID)
			print(""+rawID)
			if (i < 9) print(",")
			i=i+1
		}
		println()
	}
}


