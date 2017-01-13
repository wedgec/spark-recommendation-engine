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
//we could get rid of k right here--would require modification going forward too
//
val reviewsRaw = reviewsConsolidated.map { case (k, v) =>  (k, v.sorted) }.map ({ case (k, v) => (k, ( (v(0).substring(1), v(1).substring(1) ), v(2).substring(1) ) ) } )
// some items in the targeted items have ‘unkown’ user, so we cannot ignore them!
//val reviews_clean = reviews.filter(_._2._1._2!=("unknown"))
//With singles removed:
//index, ((productID, userID), score)
//val reviews_singles_removed = reviews.map { case (k, v) => (v._1._1, //List(v))}.reduceByKey((v1, v2) => v1 ::: v2).filter( { case (k, v) => (v.length > 1) } ).flatMap ( { //case (k,v) => v } )
val reviews = reviewsRaw.map { case (k,v) => v }
//Next two collect the product and user ids 
val productIds = reviews.map( { case ((productId, userId), score) =>productId })
val userIds = reviews.map( { case ((productId, userId), score) =>userId })
//Get a map <raw product/user id, unique id>. Used in training
val products = productIds.distinct.zipWithIndex.collectAsMap
val users = userIds.distinct.zipWithIndex.collectAsMap
//Reverse the previously created map used to produce the final output
val productsReverse = products.map { case (l , r) => (r, l)}
//create the rating used in training
val ratings = reviews.map( { case ((productId, userId), score) => Rating(users(userId).toInt, products(productId).toInt, score.toDouble) })
//create model:
val rank = 10
val numIterations = 10
val model = ALS.train(ratings , rank, numIterations, 0.01)

//RDD with the features in an array for all products
val prodFeatures = model.productFeatures;
val prodFeaturesMap = prodFeatures.collectAsMap;

//function that returns  the dot product of 2 vectors given as arguments.  Our arguments are 
//vectors of features of two products and the result is their similarity
def dotProduct(vector1: Array[Double], vector2: Array[Double]): Double = { (0 to (vector1.size - 1)).toArray.map( Idx => vector1(Idx) * vector2(Idx)).sum }
// Then do some operation to isolate the 100 target vectors (must preserve order)
//schematically:
val targetProductsRaw = sc.textFile("/shared3/items-medium.txt")
//val targetProducts = targetProductsRaw.map( prodID => products(prodID) )
//do something like this for each value in targets (for loop over targets)
var ok = true
for (targetRaw <- targetProductsRaw.collect() ){
	ok = true
	//targetID type long
	if (! products.contains(targetRaw)) {
		println(" <product " + targetRaw + " not found >")
		ok = false
	}
	if (ok) {
		val targetID = products(targetRaw)
		val similarities = prodFeatures.filter({ case (prodID, featureVec) => (prodID != targetID.toInt) } ).map( { case (prodID, featureVec) => ( dotProduct(prodFeaturesMap(targetID.toInt), featureVec), prodID.toInt ) } )
		//more efficient, makes similarity the first value in tuple and sorts by it using reverse of 
		//default sort
		val bestTen = similarities.takeOrdered(10) (Ordering[(Double, Int)].reverse) .map({ case (similarity, prodID) => prodID } )
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


