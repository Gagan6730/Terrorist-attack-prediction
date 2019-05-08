import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.regression.*;
import org.apache.spark.sql.*;


import static com.sun.codemodel.internal.JExpr.lit;


public class GNamePrediction {

    public static void main(String args[]) throws ExceptionInInitializerError
    {

        SparkSession spark= SparkSession.builder()
                .master("local")
                .appName("BDAProject")
//                .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/Admission.Admission_Prediction")
//                .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/Admission.Admission_Prediction")
                .getOrCreate();

//        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
//
//        // Load and analyze data from MongoDB
//        JavaMongoRDD<Document> rdd = MongoSpark.load(jsc);
//
////        StructType str=StructType(StructField("Serial No","IntegerType",true), StructField(GRE Score,IntegerType,true), StructField(TOEFL Score,IntegerType,true), StructField(University Rating,IntegerType,true), StructField(SOP,DoubleType,true), StructField(LOR,DoubleType,true), StructField(CGPA,DoubleType,true), StructField(Research,IntegerType,true), StructField(Chance of Admit,DoubleType,true))
//
////        rdd.foreach(s-> System.out.println(s));
//        System.out.println(rdd.count());
//        Dataset<Row> data=rdd.toDF();

        Dataset<Row> data = spark.read().format("csv").option("sep", ",")
                .option("inferSchema", "true")
                .option("header", "true")
                .load("src/main/resources/target_prediction_data.csv");
//        data.show();
//        Dataset<Row> data=spark.read().load("src/main/resources/Admission_Predict.csv");
//        data.show();
        System.out.println(data.schema());





//        StringIndexer indexer=null;


// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.

        String[] arr={"imonth","iday","country_txt_indexed","region_txt_indexed","city_indexed","success","suicide","attacktype1_txt_indexed","natlty1_txt_indexed","weaptype1_txt_indexed","targtype1_txt_indexed"};
//
//        for(int i=0;i<arr1.length;i++)
//        {
//            StringIndexer indexer1 = new StringIndexer().setInputCol(arr1[i]).setOutputCol(arr1[i] + "_indexed");
//            data=indexer1.fit(data).transform(data);
//
//        }
        StringIndexer indexer1 = new StringIndexer().setInputCol("country_txt").setOutputCol("country_txt" + "_indexed").setHandleInvalid("skip");
        data=indexer1.fit(data).transform(data);
        StringIndexer indexer2 = new StringIndexer().setInputCol("region_txt").setOutputCol("region_txt" + "_indexed").setHandleInvalid("skip");
        data=indexer2.fit(data).transform(data);
        StringIndexer indexer4 = new StringIndexer().setInputCol("city").setOutputCol("city" + "_indexed").setHandleInvalid("skip");
        data=indexer4.fit(data).transform(data);
        StringIndexer indexer5 = new StringIndexer().setInputCol("attacktype1_txt").setOutputCol("attacktype1_txt" + "_indexed").setHandleInvalid("skip");
        data=indexer5.fit(data).transform(data);
        StringIndexer indexer6 = new StringIndexer().setInputCol("natlty1_txt").setOutputCol("natlty1_txt" + "_indexed").setHandleInvalid("skip");
        data=indexer6.fit(data).transform(data);
        StringIndexer indexer7 = new StringIndexer().setInputCol("weaptype1_txt").setOutputCol("weaptype1_txt" + "_indexed").setHandleInvalid("skip");
        data=indexer7.fit(data).transform(data);
        StringIndexer indexer8 = new StringIndexer().setInputCol("gname").setOutputCol("label").setHandleInvalid("skip");
        data=indexer8.fit(data).transform(data);
        StringIndexer indexer9 = new StringIndexer().setInputCol("targtype1_txt").setOutputCol("targtype1_txt" + "_indexed").setHandleInvalid("skip");
        indexer9.setHandleInvalid("skip");
        data=indexer9.fit(data).transform(data);


//        data_indexed.withColumn("country_txt" + "_indexed", (Column) lit(1));
        Dataset<Row> data_indexed=data.select("imonth","iday","country_txt_indexed",
                "region_txt_indexed","city_indexed","success","suicide","attacktype1_txt_indexed",
                "natlty1_txt_indexed","weaptype1_txt_indexed","targtype1_txt_indexed","label");
        data_indexed.show(false);




        VectorAssembler featureIndexer = new VectorAssembler()
                .setInputCols(arr)
                .setOutputCol("features");



        Dataset<Row> out=featureIndexer.transform(data);
        Dataset<Row> final_data=out.select("features","label");
        System.out.println("Final Data:");
        final_data.show(false);


//        f.select("targtype1_txt_indexed","originalCategory").show(100,false);


// Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = final_data.randomSplit(new double[]{0.9, 0.1});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];
        System.out.println("Train Data:");
        trainingData.show(false);
        System.out.println("Test Data:");
        testData.show(false);
        NaiveBayes nb = new NaiveBayes();

// train the model
        NaiveBayesModel model = nb.fit(trainingData);

// Select example rows to display.
        Dataset<Row> predictions = model.transform(testData);
        predictions.select("features","label","prediction").show(false);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test set accuracy = " + accuracy*100);
//
//        String[] array={"a","b","c","d","e","f","g","h","i","j","k"};
//
//        IndexToString converter = new IndexToString()
//                .setInputCol("label")
//                .setOutputCol("originalCategory").setLabels(arr);
//        IndexToString converter2 = new IndexToString()
//                .setInputCol("prediction")
//                .setOutputCol("predCategory").setLabels(arr);
//        Dataset<Row> f=converter.transform(predictions);
//        Dataset<Row> f2=converter2.transform(predictions);
//        System.out.println("Original");
////        f.show();
//        System.out.println("predicted");
//        f2.show();

// compute accuracy on the test set


//        LinearRegression reg=new LinearRegression()
//                .setFeaturesCol("indexedFeatures")
//                .setLabelCol("targtype1_txt_indexed");
//        LinearRegressionModel regressor=reg.fit(trainingData);
//        LinearRegressionSummary pred_result=regressor.evaluate(testData);
//        Dataset<Row> final_output=pred_result.predictions();
//        System.out.println("Prediction Result:");
//        final_output.show(false);
//        Dataset<Row> predictions=final_output.select("prediction");
//        predictions.foreach((ForeachFunction<Row>)row->{
//            double val =(double) row.get(0);
//            System.out.println(val);
//        });
//
//        RegressionEvaluator evaluator = new RegressionEvaluator()
//                .setLabelCol("targtype1_txt_indexed")
//                .setPredictionCol("prediction")
//                .setMetricName("rmse");
//        double rmse = evaluator.evaluate(pred_result.predictions());
//        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);


    }
}
