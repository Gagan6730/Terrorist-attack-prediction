import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeansSummary;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.*;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

public class Clustering {
    public static void main(String args[])
    {
        SparkSession spark= SparkSession.builder()
                .master("local")
                .appName("BDAProject")
                .getOrCreate();


        Dataset<Row> data = spark.read().format("csv").option("sep", ",")
                .option("inferSchema", "true")
                .option("header", "true")
                .load("src/main/resources/target_prediction_data.csv");
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
        StringIndexer indexer8 = new StringIndexer().setInputCol("gname").setOutputCol("gname" + "_indexed").setHandleInvalid("skip");
        data=indexer8.fit(data).transform(data);
        StringIndexer indexer9 = new StringIndexer().setInputCol("targtype1_txt").setOutputCol("targtype1_txt_indexed").setHandleInvalid("skip");
        indexer9.setHandleInvalid("skip");
        data=indexer9.fit(data).transform(data);

//        Dataset<Row> data_indexed=data.select("imonth","iday","country_txt_indexed",
//                "region_txt_indexed","city_indexed","success","suicide","attacktype1_txt_indexed",
//                "natlty1_txt_indexed","weaptype1_txt_indexed","gname_indexed","targtype1_txt_indexed");
//
        String[] arr={"imonth","iday","country_txt_indexed",
                "region_txt_indexed","city_indexed","success","suicide","attacktype1_txt_indexed",
                "natlty1_txt_indexed","weaptype1_txt_indexed","gname_indexed","targtype1_txt_indexed"};
        VectorAssembler featureIndexer = new VectorAssembler()
                .setInputCols(arr)
                .setOutputCol("features");



        Dataset<Row> out=featureIndexer.transform(data);
        Dataset<Row> final_data=out.select("features","imonth","iday","country_txt","region_txt","city","success","suicide","attacktype1_txt","natlty1_txt","weaptype1_txt","gname","targtype1_txt","country_txt_indexed",
                "region_txt_indexed","city_indexed","attacktype1_txt_indexed",
                "natlty1_txt_indexed","weaptype1_txt_indexed","gname_indexed","targtype1_txt_indexed");
        System.out.println("Final Data:");
        final_data.show(false);

        KMeans kmeans = new KMeans().setK(5).setSeed(1L);
        KMeansModel model = kmeans.fit(final_data);





// Make predictions
        Dataset<Row> predictions = model.transform(final_data);
        predictions.show(100,false);

//        Row[] rows=(Row[])predictions.collect();
//        for(Row r:rows)
//        {
//            System.out.println(r);
//        }

// Evaluate clustering by computing Silhouette score
        ClusteringEvaluator evaluator = new ClusteringEvaluator();

        double silhouette = evaluator.evaluate(predictions);
        System.out.println("Silhouette with squared euclidean distance = " + silhouette);

// Shows the result.
        Vector[] centers = model.clusterCenters();
        System.out.println("Cluster Centers: ");
        for (Vector center: centers) {
            System.out.println(center);
        }

        predictions.groupBy("prediction").count().show(false);

        predictions.groupBy("prediction").count().coalesce(1).
                write().
                format("com.databricks.spark.csv").
                option("header", "true").
                save("src/main/resources/clusters.csv");
        predictions.select("imonth","iday","country_txt","region_txt","city","success","suicide","attacktype1_txt","natlty1_txt","weaptype1_txt","gname","targtype1_txt","prediction").coalesce(1).
                write().
                format("com.databricks.spark.csv").
                option("header", "true").
                save("src/main/resources/country_clusters.csv");
        predictions.select("imonth","iday","country_txt_indexed",
                "region_txt_indexed","city_indexed","success","suicide","attacktype1_txt_indexed",
                "natlty1_txt_indexed","weaptype1_txt_indexed","gname_indexed","targtype1_txt_indexed","prediction").coalesce(1).
                write().
                format("com.databricks.spark.csv").
                option("header", "true").
                save("src/main/resources/country_indexed_clusters.csv");
        final_data.show(false);





    }
}
