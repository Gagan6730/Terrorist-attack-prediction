import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import scala.Function1;
import scala.xml.persistent.Index;

import java.util.concurrent.atomic.AtomicInteger;

public class Prediction {
    public static void main(String args[])
    {
        SparkSession spark= SparkSession.builder()
                .master("local")
                .appName("BDAProject")
//                .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/Admission.Admission_Prediction")
//                .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/Admission.Admission_Prediction")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());


        Dataset<Row> data = spark.read().format("csv").option("sep", ",")
                .option("inferSchema", "true")
                .option("header", "true")
                .load("src/main/resources/target_prediction_data.csv");

        int count=0;

//        Dataset<String> dataset=data.map(new Function<Row, String>() {
//
//            @Override
//            public String call(Row row) throws Exception {
//                int index=row.fieldIndex("targtype1_txt");
//                String val=row.getString(index);
//
//                return val;
//            }
//        });
        Dataset<Row> dataset=data.filter(new FilterFunction<Row>() {
            @Override
            public boolean call(Row row) throws Exception {
                int index=row.fieldIndex("targtype1_txt");
                String val=row.getString(index);
                if( val.equals("Government (Diplomatic)"))
                {return true;}
                else {
                    return false;
                }
            }
        });
        System.out.println(dataset.count());


        LongAccumulator acc=jsc.sc().longAccumulator();
        data.foreach(row->{
            int index=row.fieldIndex("targtype1_txt");
            String val=row.getString(index);
            if(val.equals("Government (General)") || val.equals("Government (Diplomatic)"))
            {
                acc.add(1);
            }
        });
        System.out.println(acc.count());



    }
}
