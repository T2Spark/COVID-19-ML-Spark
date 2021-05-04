package com.virtualpairprogrammers;

import java.util.ArrayList;
import java.util.Collections;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
 

public class FeatureSelectionAndCleaning {
	
public static void ComputeCorrelation(Dataset<Row> cleanedData) {
	
	ArrayList<Double> corrValues = new ArrayList<Double>();
	System.out.println("----------------------Correlation For Dataset combined------------------");
	for (String col : cleanedData.columns() ) {
		double correlation_value = cleanedData.stat().corr("CovidResult", col);
	
		System.out.println("The correlation between the CovidResult and " + col + " is " +correlation_value );
		corrValues.add(correlation_value);
		
	
		
	}
	Collections.sort(corrValues);
	System.out.println("After sorting : "+corrValues);
		
	}

	public static Dataset<Row> CleanDataSet()
	{
	
		System.setProperty("hadoop.home.dir","c:/hadoop");
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		SparkSession spark = SparkSession.builder()
				.appName("Gym Competitors")
				.config("spark.sql.warehouse.dir","file:///d:/tmp/")
				.master("local[*]").getOrCreate();
		
	
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true) 
				//.csv("src\\main\\resources\\*.csv");
				.csv("src\\main\\resources\\04-07_carbonhealth_and_braidhealth.csv");
		
	
		csvData.createOrReplaceTempView("table3");
		csvData = spark.sql("select * from table3 where covid19_test_results is not null");
		csvData.createOrReplaceTempView("table4");
		csvData = spark.sql("select * from table4 where covid19_test_results='Negative' OR covid19_test_results= 'Positive'");
	
		StringIndexer covidIndexer = new StringIndexer();
		
		csvData=covidIndexer.setInputCol("covid19_test_results")
				.setOutputCol("CovidResult").fit(csvData).transform(csvData); 
	
		
		csvData.createOrReplaceTempView("table5");
		csvData = spark.sql("select * from table5");
		
		csvData = csvData.drop("batch_date","test_name","swab_type","age","cough_severity","sob_severity","cxr_findings","cxr_impression","cxr_label","cxr_link","er_referral");//drop the cols
		csvData.createOrReplaceTempView("table6");
		csvData = spark.sql("select * from table6");
		
	
		Dataset<Row> csvDataWithFeatures = spark.sql("select CovidResult,int(high_risk_exposure_occupation),int(high_risk_interactions),int(cancer),int(copd),int(ctab),int(labored_respiration),int(fever),int(headache),int(loss_of_smell),int(loss_of_taste),int(runny_nose),int(muscle_sore) from table6");		
		csvDataWithFeatures.createOrReplaceTempView("table7");
			csvDataWithFeatures = spark.sql("select * from table7");
			csvDataWithFeatures.show();	
			Dataset<Row> csvData_modified2 = spark.sql("select CovidResult,high_risk_exposure_occupation,high_risk_interactions,cancer,copd,ctab,labored_respiration,fever,headache,loss_of_smell,loss_of_taste,runny_nose,muscle_sore from table7 where high_risk_exposure_occupation is not null and high_risk_interactions is not null and cancer is not null and copd is not null "
				+ " and ctab is not null and labored_respiration is not null and fever is not null  and headache is not null  and loss_of_smell is not null  and loss_of_taste is not null  "
				+ "and runny_nose is not null  and muscle_sore is not null");
	
			csvData_modified2.show();	
		return csvData_modified2;
		
	}
	
}

