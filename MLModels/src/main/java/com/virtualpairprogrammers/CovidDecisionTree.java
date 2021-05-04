package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;



public class  CovidDecisionTree {

	public static void CDRunner() {

		Logger.getLogger("org.apache").setLevel(Level.WARN);
		SparkSession spark = SparkSession.builder()
				.appName("Gym Competitors")
				.config("spark.sql.warehouse.dir","file:///d:/tmp/")
				.master("local[*]").getOrCreate();

		

				Dataset<Row> cleanedData=FeatureSelectionAndCleaning.CleanDataSet();
				VectorAssembler vectorAssembler = new VectorAssembler();
				vectorAssembler.setInputCols(new String[] {"high_risk_exposure_occupation","diabetes",
						"chd", "htn", "cancer", "asthma", "copd", "autoimmune_dis", "smoker", "ctab", "labored_respiration","rhonchi", "wheezes", 
						"cough", "sob", "diarrhea", "fatigue", "headache", "loss_of_smell", "loss_of_taste", "runny_nose", "muscle_sore", "sore_throat"});
				vectorAssembler.setOutputCol("features");
				
				Dataset<Row> inputData = vectorAssembler.transform(cleanedData)
						.select("CovidResult","features")
						.withColumnRenamed("CovidResult", "label");
				
				
				Dataset<Row>[] trainingAndHoldoutData = inputData.randomSplit(new double[] {0.8,0.2});
				Dataset<Row> trainingData = trainingAndHoldoutData[0];
				Dataset<Row> holdoutData = trainingAndHoldoutData[1];
				
				DecisionTreeClassifier dtClassifier = new DecisionTreeClassifier();
				dtClassifier.setMaxDepth(3);
				
				DecisionTreeClassificationModel model = dtClassifier.fit(trainingData);
				
				Dataset<Row> predictions = model.transform(holdoutData);
				predictions.show();
				
				System.out.println(model.toDebugString());
				
				MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
				evaluator.setMetricName("accuracy");
				System.out.println("The accuracy of the model is " + evaluator.evaluate(predictions));





	}

}