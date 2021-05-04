package com.virtualpairprogrammers;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;



public class  CovidLR {

	public static void LRRunner() {
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
				
				LogisticRegression lr=new LogisticRegression();

				ParamGridBuilder pgb = new ParamGridBuilder();
				ParamMap[] paramMap = pgb.addGrid(lr.regParam(), new double[] {0.01,0.1,0.3,0.5,0.7,1})
					.addGrid(lr.elasticNetParam(),new double[] {0,0.5,1})
					.build();
				
				TrainValidationSplit tvs = new TrainValidationSplit();
				tvs.setEstimator(lr)
					.setEvaluator(new RegressionEvaluator().setMetricName("r2"))
					.setEstimatorParamMaps(paramMap)
					.setTrainRatio(0.9);
				
				TrainValidationSplitModel model = tvs.fit(trainingData);
				
				LogisticRegressionModel lrModel = (LogisticRegressionModel)model.bestModel();
				System.out.println("The accuracyScore " + lrModel.summary().accuracy());
				
				System.out.println("coefficients : " + lrModel.coefficients() + " intercept : " + lrModel.intercept());
				System.out.println("reg param : " + lrModel.getRegParam() + " elastic net param : " + lrModel.getElasticNetParam());

	}

}