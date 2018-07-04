package edu.handong.csee.weka.camp;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class MyMachineLearner {

	public static void main(String[] args) {
		MyMachineLearner learner = new MyMachineLearner();
		learner.run(args);
	}

	private void run(String[] args) {
		
		// (1) Read Training, Test arff files
		System.out.println(args[0]);
		System.out.println(args[1]);
		String arffForTraining = args[0];
		String arffForTest = args[1];
		try {
			BufferedReader reader = new BufferedReader(new FileReader(arffForTraining));
			Instances trainingData = new Instances(reader);
			trainingData.setClassIndex(trainingData.numAttributes()-1);
			reader.close();
			
			reader = new BufferedReader(new FileReader(arffForTest));
			Instances testData = new Instances(reader);
			testData.setClassIndex(trainingData.numAttributes()-1);
			reader.close();
			
			// (2) Build a learner
			Classifier cls = new J48();
			cls.buildClassifier(trainingData);
			
			// (3) Test
			Evaluation eval = new Evaluation(trainingData);
			eval.evaluateModel(cls, testData);
			
			// (4) Show prediction results
			int i=0;
			for(Prediction prediction:eval.predictions()) {
				String predictedValue = getClassValue(trainingData,prediction.predicted());
				System.out.println("Instance " + (++i) + " " + predictedValue);
			}
			
			System.out.println("\n\n\n=====Test summary in case the test set has labels");
			System.out.println(eval.toSummaryString());
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	String getClassValue(Instances instances, double index) {
		return instances.attribute(instances.classIndex()).value((int) index);
	}
}
