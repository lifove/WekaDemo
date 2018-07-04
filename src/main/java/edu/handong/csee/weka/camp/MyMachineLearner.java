package edu.handong.csee.weka.camp;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveRange;

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
			
			// preprocessing
			AttributeSelection attrSelector = getAttributesSelectionFilterByCfsSubsetEval(trainingData);
			trainingData = selectFeaturesByAttributeSelection(attrSelector, trainingData);
			testData = selectFeaturesByAttributeSelection(attrSelector, testData);
			
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

	/**
	 * Get instances by removing specific attributes
	 * @param instances
	 * @param attributeIndices attribute indices (e.g., 1,3,4) first index is 1
	 * @param invertSelection for invert selection, if true, select attributes with attributeIndices bug if false, remote attributes with attributeIndices
	 * @return new instances with specific attributes
	 */
	public Instances getInstancesByRemovingSpecificAttributes(Instances instances,String attributeIndices,boolean invertSelection){
		Instances newInstances = new Instances(instances);

		Remove remove;

		remove = new Remove();
		remove.setAttributeIndices(attributeIndices);
		remove.setInvertSelection(invertSelection);
		try {
			remove.setInputFormat(newInstances);
			newInstances = Filter.useFilter(newInstances, remove);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}

		return newInstances;
	}

	/**
	 * Get instances by removing specific instances
	 * @param instances
	 * @param instance indices (e.g., 1,3,4) first index is 1
	 * @param option for invert selection
	 * @return selected instances
	 */
	public Instances getInstancesByRemovingSpecificInstances(Instances instances,String instanceIndices,boolean invertSelection){
		Instances newInstances = null;

		RemoveRange instFilter = new RemoveRange();
		instFilter.setInstancesIndices(instanceIndices);
		instFilter.setInvertSelection(invertSelection);

		try {
			instFilter.setInputFormat(instances);
			newInstances = Filter.useFilter(instances, instFilter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newInstances;
	}
	
	/**
	 * Feature selection by GainRatioAttributeEval
	 * @param data
	 * @return newData with selected attributes
	 */
	static public Instances featrueSelectionByGainRatioAttributeEval(Instances data){
		Instances newData = null;

		AttributeSelection filter = new AttributeSelection();  // package weka.filters.supervised.attribute!
		GainRatioAttributeEval eval = new GainRatioAttributeEval();
		Ranker search = new Ranker();
		search.setThreshold(-1.7976931348623157E308);
		search.setNumToSelect(-1);
		filter.setEvaluator(eval);
		filter.setSearch(search);
		try {
			filter.setInputFormat(data);

			// generate new data
			newData = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newData;
	}

	/**
	 * Feature selection by CfsSubsetEval
	 * @param data
	 * @return newData with selected attributes
	 */
	public Instances featrueSelectionByCfsSubsetEval(Instances data){
		Instances newData = null;

		AttributeSelection filter = getAttributesSelectionFilterByCfsSubsetEval(data);  // package weka.filters.supervised.attribute!
		try {
			filter.setInputFormat(data);

			// generate new data
			newData = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newData;
	}
	
	/**
	 * Feature selection filter by CfsSubsetEval
	 * @param data
	 * @return AttributeSelection filter
	 */
	public AttributeSelection getAttributesSelectionFilterByCfsSubsetEval(Instances data){

		AttributeSelection filter = new AttributeSelection();  // package weka.filters.supervised.attribute!
		CfsSubsetEval eval = new CfsSubsetEval();
		BestFirst search = new BestFirst();
		//search.ssetSearchBackwards(false);
		filter.setEvaluator(eval);
		filter.setSearch(search);

		try {
			filter.setInputFormat(data);

			// generate new data
			//newData = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return filter;
	}
	
	public Instances selectFeaturesByAttributeSelection(AttributeSelection filter,Instances data) {
		Instances newData = null;
		
		try {
			filter.setInputFormat(data);

			// generate new data
			newData = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newData;
	}
}
