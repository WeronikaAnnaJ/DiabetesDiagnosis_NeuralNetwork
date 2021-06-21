package DiabetesDiagnosis;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {


    private List<String[]> learningDataSet= new ArrayList<>();
    //8 features
    private List<double[]> learningDataSetFeatures= new ArrayList<>();
    //decision
    private List<Double> learningDataSetDecisions=new ArrayList<>();


    private  List<String[]> testDataSet= new ArrayList<>();
    //8 features
    private List<double[]> testDataSetFeatures= new ArrayList<>();
    //decision
    private List<Double> testDataSetDecisions= new ArrayList<>();


    NeuralNetwork(List<String[]> learningDataSet, List<String[]>testDataSet){

        for (String [] set:learningDataSet) {
            int columnNumber=set.length;
            double [] features= new double[columnNumber-1];

            for(int i=0 ; i< columnNumber-1 ;i ++ ){
                features[i]= Double.parseDouble(set[i]);
            }
            learningDataSetFeatures.add(features);
        }

        for (String [] set:learningDataSet) {
            int columnNumber=set.length;
            double decision=0;
            decision= Double.parseDouble(set[columnNumber-1]);
            learningDataSetDecisions.add(decision);
        }

    }

    public void showLearningFeaturesDecisions(){
        int count=1;

        for (int i =0 ; i <learningDataSetFeatures.size() ; i ++ ) {
            System.out.print(count + ".  " );
            count++;
            double []array=learningDataSetFeatures.get(i);

            for(int j =0 ; j < array.length ; j++ ){
                System.out.print(array[j] + ", ");
            }
            System.out.println(" ----->  " + learningDataSetDecisions.get(i));
            System.out.println();
        }
    }



}
