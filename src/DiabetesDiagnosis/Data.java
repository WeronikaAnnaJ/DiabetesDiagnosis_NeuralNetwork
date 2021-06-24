package DiabetesDiagnosis;

import java.util.ArrayList;
import java.util.List;

public class Data {

    private List<String[]> learningDataSet;
    //8 features
    private List<double[]> learningDataSetFeatures= new ArrayList<>();
    //decision - one signal
    private List<Double> learningDataSetDecisions=new ArrayList<>();


    private  List<String[]> testingDataSet;
    //8 features
    private List<double[]> testingDataSetFeatures= new ArrayList<>();
    //decision - one signal
    private List<Double> testingDataSetDecisions= new ArrayList<>();



    Data(List<String[]> learningDataSet, List<String[]>testingDataSet){
        this.learningDataSet=learningDataSet;
        this.testingDataSet=testingDataSet;

        segregateLearningDataSet();
        segregateTestingDataSet();
    }



    private void segregateLearningDataSet(){
        for (String [] set : learningDataSet) {
            int columnNumber= set.length;
            double [] features= new double[columnNumber-1];
            for(int i=0; i< columnNumber-1; i ++ ){
                features[i]= Double.parseDouble(set[i]);
            }
            learningDataSetFeatures.add(features);
        }
        for (String [] set : learningDataSet) {
            int columnNumber= set.length;
            double decision= Double.parseDouble(set[columnNumber-1]);
            learningDataSetDecisions.add(decision);
        }
    }



    private void segregateTestingDataSet() {
        for (String[] set : testingDataSet) {
            int columnNumber = set.length;
            double[] features = new double[columnNumber - 1];
            for (int i = 0; i < columnNumber - 1; i++) {
                features[i] = Double.parseDouble(set[i]);
            }
            testingDataSetFeatures.add(features);
        }
        for (String[] set : testingDataSet) {
            int columnNumber = set.length;
            double decision = Double.parseDouble(set[columnNumber - 1]);
            testingDataSetDecisions.add(decision);
        }
    }



    public void showLearningSets(){
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




    public List<double[]> getLearningDataSetFeatures() {
        return learningDataSetFeatures;
    }

    public List<Double> getLearningDataSetDecisions() {
        return learningDataSetDecisions;}

    public List<double[]> getTestingDataSetFeatures() {
        return testingDataSetFeatures;
    }

    public void setTestingDataSetDecisions(List<Double> testDataSetDecisions) {
        this.testingDataSetDecisions = testDataSetDecisions;
    }

    public List<String[]> getLearningDataSet() {
        return learningDataSet;
    }

    public List<String[]> getTestingDataSet() {
        return testingDataSet;
    }



}
