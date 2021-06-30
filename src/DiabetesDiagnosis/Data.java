package DiabetesDiagnosis;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Data {

    private List<String[]> learningDataSet;
    //8 features
    private List<double[]> learningDataSetFeatures= new ArrayList<>();
    //decision - one signal
    private List<Double> learningDataSetDecisions=new ArrayList<>();



    private List<String[]> multipliedLearningDataSet= new ArrayList<>();
    //8 features
    private List<double[]> multipliedLearningDataSetFeatures= new ArrayList<>();
    //decision - one signal
    private List<Double> multipliedLearningDataSetDecisions=new ArrayList<>();


    private List<String[]> mixedDataSet= new ArrayList<>();
    //8 features
    private List<double[]> mixedLearningDataSetFeatures= new ArrayList<>();
    //decision - one signal
    private List<Double> mixedLearningDataSetDecisions=new ArrayList<>();




    private  List<String[]> testingDataSet;
    //8 features
    private List<double[]> testingDataSetFeatures= new ArrayList<>();
    //decision - one signal
    private List<Double> testingDataSetDecisions= new ArrayList<>();



    Data(List<String[]> learningDataSet, List<String[]>testingDataSet){
        Collections.shuffle(learningDataSet);
        this.learningDataSet=learningDataSet;
        this.testingDataSet=testingDataSet;

        segregateLearningDataSet();
        segregateTestingDataSet();
    }


    public void mixAndMultiplyLearningData(List<String[]> learningDataSet){
        Collections.shuffle(learningDataSet);
        multipliedLearningDataSet.addAll(learningDataSet);

        for (String[] set : multipliedLearningDataSet) {
            int columnNumber = set.length;
            double[] features = new double[columnNumber - 1];
            for (int i = 0; i < columnNumber - 1; i++) {
                features[i] = Double.parseDouble(set[i]);
            }
            multipliedLearningDataSetFeatures.add(features);
        }
        for (String[] set : multipliedLearningDataSet) {
            int columnNumber = set.length;
            double decision = Double.parseDouble(set[columnNumber - 1]);
            multipliedLearningDataSetDecisions.add(decision);
        }

    }


    public void mixAndMultiplyLearningData(){
        Collections.shuffle(learningDataSet);
        multipliedLearningDataSet.addAll(learningDataSet);
        for (String[] set : multipliedLearningDataSet) {
            int columnNumber = set.length;
            double[] features = new double[columnNumber - 1];
            for (int i = 0; i < columnNumber - 1; i++) {
                features[i] = Double.parseDouble(set[i]);
            }
            multipliedLearningDataSetFeatures.add(features);
        }
        for (String[] set : multipliedLearningDataSet) {
            int columnNumber = set.length;
            double decision = Double.parseDouble(set[columnNumber - 1]);
            multipliedLearningDataSetDecisions.add(decision);
        }
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

    public void showExtendedLearningSets(){
        int count=1;

        for (int i =0 ; i <multipliedLearningDataSetFeatures.size() ; i ++ ) {
            System.out.print(count + ".  " );
            count++;
            double []array=multipliedLearningDataSetFeatures.get(i);

            for(int j =0 ; j < array.length ; j++ ){
                System.out.print(array[j] + ", ");
            }
           System.out.println(" ----->  " + multipliedLearningDataSetDecisions.get(i));
            System.out.println();
        }
    }




    public List<double[]> getLearningDataSetFeatures() {
        return learningDataSetFeatures;
    }

    public List<Double> getLearningDataSetDecisions() {
        return learningDataSetDecisions;}

    public List<Double> getMultipliedLearningDataSetDecisions() {
        return multipliedLearningDataSetDecisions;
    }

    public List<double[]> getMultipliedLearningDataSetFeatures() {
        return multipliedLearningDataSetFeatures;
    }

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

    public void resetMultipliedLearningDataSet(){
        multipliedLearningDataSetDecisions.removeAll(multipliedLearningDataSetDecisions);
        multipliedLearningDataSetFeatures.removeAll(multipliedLearningDataSetFeatures);
        multipliedLearningDataSet.removeAll(multipliedLearningDataSet);
    }

    public void mixDataSets(List<String []> dataSet){
        mixedDataSet.removeAll(mixedDataSet);
        mixedLearningDataSetDecisions.removeAll(mixedLearningDataSetDecisions);
        mixedLearningDataSetFeatures.removeAll(mixedLearningDataSetFeatures);

        Collections.shuffle(dataSet);
        mixedDataSet.addAll(dataSet);
        for (String[] set : mixedDataSet) {
            int columnNumber = set.length;
            double[] features = new double[columnNumber - 1];
            for (int i = 0; i < columnNumber - 1; i++) {
                features[i] = Double.parseDouble(set[i]);
            }
            mixedLearningDataSetFeatures.add(features);
            System.out.println("deatues >>>>>>>>>>>>>>>>>>>>>>>>>>"+ mixedLearningDataSetFeatures.size());
        }
        for (String[] set : mixedDataSet) {
            int columnNumber = set.length;
            double decision = Double.parseDouble(set[columnNumber - 1]);
            mixedLearningDataSetDecisions.add(decision);
        }
    }

    public List<Double> getMixedLearningDataSetDecisions() {
        return mixedLearningDataSetDecisions;
    }

    public List<double[]> getMixedLearningDataSetFeatures(){
        return  mixedLearningDataSetFeatures;
    }
}
