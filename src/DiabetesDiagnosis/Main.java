package DiabetesDiagnosis;

import java.io.IOException;

public class Main {

    public static void main(String[] args) {
	// write your code here
        try {
           ReadFile file =  new ReadFile();
            file.readCsv();
            file.showAllData();
            file.segregateData();

            NeuralNetwork neuralNetwork= new NeuralNetwork(file.getLearningDataSet(), file.getTestDataSet());
            neuralNetwork.showLearningFeaturesDecisions();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
