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
            NeuralNetwork.showMatrix(NeuralNetwork.getRandomWeightsMatrix(8,3));
            double[] matrix= {1.0d , 2.4d,3.5d, 5.4d};
            double [][] matrix1={{1,2,3},
                    {3,4,5},
                    {6,7,8},
                    {5,7,8}};
            double [] matrix2= {2,-6,8};
            double[] outputMatrix= neuralNetwork.getOutputVector(matrix1,matrix2);
            System.out.println("Show outpus matrix : ");
            NeuralNetwork.showMatrix(outputMatrix);
            double [] matrix3= {0,-6,8};
            System.out.println("Unipolar Sigmoid Function:");
            NeuralNetwork.showMatrix(neuralNetwork.transformWithUnipolarSigmoidFunction(matrix3,1 ));
            System.out.println("Unipolar Step Function:");
            NeuralNetwork.showMatrix(neuralNetwork.transformWithUnipolarStepFunction(matrix3));

            neuralNetwork.getOutputVector(NeuralNetwork.getRandomWeightsMatrix(4,3),matrix );
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
