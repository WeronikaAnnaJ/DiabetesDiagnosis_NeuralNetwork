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

            //output values
            double [][] weightsW={
                    {1,0,3},
                    {-3,2,8},
                    {4, -3, -2},
                    {7,-2,-1}};
            double [][] weightsM={
                    {2,4,1,3},
                    {3,-5,3,2},
                    {4,3,-4,-3}
            };
            double inputX [] ={-3,4,1};
            double [] net1= neuralNetwork.getOutputVector(weightsW, inputX);
            NeuralNetwork.showMatrix(net1);

            double [] Y1= neuralNetwork.transformWithUnipolarSigmoidFunction(net1, 1);
            NeuralNetwork.showMatrix(Y1);

            double [] net2 = neuralNetwork.getOutputVector(weightsM,Y1);
            NeuralNetwork.showMatrix(net2);

            double [] Y2 = neuralNetwork.transformWithUnipolarSigmoidFunction(net2,1);
            NeuralNetwork.showMatrix(Y2);




            //Backpropagation
            System.out.println("Backpropagation");


            double [][] weightsW1={
                    {1, -2},
                    {-3, 1},
                    {1, -1}};
            double [][] weightsM2={
                    {-1,-1,-1},
                    {1,1,1}
            };
            double inputX_1 [] ={-1,1};
            double [] net1_1= neuralNetwork.getOutputVector(weightsW1, inputX_1);
            NeuralNetwork.showMatrix(net1_1);

            double [] Y1_1= neuralNetwork.transformWithUnipolarSigmoidFunction(net1_1, 1);
            NeuralNetwork.showMatrix(Y1_1);

            double [] net2_1 = neuralNetwork.getOutputVector(weightsM2,Y1_1);
            NeuralNetwork.showMatrix(net2_1);

            double [] Y2_1 = neuralNetwork.transformWithUnipolarSigmoidFunction(net2_1,1);
            NeuralNetwork.showMatrix(Y2_1);

            //error neurons from output layer
            double [] outpuLayerError= new double[weightsM2.length];
            double []expectedValuesOutputLayer={1, 0};

            for(int i =0 ; i < outpuLayerError.length; i++){
                outpuLayerError[i]=neuralNetwork.determineErrorFor0utputNeuron(expectedValuesOutputLayer[i],Y2_1[i], 1, Y2_1[i] );
            }
            NeuralNetwork.showMatrix(outpuLayerError);


            //error neurons from hidden layer
            double [] hiddenLayerError= new double[weightsW1.length];

            for (int i =0 ; i < hiddenLayerError.length ; i ++){

                double [] weightsForNextNeuron= new double[weightsM2.length];

                for(int j=0 ; j< weightsM2.length; j++){
                    weightsForNextNeuron[j]=weightsM2[j][i]; // ?? ERROR
                    System.out.println("Error" + i +"-> weightsForNextNeuron[" + j +"] : " + weightsForNextNeuron[j]);
                }
                hiddenLayerError[i]=neuralNetwork.determineErrorForHiddenNeuron(outpuLayerError,weightsForNextNeuron,Y1_1[i], 1);
            }
            NeuralNetwork.showMatrix(hiddenLayerError);

            double [][] newWeightsForHiddenNeurons= new double[weightsW1.length][];
            for(int i=0 ; i< newWeightsForHiddenNeurons.length ; i ++){
                double [] oldWeightsForHiddenNeuron= new double[weightsW1[i].length];
                for(int j=0; j<weightsW1[i].length; j++){
                    oldWeightsForHiddenNeuron[j]=weightsW1[i][j];
                }
                System.out.println("old weights for neuron : ");
                NeuralNetwork.showMatrix(oldWeightsForHiddenNeuron);

                double[] newWeightsForHiddenNeuron= neuralNetwork.determineWeightsForNeuron(oldWeightsForHiddenNeuron,1,hiddenLayerError[i],inputX_1);
                System.out.println("new weights for neuron : ");
                NeuralNetwork.showMatrix(newWeightsForHiddenNeuron);

                newWeightsForHiddenNeurons[i]=newWeightsForHiddenNeuron;
            }
            System.out.println("NEW WEIGHTS FOR NEURONS IN HIDDEN LAYER");
            NeuralNetwork.showMatrix(newWeightsForHiddenNeurons);


            double [] biasHiddenLayer={0,0,0};
            double[] biasOutputLayer={0,0};

            double[] newbiasHiddenLayer= new double[biasHiddenLayer.length];
            for (int i=0 ; i < newbiasHiddenLayer.length ; i ++){
                newbiasHiddenLayer[i]=neuralNetwork.determineNewBiasForNeuron(biasHiddenLayer[i],1,hiddenLayerError[i]);
            }
            NeuralNetwork.showMatrix(newbiasHiddenLayer);



//tested <

            System.out.println("OUTPUT LAYER");


            double [][] newWeightsForOutputNeurons= new double[weightsM2.length][];
            for(int i=0 ; i< newWeightsForOutputNeurons.length ; i ++){
                double [] oldWeightsForOutputNeuron= new double[weightsM2[i].length];
                for(int j=0; j<weightsM2[i].length; j++){
                    oldWeightsForOutputNeuron[j]=weightsM2[i][j];
                }
                System.out.println("old weights for neuron : ");
                NeuralNetwork.showMatrix(oldWeightsForOutputNeuron);

                double[] newWeightsForOutputNeuron= neuralNetwork.determineWeightsForNeuron(oldWeightsForOutputNeuron,1,outpuLayerError[i],Y1_1);
                System.out.println("new weights for neuron : ");
                NeuralNetwork.showMatrix(newWeightsForOutputNeuron);

                newWeightsForOutputNeurons[i]=newWeightsForOutputNeuron;
            }
            System.out.println("NEW WEIGHTS FOR NEURONS IN OUTPUT LAYER");
            NeuralNetwork.showMatrix(newWeightsForOutputNeurons);


            double[] newBiasOutputLayer= new double[biasOutputLayer.length];
            for (int i=0 ; i < newBiasOutputLayer.length ; i ++){
                newBiasOutputLayer[i]=neuralNetwork.determineNewBiasForNeuron(biasOutputLayer[i],1,outpuLayerError[i]);
            }
            NeuralNetwork.showMatrix(newBiasOutputLayer);







        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
