package DiabetesDiagnosis;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        // write your code here
        try {
            ReadFile file = new ReadFile();
            file.readCsv();
            file.showAllData();
            file.segregateData();

            Data data = new Data(file.getLearningDataSet(), file.getTestingDataSet());
            data.showLearningSets();

            NeuralNetwork neuralNetwork = new NeuralNetwork(data.getLearningDataSetFeatures(), data.getLearningDataSetDecisions());
            double[] biasHiddenLayer = {0, 0, 0, 0, 0, 0, 0, 0};
            double[] biasOutputLayer = {0};
            neuralNetwork.setBiasHiddenLayer(biasHiddenLayer);
            neuralNetwork.setBiasOutputLayer(biasOutputLayer);
            neuralNetwork.setLambda(1);
            neuralNetwork.setLearningRate(0.01);
            neuralNetwork.setWeightsHiddenLayer(NeuralNetwork.getRandomWeightsMatrix(8, 8));
            neuralNetwork.setWeightsOutputLayer(NeuralNetwork.getRandomWeightsMatrix(8, 1));

            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?


                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer, biasHiddenLayer, biasOutputLayer);


                System.out.println("\n------- > neural network net1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet1());
                System.out.println("\n------- > neural network Y1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY1());
                System.out.println("\n------- > neural network net2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet2());
                System.out.println("\n------- > neural network Y2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY2());

                //errore
                neuralNetwork.calculateErrorsForLayers(neuralNetwork.getWeightsForHiddenLayer(), neuralNetwork.getWeightsForOutputLayer(), expectedValuesOutputLayer);
                System.out.println("\n------- > error for hidden layer:");
                NeuralNetwork.showMatrix(neuralNetwork.getHiddenLayerError());
                System.out.println("\n------- > error for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getOutpuLayerError());

                System.out.println("\n------- > new weights for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForHiddenLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasHiddenLayer());

                System.out.println("\n------- > new weights for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForOutputLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasOutputLayer());


                System.out.println("\n------- > error before :" + neuralNetwork.getMeanSquaredErrorBeforeEpoch());
                System.out.println("\n------- > error after :" + neuralNetwork.getMeanSquaredErrorAfterEpoch());

            }
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?


                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer, biasHiddenLayer, biasOutputLayer);


                System.out.println("\n------- > neural network net1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet1());
                System.out.println("\n------- > neural network Y1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY1());
                System.out.println("\n------- > neural network net2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet2());
                System.out.println("\n------- > neural network Y2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY2());

                //errore
                neuralNetwork.calculateErrorsForLayers(neuralNetwork.getWeightsForHiddenLayer(), neuralNetwork.getWeightsForOutputLayer(), expectedValuesOutputLayer);
                System.out.println("\n------- > error for hidden layer:");
                NeuralNetwork.showMatrix(neuralNetwork.getHiddenLayerError());
                System.out.println("\n------- > error for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getOutpuLayerError());

                System.out.println("\n------- > new weights for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForHiddenLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasHiddenLayer());

                System.out.println("\n------- > new weights for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForOutputLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasOutputLayer());


                System.out.println("\n------- > error before :" + neuralNetwork.getMeanSquaredErrorBeforeEpoch());
                System.out.println("\n------- > error after :" + neuralNetwork.getMeanSquaredErrorAfterEpoch());

            }


            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?


                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer, biasHiddenLayer, biasOutputLayer);


                System.out.println("\n------- > neural network net1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet1());
                System.out.println("\n------- > neural network Y1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY1());
                System.out.println("\n------- > neural network net2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet2());
                System.out.println("\n------- > neural network Y2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY2());

                //errore
                neuralNetwork.calculateErrorsForLayers(neuralNetwork.getWeightsForHiddenLayer(), neuralNetwork.getWeightsForOutputLayer(), expectedValuesOutputLayer);
                System.out.println("\n------- > error for hidden layer:");
                NeuralNetwork.showMatrix(neuralNetwork.getHiddenLayerError());
                System.out.println("\n------- > error for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getOutpuLayerError());

                System.out.println("\n------- > new weights for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForHiddenLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasHiddenLayer());

                System.out.println("\n------- > new weights for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForOutputLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasOutputLayer());


                System.out.println("\n------- > error before :" + neuralNetwork.getMeanSquaredErrorBeforeEpoch());
                System.out.println("\n------- > error after :" + neuralNetwork.getMeanSquaredErrorAfterEpoch());

            }

            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?


                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer, biasHiddenLayer, biasOutputLayer);


                System.out.println("\n------- > neural network net1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet1());
                System.out.println("\n------- > neural network Y1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY1());
                System.out.println("\n------- > neural network net2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet2());
                System.out.println("\n------- > neural network Y2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY2());

                //errore
                neuralNetwork.calculateErrorsForLayers(neuralNetwork.getWeightsForHiddenLayer(), neuralNetwork.getWeightsForOutputLayer(), expectedValuesOutputLayer);
                System.out.println("\n------- > error for hidden layer:");
                NeuralNetwork.showMatrix(neuralNetwork.getHiddenLayerError());
                System.out.println("\n------- > error for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getOutpuLayerError());

                System.out.println("\n------- > new weights for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForHiddenLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasHiddenLayer());

                System.out.println("\n------- > new weights for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForOutputLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasOutputLayer());


                System.out.println("\n------- > error before :" + neuralNetwork.getMeanSquaredErrorBeforeEpoch());
                System.out.println("\n------- > error after :" + neuralNetwork.getMeanSquaredErrorAfterEpoch());

            }

            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?


                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer, biasHiddenLayer, biasOutputLayer);


                System.out.println("\n------- > neural network net1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet1());
                System.out.println("\n------- > neural network Y1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY1());
                System.out.println("\n------- > neural network net2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet2());
                System.out.println("\n------- > neural network Y2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY2());

                //errore
                neuralNetwork.calculateErrorsForLayers(neuralNetwork.getWeightsForHiddenLayer(), neuralNetwork.getWeightsForOutputLayer(), expectedValuesOutputLayer);
                System.out.println("\n------- > error for hidden layer:");
                NeuralNetwork.showMatrix(neuralNetwork.getHiddenLayerError());
                System.out.println("\n------- > error for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getOutpuLayerError());

                System.out.println("\n------- > new weights for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForHiddenLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasHiddenLayer());

                System.out.println("\n------- > new weights for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForOutputLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasOutputLayer());


                System.out.println("\n------- > error before :" + neuralNetwork.getMeanSquaredErrorBeforeEpoch());
                System.out.println("\n------- > error after :" + neuralNetwork.getMeanSquaredErrorAfterEpoch());

            }

            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?


                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer, biasHiddenLayer, biasOutputLayer);


                System.out.println("\n------- > neural network net1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet1());
                System.out.println("\n------- > neural network Y1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY1());
                System.out.println("\n------- > neural network net2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet2());
                System.out.println("\n------- > neural network Y2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY2());

                //errore
                neuralNetwork.calculateErrorsForLayers(neuralNetwork.getWeightsForHiddenLayer(), neuralNetwork.getWeightsForOutputLayer(), expectedValuesOutputLayer);
                System.out.println("\n------- > error for hidden layer:");
                NeuralNetwork.showMatrix(neuralNetwork.getHiddenLayerError());
                System.out.println("\n------- > error for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getOutpuLayerError());

                System.out.println("\n------- > new weights for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForHiddenLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasHiddenLayer());

                System.out.println("\n------- > new weights for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForOutputLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasOutputLayer());


                System.out.println("\n------- > error before :" + neuralNetwork.getMeanSquaredErrorBeforeEpoch());
                System.out.println("\n------- > error after :" + neuralNetwork.getMeanSquaredErrorAfterEpoch());

            }

            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?


                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer, biasHiddenLayer, biasOutputLayer);


                System.out.println("\n------- > neural network net1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet1());
                System.out.println("\n------- > neural network Y1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY1());
                System.out.println("\n------- > neural network net2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet2());
                System.out.println("\n------- > neural network Y2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY2());

                //errore
                neuralNetwork.calculateErrorsForLayers(neuralNetwork.getWeightsForHiddenLayer(), neuralNetwork.getWeightsForOutputLayer(), expectedValuesOutputLayer);
                System.out.println("\n------- > error for hidden layer:");
                NeuralNetwork.showMatrix(neuralNetwork.getHiddenLayerError());
                System.out.println("\n------- > error for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getOutpuLayerError());

                System.out.println("\n------- > new weights for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForHiddenLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasHiddenLayer());

                System.out.println("\n------- > new weights for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForOutputLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasOutputLayer());


                System.out.println("\n------- > error before :" + neuralNetwork.getMeanSquaredErrorBeforeEpoch());
                System.out.println("\n------- > error after :" + neuralNetwork.getMeanSquaredErrorAfterEpoch());

            }

            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?


                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer, biasHiddenLayer, biasOutputLayer);


                System.out.println("\n------- > neural network net1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet1());
                System.out.println("\n------- > neural network Y1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY1());
                System.out.println("\n------- > neural network net2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet2());
                System.out.println("\n------- > neural network Y2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY2());

                //errore
                neuralNetwork.calculateErrorsForLayers(neuralNetwork.getWeightsForHiddenLayer(), neuralNetwork.getWeightsForOutputLayer(), expectedValuesOutputLayer);
                System.out.println("\n------- > error for hidden layer:");
                NeuralNetwork.showMatrix(neuralNetwork.getHiddenLayerError());
                System.out.println("\n------- > error for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getOutpuLayerError());

                System.out.println("\n------- > new weights for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForHiddenLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasHiddenLayer());

                System.out.println("\n------- > new weights for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForOutputLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasOutputLayer());


                System.out.println("\n------- > error before :" + neuralNetwork.getMeanSquaredErrorBeforeEpoch());
                System.out.println("\n------- > error after :" + neuralNetwork.getMeanSquaredErrorAfterEpoch());

            }

            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?


                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer, biasHiddenLayer, biasOutputLayer);


                System.out.println("\n------- > neural network net1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet1());
                System.out.println("\n------- > neural network Y1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY1());
                System.out.println("\n------- > neural network net2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet2());
                System.out.println("\n------- > neural network Y2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY2());

                //errore
                neuralNetwork.calculateErrorsForLayers(neuralNetwork.getWeightsForHiddenLayer(), neuralNetwork.getWeightsForOutputLayer(), expectedValuesOutputLayer);
                System.out.println("\n------- > error for hidden layer:");
                NeuralNetwork.showMatrix(neuralNetwork.getHiddenLayerError());
                System.out.println("\n------- > error for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getOutpuLayerError());

                System.out.println("\n------- > new weights for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForHiddenLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasHiddenLayer());

                System.out.println("\n------- > new weights for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForOutputLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasOutputLayer());


                System.out.println("\n------- > error before :" + neuralNetwork.getMeanSquaredErrorBeforeEpoch());
                System.out.println("\n------- > error after :" + neuralNetwork.getMeanSquaredErrorAfterEpoch());

            }

            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?


                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer, biasHiddenLayer, biasOutputLayer);


                System.out.println("\n------- > neural network net1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet1());
                System.out.println("\n------- > neural network Y1 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY1());
                System.out.println("\n------- > neural network net2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getNet2());
                System.out.println("\n------- > neural network Y2 :");
                NeuralNetwork.showMatrix(neuralNetwork.getY2());

                //errore
                neuralNetwork.calculateErrorsForLayers(neuralNetwork.getWeightsForHiddenLayer(), neuralNetwork.getWeightsForOutputLayer(), expectedValuesOutputLayer);
                System.out.println("\n------- > error for hidden layer:");
                NeuralNetwork.showMatrix(neuralNetwork.getHiddenLayerError());
                System.out.println("\n------- > error for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getOutpuLayerError());

                System.out.println("\n------- > new weights for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForHiddenLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasHiddenLayer());

                System.out.println("\n------- > new weights for output layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getWeightsForOutputLayer());

                System.out.println("\n------- > new bias for hidden layer :");
                NeuralNetwork.showMatrix(neuralNetwork.getBiasOutputLayer());


                System.out.println("\n------- > error before :" + neuralNetwork.getMeanSquaredErrorBeforeEpoch());
                System.out.println("\n------- > error after :" + neuralNetwork.getMeanSquaredErrorAfterEpoch());

            }


            System.out.println(" ooooooooooooooooooooooo" + neuralNetwork.getLearningDatataSetResults().size());


            int j=0;
            for (int i = 0; i < neuralNetwork.getLearningDatataSetResults().size(); i++) {
                if( i %(neuralNetwork.getLearningDataSetDecisions().size()-1)==0){ //multiplicity
                    j=0;
                }
                System.out.println(i + ". " + "error: " + neuralNetwork.getErrorsBefore().get(i) + ",  value: " + neuralNetwork.getLearningDatataSetResults().get(i)[0] + " , expected value: " + neuralNetwork.getLearningDataSetDecisions().get(j));
                j++;

            }













   /*         NeuralNetwork neuralNetwork= new NeuralNetwork(file.getLearningDataSet(), file.getTestDataSet(),1,null,null);
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



            //new weights

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

            System.out.println("NEW BIAS FOR NEURONS IN HIDDEN LAYER");
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



            System.out.println();
            System.out.println();
            System.out.println();
            System.out.println();
            System.out.println();
            System.out.println();
            System.out.println("NEURAL NETWORK 1");

            //testing Method calculateOutputForNetwork and constructor parapeters and class atributes
            NeuralNetwork neuralNetwork1= new NeuralNetwork(file.getLearningDataSet(), file.getTestDataSet(), 1, biasHiddenLayer,biasOutputLayer);
            neuralNetwork1.calculateOutputForNetwork(weightsW1, weightsM2,inputX_1,1);
            System.out.println("\n------- > neural network net1 :");
            NeuralNetwork.showMatrix(neuralNetwork1.getNet1());
            System.out.println("\n------- > neural network Y1 :");
            NeuralNetwork.showMatrix(neuralNetwork1.getY1());
            System.out.println("\n------- > neural network net2 :");
            NeuralNetwork.showMatrix(neuralNetwork1.getNet2());
            System.out.println("\n------- > neural network Y2 :");
            NeuralNetwork.showMatrix(neuralNetwork1.getY2());

            //errore
            neuralNetwork1.calculateErrorsForLayers(weightsW1,weightsM2, expectedValuesOutputLayer);
            System.out.println("\n------- > error for hidden layer:");
            NeuralNetwork.showMatrix(neuralNetwork1.getHiddenLayerError());
            System.out.println("\n------- > error for output layer :");
            NeuralNetwork.showMatrix(neuralNetwork1.getOutpuLayerError());

            //new weights for hidden layer

            neuralNetwork1.calculateWeightsForHiddenLayer(weightsW1,1,inputX_1);
            System.out.println("\n------- > new weights for hidden layer :");
            NeuralNetwork.showMatrix(neuralNetwork1.getWeightsForHiddenLayer());

            //new bias for hidden layer
            neuralNetwork1.calculateNewBiasForHiddenLayer(biasHiddenLayer,1, hiddenLayerError);
            System.out.println("\n------- > new bias for hidden layer :");
            NeuralNetwork.showMatrix(neuralNetwork1.getBiasHiddenLayer());

            //new weights for hidden layer
            neuralNetwork1.calculateWeightsForOutputLayer(weightsM2,1,Y1_1);
            System.out.println("\n------- > new weights for output layer :");
            NeuralNetwork.showMatrix(neuralNetwork1.getWeightsForOutputLayer());

            //new bias for output layer
            neuralNetwork1.calculateNewBiasForOutputLayer(biasOutputLayer,1, outpuLayerError);
            System.out.println("\n------- > new bias for hidden layer :");
            NeuralNetwork.showMatrix(neuralNetwork1.getBiasOutputLayer());


            //group methods to backpropagation
            //decita about static or void methods, parameters in methods
            //write epoch with 1 method
            //how to calculate how backpropagation affect
            //how error is lower?

            System.out.println("NEURAL NETWORK 2");

            NeuralNetwork neuralNetwork2= new NeuralNetwork(file.getLearningDataSet(), file.getTestDataSet(), 1, biasHiddenLayer,biasOutputLayer);
            neuralNetwork1.carryOutEpoch(weightsW1,weightsM2, inputX_1,1, expectedValuesOutputLayer, 1, biasHiddenLayer, biasOutputLayer);


            System.out.println("\n------- > new weights for hidden layer :");
            NeuralNetwork.showMatrix(neuralNetwork1.getWeightsForHiddenLayer());

            System.out.println("\n------- > new bias for hidden layer :");
            NeuralNetwork.showMatrix(neuralNetwork1.getBiasHiddenLayer());

            System.out.println("\n------- > new weights for output layer :");
            NeuralNetwork.showMatrix(neuralNetwork1.getWeightsForOutputLayer());

            System.out.println("\n------- > new bias for hidden layer :");
            NeuralNetwork.showMatrix(neuralNetwork1.getBiasOutputLayer());


            //check method and neuram network for 1 row from file
            //error pefore and after


            System.out.println("\n\nNEURAL NETWORK 3");

            //metoda losujaca ?
            double [] biasHiddenLayer3={0,0,0,0,0,0,0,0};
            double [] biasOutputLayer3={0};
            NeuralNetwork neuralNetwork3= new NeuralNetwork(file.getLearningDataSet(), file.getTestDataSet(), 1, biasHiddenLayer3,biasOutputLayer3);
            //2 layers ( without input layer)
            //hidden layer 8 neurons
            //output layer 1 neuron
            //sigmoid function


            //get input Vectot ( 8 features)
            double [] inputVector= neuralNetwork3.getLearningDataSetFeatures().get(0);
            double []expectedValue=  neuralNetwork3.getLearningDataSetDecisions().get(0);// what if is onlu one neuron in output vector ? make methods for double instead double []?
            NeuralNetwork.showMatrix(inputVector);
            System.out.println(expectedValue[0]);
            double[][] weightsHidden= NeuralNetwork.getRandomWeightsMatrix(8,8);
            double[][] weightsOutput= NeuralNetwork.getRandomWeightsMatrix(8,1);
            NeuralNetwork.showMatrix(weightsHidden);
            NeuralNetwork.showMatrix(weightsOutput);
            neuralNetwork3.carryOutEpoch(weightsHidden,weightsOutput,inputVector,1,expectedValue,0.01, biasHiddenLayer3,biasOutputLayer3);


            System.out.println("\n------- > neural network net1 :");
            NeuralNetwork.showMatrix(neuralNetwork3.getNet1());
            System.out.println("\n------- > neural network Y1 :");
            NeuralNetwork.showMatrix(neuralNetwork3.getY1());
            System.out.println("\n------- > neural network net2 :");
            NeuralNetwork.showMatrix(neuralNetwork3.getNet2());
            System.out.println("\n------- > neural network Y2 :");
            NeuralNetwork.showMatrix(neuralNetwork3.getY2());

            //errore
            neuralNetwork1.calculateErrorsForLayers(weightsW1,weightsM2, expectedValuesOutputLayer);
            System.out.println("\n------- > error for hidden layer:");
            NeuralNetwork.showMatrix(neuralNetwork3.getHiddenLayerError());
            System.out.println("\n------- > error for output layer :");
            NeuralNetwork.showMatrix(neuralNetwork3.getOutpuLayerError());

            System.out.println("\n------- > new weights for hidden layer :");
            NeuralNetwork.showMatrix(neuralNetwork3.getWeightsForHiddenLayer());

            System.out.println("\n------- > new bias for hidden layer :");
            NeuralNetwork.showMatrix(neuralNetwork3.getBiasHiddenLayer());

            System.out.println("\n------- > new weights for output layer :");
            NeuralNetwork.showMatrix(neuralNetwork3.getWeightsForOutputLayer());

            System.out.println("\n------- > new bias for hidden layer :");
            NeuralNetwork.showMatrix(neuralNetwork3.getBiasOutputLayer());



         System.out.println("\n------- > error before :" +neuralNetwork3.getMeanSquaredErrorBeforeEpoch());
         System.out.println("\n------- > error after :" +neuralNetwork3.getMeanSquaredErrorAfterEpoch());



         List<double[]> emptyList= new ArrayList<>();
         neuralNetwork1.setLearningDataSetResults(emptyList);


         List<double[]> data= neuralNetwork1.getLearningDataSetFeatures();
        for(int i =0 ; i< data.size(); i++){
         double [] input=data.get(i);
         double [] expected= neuralNetwork1.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
         NeuralNetwork.showMatrix(inputVector);
         System.out.println(expectedValue[0]);
         double[][] weightsHidden_1= NeuralNetwork.getRandomWeightsMatrix(8,8);
         double[][] weightsOutput_1= NeuralNetwork.getRandomWeightsMatrix(8,1);
         NeuralNetwork.showMatrix(weightsHidden);
         NeuralNetwork.showMatrix(weightsOutput);
         neuralNetwork3.carryOutEpoch(weightsHidden_1,weightsOutput_1,input,1,expected,0.01, biasHiddenLayer3,biasOutputLayer3);
        }

        //decide if expected value, and real values should be stored in double [] or double






         List<double[]> allValues=neuralNetwork1.getLearningDataSetResults();
         List<double[]> allExpectedValues=  neuralNetwork1.getLearningDataSetDecisions();

        for(int i = 0; i < allValues.size(); i++){
         System.out.print("\n    "+ i +".  ");
         //for one neuron
          System.out.print(" expected: "+ allValues.get(i)[0]+ " expected: "+ allExpectedValues.get(i)[0] );

        }


*/



        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
