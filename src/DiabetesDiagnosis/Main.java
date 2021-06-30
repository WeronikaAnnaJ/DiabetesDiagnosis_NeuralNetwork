package DiabetesDiagnosis;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;



public class Main {

    public static void main(String[] args) {

        try {
            /** Get data from file */
            ReadFile file = new ReadFile();
            file.readCsv();
            file.segregateData();

            /** Segregate data for  */
            Data data = new Data(file.getLearningDataSet(), file.getTestingDataSet());


            /**
             *
             *
             * Neural network 1
             *
             *
             * */
            /** Neural network - one hidden layer -> 14 neurons, one output layer - 1 neuron  */
            data.mixAndMultiplyLearningData(data.getLearningDataSet());
            data.mixAndMultiplyLearningData(data.getLearningDataSet());
            data.mixAndMultiplyLearningData(data.getLearningDataSet());
            System.out.println("How many rows/epoch for testing learning rate : "   +data.getMultipliedLearningDataSetFeatures().size());


            NeutalNetwork1Layer neuralNetwork = new NeutalNetwork1Layer(data.getMultipliedLearningDataSetFeatures(), data.getMultipliedLearningDataSetDecisions());
            double[] biasHiddenLayer = {0, 0, 0, 0, 0, 0, 0, 0,0, 0,0, 0, 0, 0};
            double[] biasOutputLayer = {0};
            neuralNetwork.setBiasHiddenLayer(biasHiddenLayer);
            neuralNetwork.setBiasOutputLayer(biasOutputLayer);
            neuralNetwork.setLambda(1);
            neuralNetwork.setLearningRate(0.01);
            double [][] randomWeightsHiddenLayer=Matrix.getRandomWeightsMatrix(8, 14);
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            double [][] randomWeightsOutputLayer=Matrix.getRandomWeightsMatrix(14, 1);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);


            double errorBefore_1= 0;
            double errorAfter_1= 0;

            List<Double> errorsForLearningRate=new ArrayList<>();

            /** test which learning date is best* */
            System.out.println("\nNEURAL NETWORK 1\n");

            /** without learning  */
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
                neuralNetwork.calculateOutputForNetwork(inputVector);
            }

            errorBefore_1=neuralNetwork.calculateMeanSquaredError();
            neuralNetwork.resetLearningDataSetResults();
            errorsForLearningRate.add(errorBefore_1);

            /** learning rate 0.01 */
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForLearningRate.add(errorAfter_1);
            System.out.println("Neural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("learning rate 0.01");
            System.out.println("ERROR BEFORE: " + errorBefore_1);
            System.out.println("ERROR AFTER:  " + errorAfter_1);


            neuralNetwork.setLearningRate(0.1);
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            neuralNetwork.resetLearningDataSetResults();


            /** learning rate 0.1  */
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForLearningRate.add(errorAfter_1);
            System.out.println("\nNeural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("> learning rate 0.1");
            System.out.println("ERROR BEFORE: " + errorBefore_1);
            System.out.println("ERROR AFTER:  " + errorAfter_1);

            neuralNetwork.setLearningRate(0.2);
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            neuralNetwork.resetLearningDataSetResults();


            /** learning rate 0.2  */
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForLearningRate.add(errorAfter_1);
            System.out.println("\nNeural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("> learning rate 0.2");
            System.out.println("ERROR BEFORE: " + errorBefore_1);
            System.out.println("ERROR AFTER:  " + errorAfter_1);

            neuralNetwork.setLearningRate(0.3);
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            neuralNetwork.resetLearningDataSetResults();


            /** learning rate 0.3  */
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForLearningRate.add(errorAfter_1);
            System.out.println("\nNeural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("> learning rate 0.3");
            System.out.println("ERROR BEFORE: " + errorBefore_1);
            System.out.println("ERROR AFTER:  " + errorAfter_1);

            neuralNetwork.setLearningRate(0.4);
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            neuralNetwork.resetLearningDataSetResults();


            /** learning rate 0.4 */
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForLearningRate.add(errorAfter_1);
            System.out.println("\nNeural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("> learning rate 0.4");
            System.out.println("ERROR BEFORE: " + errorBefore_1);
            System.out.println("ERROR AFTER:  " + errorAfter_1);

            neuralNetwork.setLearningRate(0.5);
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            neuralNetwork.resetLearningDataSetResults();


            /** learning rate 0.5 */
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForLearningRate.add(errorAfter_1);
            System.out.println("\nNeural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("> learning rate 0.5");
            System.out.println("ERROR BEFORE: " + errorBefore_1);
            System.out.println("ERROR AFTER:  " + errorAfter_1);

            neuralNetwork.setLearningRate(0.6);
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            neuralNetwork.resetLearningDataSetResults();


            /** learning rate 0.6 */
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForLearningRate.add(errorAfter_1);
            System.out.println("\nNeural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("> learning rate 0.6");
            System.out.println("ERROR BEFORE: " + errorBefore_1);
            System.out.println("ERROR AFTER:  " + errorAfter_1);

            neuralNetwork.setLearningRate(0.7);
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            neuralNetwork.resetLearningDataSetResults();


            /** learning rate 0.7 */
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForLearningRate.add(errorAfter_1);
            System.out.println("\nNeural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("> learning rate 0.7");
            System.out.println("ERROR BEFORE: " + errorBefore_1);
            System.out.println("ERROR AFTER:  " + errorAfter_1);

            neuralNetwork.setLearningRate(0.8);
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            neuralNetwork.resetLearningDataSetResults();


            /** learning rate 0.8*/
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForLearningRate.add(errorAfter_1);
            System.out.println("\nNeural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("> learning rate 0.8");
            System.out.println("ERROR BEFORE: " + errorBefore_1);
            System.out.println("ERROR AFTER:  " + errorAfter_1);

            neuralNetwork.setLearningRate(0.9);
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            neuralNetwork.resetLearningDataSetResults();


            /** learning rate 0.9*/
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForLearningRate.add(errorAfter_1);
            System.out.println("\nNeural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("> learning rate 0.9");
            System.out.println("ERROR BEFORE: " + errorBefore_1);
            System.out.println("ERROR AFTER:  " + errorAfter_1);

            neuralNetwork.setLearningRate(1);
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            neuralNetwork.resetLearningDataSetResults();


            /** learning rate 1*/
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForLearningRate.add(errorAfter_1);
            System.out.println("\nNeural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("> learning rate 1");
            System.out.println("ERROR BEFORE: " + errorBefore_1);
            System.out.println("ERROR AFTER:  " + errorAfter_1);


            Double [] learningRates= {0.0, 0.01, 0.1, 0.2 ,0.3 ,0.4 ,0.5, 0.6 ,0.7 ,0.8, 0.9, 1.0 };

            System.out.println("\nErrors for learning rates");
            int count=0;
            for (Double error:errorsForLearningRate) {
                System.out.println("> learning rate : " + learningRates[count] + "  error: " +error);
                count++;
            }


            /** Neural network 1 */

            /** test how many epoch is best (learning rate 0.01) */

            List<Double> errorsForEpochs=new ArrayList<>();
            neuralNetwork.setLearningRate(0.01);

            System.out.println("\nHow many epochs is best ? (learning rate 0.01) ");

            /** 576 epochs */

            data.resetMultipliedLearningDataSet();
            data.mixAndMultiplyLearningData();
            neuralNetwork.resetLearningDataSetResults();
            neuralNetwork.setDataSets(data.getMultipliedLearningDataSetFeatures(), data.getMultipliedLearningDataSetDecisions());
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);

            System.out.println("\n> Number of Epochs : "+ neuralNetwork.getLearningDataSetFeatures().size());
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {
                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?

                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForEpochs.add(errorAfter_1);
            System.out.println("Neural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("ERROR AFTER:  " + errorAfter_1);


            /** 1728 epochs */

            data.mixAndMultiplyLearningData();
            neuralNetwork.resetLearningDataSetResults();
            neuralNetwork.setDataSets(data.getMultipliedLearningDataSetFeatures(), data.getMultipliedLearningDataSetDecisions());
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            System.out.println("\n>  Number of Epochs : "+ neuralNetwork.getLearningDataSetFeatures().size());
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {
                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForEpochs.add(errorAfter_1);
            System.out.println("Neural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("ERROR AFTER:  " + errorAfter_1);

            /** 3456 epochs */

            data.mixAndMultiplyLearningData();
            neuralNetwork.resetLearningDataSetResults();
            neuralNetwork.setDataSets(data.getMultipliedLearningDataSetFeatures(), data.getMultipliedLearningDataSetDecisions());
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            System.out.println("\n> Number of Epochs : "+ neuralNetwork.getLearningDataSetFeatures().size());
            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {
                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForEpochs.add(errorAfter_1);
            System.out.println("Neural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("ERROR AFTER:  " + errorAfter_1);


            /** 5760 epochs */

            data.mixAndMultiplyLearningData();
            neuralNetwork.resetLearningDataSetResults();
            neuralNetwork.setDataSets(data.getMultipliedLearningDataSetFeatures(), data.getMultipliedLearningDataSetDecisions());
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            System.out.println("\n> Number of Epochs : "+ neuralNetwork.getLearningDataSetFeatures().size());

            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {
                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForEpochs.add(errorAfter_1);
            System.out.println("Neural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("ERROR AFTER:  " + errorAfter_1);



            /** 8640 epochs */

            data.mixAndMultiplyLearningData();
            neuralNetwork.resetLearningDataSetResults();
            neuralNetwork.setDataSets(data.getMultipliedLearningDataSetFeatures(), data.getMultipliedLearningDataSetDecisions());
            neuralNetwork.setWeightsHiddenLayer(randomWeightsHiddenLayer);
            neuralNetwork.setWeightsOutputLayer(randomWeightsOutputLayer);
            System.out.println("\n> Number of Epochs : "+ neuralNetwork.getLearningDataSetFeatures().size());

            for (int i = 0; i < neuralNetwork.getLearningDataSetFeatures().size(); i++) {
                double[] inputVector = neuralNetwork.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
                neuralNetwork.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            neuralNetwork.resetCorrectResults();
            errorAfter_1=neuralNetwork.calculateMeanSquaredError();
            errorsForEpochs.add(errorAfter_1);
            System.out.println("Neural Network wih 1 hidden layer: Neural Network 1");
            System.out.println("ERROR AFTER:  " + errorAfter_1);
            System.out.println("Correct results : "+ neuralNetwork.getCorrectResult()+ " everything : " + neuralNetwork.getLearningDataSetDecisions().size());


            Integer [] numberOfEpochs ={ 576, 1728, 3456, 5760, 8640 };
            System.out.println("\nErrors for epochs : ");
            int countNumberOfEpochs=0;
            for (Double error:errorsForEpochs) {
                System.out.println("> for " + numberOfEpochs[countNumberOfEpochs] + "   error: "+ error);
                countNumberOfEpochs++;
            }






            /**
             *
             *
             * Sieć neuronowa 2_A
             *
             *
             * */


            System.out.println("\n\nNEURAL NETWORK 2_A\n");
            /** Neural network - 2 hidden layer -> 8 neurons, 4 neurons, one output layer - 1 neuron  */
            NeuralNetwork neuralNetwork2_A = new NeuralNetwork(data.getMultipliedLearningDataSetFeatures(), data.getMultipliedLearningDataSetDecisions());
            double[] biasHiddenLayer2_A = {0, 0, 0, 0, 0, 0, 0, 0,0,0}; //whitch matrix
            double[] biasHiddenLayer2_A_2 = {0, 0, 0, 0};
            double[] biasOutputLayer2_A= {0};
            neuralNetwork2_A.setBiasHiddenLayer(biasHiddenLayer2_A);
            neuralNetwork2_A.setBiasHiddenLayer2(biasHiddenLayer2_A_2);
            neuralNetwork2_A.setBiasOutputLayer(biasOutputLayer2_A);
            neuralNetwork2_A.setLearningRate(0.01);
            neuralNetwork2_A.setWeightsHiddenLayer(Matrix.getRandomWeightsMatrix(8, 10));
            neuralNetwork2_A.setWeightsHiddenLayer2(Matrix.getRandomWeightsMatrix(10, 4));
            neuralNetwork2_A.setWeightsOutputLayer(Matrix.getRandomWeightsMatrix(4, 1));
            double errorBefore2_A=0;
            double errorAfter2_A=0;

            for (int i = 0; i < neuralNetwork2_A.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork2_A.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork2_A.getLearningDataSetDecisions().get(i);
                neuralNetwork2_A.calculateOutputForNetwork(neuralNetwork2_A.getWeightsForHiddenLayer(),neuralNetwork2_A.getWeightsForHiddenLayer2(), neuralNetwork2_A.getWeightsForOutputLayer(),inputVector,1);
            }

            errorBefore2_A=neuralNetwork2_A.calculateMeanSquaredError();
            neuralNetwork2_A.resetLearningDataSetResults();

            for (int i = 0; i < neuralNetwork2_A.getLearningDataSetFeatures().size(); i++) {
                double[] inputVector = neuralNetwork2_A.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork2_A.getLearningDataSetDecisions().get(i);
                neuralNetwork2_A.carryOutEpoch(inputVector,expectedValuesOutputLayer,neuralNetwork2_A.getBiasHiddenLayer(),neuralNetwork2_A.getBiasHiddenLayer2(),neuralNetwork2_A.getBiasOutputLayer());
            }

            neuralNetwork2_A.resetCorrectResults();
            errorAfter2_A=neuralNetwork2_A.calculateMeanSquaredError();
            neuralNetwork2_A.resetLearningDataSetResults();
            System.out.println("Hidden layers: 2");
            System.out.println("Neurons hidden layer 1: 10");
            System.out.println("Neurons hidden layer 2: 4");
            System.out.println("Neurons output layer: 1");
            System.out.println("Funcjon for hidden layers: sigmoid unipolar");
            System.out.println("Funcjon for output layer: step unipolar");
            System.out.println("ERROR BEFORE: " + errorBefore2_A);
            System.out.println("ERROR AFTER:  " + errorAfter2_A);
            System.out.println("Correct decisisions: "+ neuralNetwork2_A.getCorrectResult()+ "   all decisions: " + neuralNetwork2_A.getLearningDataSetDecisions().size());







            /**
             *
             *
             * Sieć neuronowa 2_B
             *
             *
             * */

            System.out.println("\n\nNEURAL NETWORK 2_B\n");
            /** Neural network - 2 hidden layer -> 24 neurons, 14 neurons, one output layer - 1 neuron  */
            NeuralNetwork neuralNetwork2_B = new NeuralNetwork(data.getMultipliedLearningDataSetFeatures(), data.getMultipliedLearningDataSetDecisions());
            double[] biasHiddenLayer2_B = {0, 0, 0, 0, 0, 0, 0, 0,0,0,0, 0, 0, 0,0,0,0, 0, 0, 0, 0,0,0, 0 }; //whitch matrix
            double[] biasHiddenLayer2_B_2 = {0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0,0, 0, 0, 0,};
            double[] biasOutputLayer2_B = {0};
            neuralNetwork2_B.setBiasHiddenLayer(biasHiddenLayer2_B);
            neuralNetwork2_B.setBiasHiddenLayer2(biasHiddenLayer2_B_2);
            neuralNetwork2_B.setBiasOutputLayer(biasOutputLayer2_B);
            neuralNetwork2_B.setLambda(1);
            neuralNetwork2_B.setLearningRate(0.01);
            neuralNetwork2_B.setWeightsHiddenLayer(Matrix.getRandomWeightsMatrix(8, 24));
            neuralNetwork2_B.setWeightsHiddenLayer2(Matrix.getRandomWeightsMatrix(24, 14));
            neuralNetwork2_B.setWeightsOutputLayer(Matrix.getRandomWeightsMatrix(14, 1));
            double errorBefore2_B=0;
            double errorAfter2_B=0;

            for (int i = 0; i <neuralNetwork2_B.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork2_B.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork2_B.getLearningDataSetDecisions().get(i);
                neuralNetwork2_B.calculateOutputForNetwork(neuralNetwork2_B.getWeightsForHiddenLayer(),neuralNetwork2_B.getWeightsForHiddenLayer2(), neuralNetwork2_B.getWeightsForOutputLayer(),inputVector,1);
            }

            errorBefore2_B=neuralNetwork2_B.calculateMeanSquaredError();
            neuralNetwork2_B.resetLearningDataSetResults();

            for (int i = 0; i < neuralNetwork2_B.getLearningDataSetFeatures().size(); i++) {
                double[] inputVector = neuralNetwork2_B.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork2_B.getLearningDataSetDecisions().get(i);
                neuralNetwork2_B.carryOutEpoch(inputVector,expectedValuesOutputLayer,neuralNetwork2_B.getBiasHiddenLayer(),neuralNetwork2_B.getBiasHiddenLayer2(),neuralNetwork2_B.getBiasOutputLayer());
            }

            neuralNetwork2_B.resetCorrectResults();
            errorAfter2_B=neuralNetwork2_B.calculateMeanSquaredError();
            neuralNetwork2_B.resetLearningDataSetResults();
            System.out.println("Hidden layers: 2");
            System.out.println("Neurons hidden layer 1: 24");
            System.out.println("Neurons hidden layer 2: 14");
            System.out.println("Neurons output layer:  1");
            System.out.println("Funcjon for hidden layers: sigmoid unipolar");
            System.out.println("Funcjon for output layer: step unipolar");
            System.out.println("ERROR BEFORE: " + errorBefore2_B);
            System.out.println("ERROR AFTER: " + errorAfter2_B);
            System.out.println("Correct decisisions: "+ neuralNetwork2_B.getCorrectResult()+ "  all decisions: " + neuralNetwork2_B.getLearningDataSetDecisions().size());




            /**
             *
             *
             * Sieć neuronowa 2_C
             *
             *
             * */
            System.out.println("\n\nNEURAL NETWORK 2_C\n");
            /** Neural network - 2 hidden layer -> 8 neurons, 4 neurons one output layer - 1 neuron  */
            NeuralNetwork2Layers neuralNetwork2Layers= new NeuralNetwork2Layers(data.getMultipliedLearningDataSetFeatures(),data.getMultipliedLearningDataSetDecisions());
            double[] biasHiddenLayer2 = {0, 0, 0, 0, 0, 0,0,0};
            double[] biasHiddenLayer2_1 = {0, 0, 0, 0};
            double[] biasOutputLayer2 = {0};
            neuralNetwork2Layers.setBiasHiddenLayer(biasHiddenLayer2);
            neuralNetwork2Layers.setBiasHiddenLayer2(biasHiddenLayer2_1);
            neuralNetwork2Layers.setBiasOutputLayer(biasOutputLayer2);
            neuralNetwork2Layers.setLambda(1);
            neuralNetwork2Layers.setLearningRate(0.1);
            neuralNetwork2Layers.setWeightsHiddenLayer(Matrix.getRandomWeightsMatrix(8, 8));
            neuralNetwork2Layers.setWeightsHiddenLayer2(Matrix.getRandomWeightsMatrix(8, 4));
            neuralNetwork2Layers.setWeightsOutputLayer(Matrix.getRandomWeightsMatrix(4, 1));

            double errorBefore_2=0;
            double errorAfter_2=0;
            for (int i = 0; i < neuralNetwork2Layers.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork2Layers.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork2Layers.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
                neuralNetwork2Layers.calculateOutputForNetwork(inputVector);
            }

            errorBefore_2=neuralNetwork2Layers.calculateMeanSquaredError();
            neuralNetwork2Layers.resetLearningDataSetResults();

            for (int i = 0; i < neuralNetwork2Layers.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork2Layers.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork2Layers.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
                neuralNetwork2Layers.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }
            neuralNetwork2Layers.resetCorrectResults();
            errorAfter_2=neuralNetwork2Layers.calculateMeanSquaredError();
            System.out.println("Hidden layers: 2");
            System.out.println("Neurons hidden layer 1: 8");
            System.out.println("Neurons hidden layer 2: 4");
            System.out.println("Neurons output layer:  1");
            System.out.println("Funcjon for hidden layers: sigmoid bipolar");
            System.out.println("Funcjon for output layer: step unipolar");
            System.out.println("ERROR BEFORE: " + errorBefore_2);
            System.out.println("ERROR AFTER: " + errorAfter_2);
            System.out.println("Correct decisisions: "+ neuralNetwork2Layers.getCorrectResult()+ "  all decisions: " + neuralNetwork2Layers.getLearningDataSetDecisions().size());




            /**
             *
             *
             * Sieć neuronowa 3_A
             *
             *
             * */

            System.out.println("\n\nNEURAL NETWORK 3_A\n");
            /** Neural network - 3 hidden layer -> 10 neurons, 4 neurons, 2 neurons one output layer - 1 neuron  */
            NeuralNetwork3Layers neuralNetwork3_A = new NeuralNetwork3Layers(data.getMultipliedLearningDataSetFeatures(), data.getMultipliedLearningDataSetDecisions());
            double[] biasHiddenLayer3_A = {1, 1, 1, 1, 1, 1, 1, 1,1,1};
            double[] biasHiddenLayer3_A_1 = {1, 1, 1, 1};
            double[] biasHiddenLayer3_A_2 = {1, 1};
            double[] biasOutputLayer3_A = {1};
            neuralNetwork3_A.setBiasHiddenLayer(biasHiddenLayer3_A);
            neuralNetwork3_A.setBiasHiddenLayer2(biasHiddenLayer3_A_1);
            neuralNetwork3_A.setBiasHiddenLayer3(biasHiddenLayer3_A_2);
            neuralNetwork3_A.setBiasOutputLayer(biasOutputLayer3_A);
            neuralNetwork3_A.setLambda(1);
            neuralNetwork3_A.setLearningRate(0.01);
            neuralNetwork3_A.setWeightsHiddenLayer(Matrix.getRandomWeightsMatrix(8, 10));
            neuralNetwork3_A.setWeightsHiddenLayer2(Matrix.getRandomWeightsMatrix(10, 4));
            neuralNetwork3_A.setWeightsHiddenLayer3(Matrix.getRandomWeightsMatrix(4, 2));
            neuralNetwork3_A.setWeightsOutputLayer(Matrix.getRandomWeightsMatrix(2, 1));

            double errorBefore3_A=0;
            double errorAfter3_A=0;

            for (int i = 0; i < neuralNetwork3_A.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork3_A.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork3_A.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
                neuralNetwork3_A.calculateOutputForNetwork(inputVector);
            }

            errorBefore3_A=neuralNetwork3_A.calculateMeanSquaredError();
            neuralNetwork3_A.resetLearningDataSetResults();

            for (int i = 0; i <neuralNetwork3_A.getLearningDataSetFeatures().size(); i++) {
                double[] inputVector =neuralNetwork3_A.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork3_A.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
                neuralNetwork3_A.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }

            neuralNetwork3_A.resetCorrectResults();
            errorAfter3_A=neuralNetwork3_A.calculateMeanSquaredError();
            neuralNetwork3_A.resetLearningDataSetResults();
            System.out.println("Hidden layers: 3");
            System.out.println("Neurons hidden layer 1: 10");
            System.out.println("Neurons hidden layer 2: 4");
            System.out.println("Neurons hidden layer 3: 2");
            System.out.println("Neurons output layer: 1");
            System.out.println("Funcjon for hidden layers: sigmoid unipolar");
            System.out.println("Funcjon for output layer: step unipolar");
            System.out.println("ERROR BEFORE: " + errorBefore3_A);
            System.out.println("ERROR AFTER:  " + errorAfter3_A);
            System.out.println("Correct decisisions: "+ neuralNetwork3_A.getCorrectResult()+ "   all decisions: " + neuralNetwork3_A.getLearningDataSetDecisions().size());







            /**
             *
             *
             * Sieć neuronowa 3_B
             *
             *
             * */

            System.out.println("\n\nNEURAL NETWORK 3_B\n");
            /** Neural network - 3 hidden layer -> 20 neurons, 5 neurons,3 neurons one output layer - 1 neuron  */
            NeuralNetwork3Layers neuralNetwork3_B = new NeuralNetwork3Layers(data.getMultipliedLearningDataSetFeatures(), data.getMultipliedLearningDataSetDecisions());
            double[] biasHiddenLayer3_B = {1, 1, 1, 1, 1, 1, 1, 1,1,1,1, 1, 1, 1, 1, 1, 1, 1,1,1,};
            double[] biasHiddenLayer3_B_1 = {1, 1, 1, 1, 1};
            double[] biasHiddenLayer3_B_2 = {1, 1,1};
            double[] biasOutputLayer3_B = {1};
            neuralNetwork3_B.setBiasHiddenLayer(biasHiddenLayer3_B);
            neuralNetwork3_B.setBiasHiddenLayer2(biasHiddenLayer3_B_1);
            neuralNetwork3_B.setBiasHiddenLayer3(biasHiddenLayer3_B_2);
            neuralNetwork3_B.setBiasOutputLayer(biasOutputLayer3_B);
            neuralNetwork3_B.setLambda(1);
            neuralNetwork3_B.setLearningRate(0.1);
            neuralNetwork3_B.setWeightsHiddenLayer(Matrix.getRandomWeightsMatrix(8, 20));
            neuralNetwork3_B.setWeightsHiddenLayer2(Matrix.getRandomWeightsMatrix(20, 5));
            neuralNetwork3_B.setWeightsHiddenLayer3(Matrix.getRandomWeightsMatrix(5, 3));
            neuralNetwork3_B.setWeightsOutputLayer(Matrix.getRandomWeightsMatrix(3, 1));

            double errorBefore3_B=0;
            double errorAfter3_B=0;

            for (int i = 0; i < neuralNetwork3_B.getLearningDataSetFeatures().size(); i++) {

                double[] inputVector = neuralNetwork3_B.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork3_B.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
                neuralNetwork3_B.calculateOutputForNetwork(inputVector);
            }

            errorBefore3_B=neuralNetwork3_B.calculateMeanSquaredError();
            neuralNetwork3_B.resetLearningDataSetResults();

           for (int i = 0; i < neuralNetwork3_B.getLearningDataSetFeatures().size(); i++) {
                double[] inputVector = neuralNetwork3_B.getLearningDataSetFeatures().get(i);
                double[] expectedValuesOutputLayer = new double[1];
                expectedValuesOutputLayer[0] = neuralNetwork3_B.getLearningDataSetDecisions().get(i);// what if is onlu one neuron in output vector ? make methods for double instead double []?
               neuralNetwork3_B.carryOutEpoch(inputVector, expectedValuesOutputLayer);
            }

            neuralNetwork3_B.resetCorrectResults();
            errorAfter3_B=neuralNetwork3_B.calculateMeanSquaredError();
            neuralNetwork3_B.resetLearningDataSetResults();
            System.out.println("Hidden layers: 3");
            System.out.println("Neurons hidden layer 1: 20");
            System.out.println("Neurons hidden layer 2: 5");
            System.out.println("Neurons hidden layer 3: 3");
            System.out.println("Neurons output layer: 1");
            System.out.println("Funcjon for hidden layers: sigmoid unipolar");
            System.out.println("Funcjon for output layer: step unipolar");
            System.out.println("ERROR BEFORE: " + errorBefore3_B);
            System.out.println("ERROR AFTER:  " + errorAfter3_B);
            System.out.println("Correct decisisions: "+ neuralNetwork3_B.getCorrectResult()+ "   all decisions: " + neuralNetwork3_B.getLearningDataSetDecisions().size());




            /**
             *
             *
             * TESTING
             *
             *
             * */


            /** Neuron Network 1 */

            System.out.println("\n\nTEST");
            System.out.println("\nneuralNetwork1\ncorrect decisions ");
            double neuralNetwork1_correction= neuralNetwork.testAccurancy(data.getTestingDataSetFeatures(), data.getTestingDataSetDecisions());
            System.out.println("\nneuralNetwork2_A\ncorrect decision ");
            double neuralNetwork2_A_correction= neuralNetwork2_A.testAccurancy(data.getTestingDataSetFeatures(), data.getTestingDataSetDecisions());
            System.out.println("\nneuralNetwork2_B\ncorrect decisions ");
            double neuralNetwork2_B_correction= neuralNetwork2_B.testAccurancy(data.getTestingDataSetFeatures(), data.getTestingDataSetDecisions());
            System.out.println("\nneuralNetwork2_C\ncorrect decisions ");
            double neuralNetwork2_C_correction= neuralNetwork2Layers.testAccurancy(data.getTestingDataSetFeatures(), data.getTestingDataSetDecisions());
            System.out.println("\nneuralNetwork3_A\ncorrect decisions ");
            double neuralNetwork3_A_correction= neuralNetwork3_A.testAccurancy(data.getTestingDataSetFeatures(), data.getTestingDataSetDecisions());
            System.out.println("\nneuralNetwork3_B\ncorrect decisions ");
            double neuralNetwork3_B_correction= neuralNetwork3_B.testAccurancy(data.getTestingDataSetFeatures(), data.getTestingDataSetDecisions());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
