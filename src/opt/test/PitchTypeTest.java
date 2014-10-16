package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying a pitch type given its spin and location.
 *
 * @author Nick Dietrich
 * @version 1.0
 */
public class PitchTypeTest {
    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) throws IOException {
        int inputLayer = 9, hiddenLayer = 2, outputLayer = 1, iterations = Integer.parseInt(args[3]);
        Instance[] instances = initializeInstances(args[0], Integer.parseInt(args[1]), Integer.parseInt(args[2]));

        ErrorMeasure measure = new SumOfSquaresError();
        DataSet set = new DataSet(instances);

        BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
        NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];
        BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
        OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];

        for (int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[]{inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E25, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for (int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], instances, iterations); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10, 9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            System.out.println("Weights: " + optimalInstance.getData().toString());

            double predicted, actual;
            start = System.nanoTime();
            for (int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10, 9);

            System.out.println("\nResults for " + oa[i].getClass().getName() + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n");
        }
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, Instance[] instances, int iterations) {
        ErrorMeasure measure = new SumOfSquaresError();

        System.out.println("\nError results for " + oa.getClass().getName() + "\n---------------------------");

        for (int i = 0; i < iterations; i++) {
            oa.train();

            double error = 0;
            for (Instance instance : instances) {
                network.setInputValues(instance.getData());
                network.run();

                Instance output = instance.getLabel();
                Instance example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

//            System.out.println(df.format(error));
        }

        System.out.println();
    }

    private static Instance[] initializeInstances(String file, int numberOfAttributes, int numberOfInstances) throws IOException {
        int i;
        double[][][] attributes = new double[numberOfInstances][][];

        BufferedReader br = new BufferedReader(new FileReader(new File(file)));

        for (i = 0; i < numberOfInstances; i++) {
            Scanner scan = new Scanner(br.readLine());
            scan.useDelimiter(",");

            attributes[i] = new double[2][];
            attributes[i][0] = new double[numberOfAttributes]; // 7 attributes
            attributes[i][1] = new double[1];

            for (int j = 0; j < numberOfAttributes; j++)
                attributes[i][0][j] = Double.parseDouble(scan.next());

            attributes[i][1][0] = Double.parseDouble(scan.next());
        }

        Instance[] instances = new Instance[attributes.length];

        for (i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // if a pitch is a fourseam fastball, give it a 1, else 0
            instances[i].setLabel(new Instance(attributes[i][1][0] == 3 ? 1 : 0));
        }

        return instances;
    }
}
