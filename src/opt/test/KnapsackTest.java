package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.*;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.ConvergenceTrainer;
import shared.FixedIterationTrainer;

/**
 * A test of the knap sack problem
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackTest {
    /**
     * Random number generator
     */
    private static final Random random = new Random();
    /**
     * The number of items
     */
//    private static final int NUM_ITEMS = 50;
    /**
     * The number of copies each
     */
    private static final int COPIES_EACH = 3;
    /**
     * The maximum weight for a single element
     */
    private static final double MAX_WEIGHT = 75;
    /**
     * The maximum volume for a single element
     */
    private static final double MAX_VOLUME = 40;
    /**
     * The volume of the knapsack
     */
    private static double KNAPSACK_VOLUME;

    /**
     * The test main
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        for (int N = 1; N < 50; N++) {
            KNAPSACK_VOLUME = MAX_VOLUME * N * COPIES_EACH * .4;

            int[] copies = new int[N];
            Arrays.fill(copies, COPIES_EACH);
            double[] weights = new double[N];
            double[] volumes = new double[N];
            for (int i = 0; i < N; i++) {
                weights[i] = random.nextDouble() * MAX_WEIGHT;
                volumes[i] = random.nextDouble() * MAX_VOLUME;
            }
            int[] ranges = new int[N];
            Arrays.fill(ranges, COPIES_EACH + 1);
            EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            OptimaTrainer fit;

            System.out.println("algorithm,optima,iterations,duration,n");

            for (int i = 0; i < 5; i++) {
                HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                fit = new OptimaTrainer(rhc, 5000000);
                fit.train();
                System.out.println("RHC," + fit.getOptima() + "," + fit.getIterations() + "," + fit.getTrainTime() + "," + N);

                SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
                fit = new OptimaTrainer(sa, 5000000);
                fit.train();
                System.out.println("SA," + fit.getOptima() + "," + fit.getIterations() + "," + fit.getTrainTime() + "," + N);

                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
                fit = new OptimaTrainer(ga, 10000);
                fit.train();
                System.out.println("GA," + fit.getOptima() + "," + fit.getIterations() + "," + fit.getTrainTime() + "," + N);

                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
                MIMIC mimic = new MIMIC(200, 20, pop);
                fit = new OptimaTrainer(mimic, 8000);
                fit.train();
                System.out.println("MIMIC," + fit.getOptima() + "," + fit.getIterations() + "," + fit.getTrainTime() + "," + N);
            }
        }
    }
}
