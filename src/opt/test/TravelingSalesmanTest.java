package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.*;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
//    private static final int N = 50;

    /**
     * The test main
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.println("algorithm,optima,iterations,duration,n");
        for (int N = 10; N < 101; N += 10) {
            Random random = new Random();
            // create the random points
            double[][] points = new double[N][2];
            for (int i = 0; i < points.length; i++) {
                points[i][0] = random.nextDouble();
                points[i][1] = random.nextDouble();
            }
            // for rhc, sa, and ga we use a permutation based encoding
            TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
            Distribution odd = new DiscretePermutationDistribution(N);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
            ef = new TravelingSalesmanSortEvaluationFunction(points);
            int[] ranges = new int[N];
            Arrays.fill(ranges, N);
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            OptimaTrainer fit;

            for (int i = 0; i < 5; i++) {
                HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
                fit = new OptimaTrainer(rhc, 5000000, true);
                fit.train();
                System.out.println("RHC," + fit.getOptima() + "," + fit.getIterations() + "," + fit.getTrainTime() + "," + N);

                SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
                fit = new OptimaTrainer(sa, 5000000, true);
                fit.train();
                System.out.println("SA," + fit.getOptima() + "," + fit.getIterations() + "," + fit.getTrainTime() + "," + N);

                GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
                fit = new OptimaTrainer(ga, 10000, true);
                fit.train();
                System.out.println("GA," + fit.getOptima() + "," + fit.getIterations() + "," + fit.getTrainTime() + "," + N);

                ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
                MIMIC mimic = new MIMIC(200, 20, pop);
                fit = new OptimaTrainer(mimic, 8000, true);
                fit.train();
                System.out.println("MIMIC," + fit.getOptima() + "," + fit.getIterations() + "," + fit.getTrainTime() + "," + N);
            }
        }
    }
}
