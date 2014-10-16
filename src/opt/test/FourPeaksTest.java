package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.*;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.ConvergenceTrainer;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 *
 * @version 1.0
 */
public class FourPeaksTest {
    /**
     * The n value
     */
//    private static final int N = 100;
    /**
     * The t value
     */
//    private static final int T = N / 4;
    private static int T = 10;

    public static void main(String[] args) {
        System.out.println("algorithm,optima,iterations,duration,n");

        for (int N = 10; N < 101; N += 10) {
            // Set T to 10% of N
            T = (int) Math.ceil(N * .1);

            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new SingleCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            OptimaTrainer fit;


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
