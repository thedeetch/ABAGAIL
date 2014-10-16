package opt;

/**
 * A convergence trainer trains a network
 * until convergence, using another trainer
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class OptimaTrainer implements shared.Trainer {
    /**
     * The maxium number of iterations
     */
    private static final int MAX_ITERATIONS = 500;

    /**
     * The trainer
     */
    private OptimizationAlgorithm trainer;

    /**
     * The number of iterations trained
     */
    private int iterations;

    /**
     * The maximum number of iterations to use
     */
    private int maxIterations;

    private int factor;

    private double max = 0;

    private long time;

    /**
     * Create a new convergence trainer
     *
     * @param trainer       the thrainer to use
     * @param maxIterations the maximum iterations
     */
    public OptimaTrainer(OptimizationAlgorithm trainer, int maxIterations) {
        this(trainer, maxIterations, false);
    }

    public OptimaTrainer(OptimizationAlgorithm trainer, int maxIterations, boolean minimize) {
        this.trainer = trainer;
        this.maxIterations = maxIterations;
        if (minimize)
            this.factor = -1;
        else
            this.factor = 1;
    }

    /**
     * Create a new convergence trainer
     *
     * @param trainer the trainer to use
     */
    public OptimaTrainer(OptimizationAlgorithm trainer) {
        this(trainer, MAX_ITERATIONS, false);
    }

    /**
     * //     * @see Trainer#train()
     */
    public double train() {
        double current;
        int count = 0;
        long start = System.currentTimeMillis();
        do {
            count++;

            trainer.train();
            current = trainer.getOptimizationProblem().value(trainer.getOptimal()) * factor;

            if (count == 1) {
                iterations = count;
                max = current;
            }
            else if (current > max) {
                max = current;
                iterations = count;
                time = System.currentTimeMillis() - start;
            }
        } while (count < maxIterations);

        return max * factor;
    }

    /**
     * Get the number of iterations used
     *
     * @return the number of iterations
     */
    public int getIterations() {
        return iterations;
    }

    public double getOptima() {
        return max * factor;
    }

    public long getTrainTime() { return time; }
}
