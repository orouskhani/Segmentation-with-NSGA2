package edu.shenzen.maysam.runner;

import edu.shenzen.maysam.algorithm.SmartNSGAII;
import edu.shenzen.maysam.entities.comparators.SimpleSolutionComparator;
import edu.shenzen.maysam.entities.math.Point;
import edu.shenzen.maysam.entities.ml.Centroid;
import edu.shenzen.maysam.entities.problems.ClusterProblem;
import edu.shenzen.maysam.entities.solutions.KMeansSolution;
import edu.shenzen.maysam.initial.DataInitialization;
import edu.shenzen.maysam.util.ImageUtils;
import edu.shenzen.maysam.util.ml.ClusteringUtil;
import edu.shenzen.maysam.util.ml.SilhouetteCoefficient;
import org.uma.jmetal.algorithm.Algorithm;
import org.uma.jmetal.operator.CrossoverOperator;
import org.uma.jmetal.operator.MutationOperator;
import org.uma.jmetal.operator.SelectionOperator;
import org.uma.jmetal.operator.impl.crossover.SBXCrossover;
import org.uma.jmetal.operator.impl.mutation.PolynomialMutation;
import org.uma.jmetal.operator.impl.selection.BinaryTournamentSelection;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.util.AbstractAlgorithmRunner;
import org.uma.jmetal.util.AlgorithmRunner;
import org.uma.jmetal.util.JMetalLogger;
import org.uma.jmetal.util.comparator.RankingAndCrowdingDistanceComparator;
import org.uma.jmetal.util.evaluator.impl.SequentialSolutionListEvaluator;
import org.uma.jmetal.util.fileoutput.SolutionListOutput;
import org.uma.jmetal.util.fileoutput.impl.DefaultFileOutputContext;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.uma.jmetal.util.SolutionListUtils.findBestSolution;

public class IrisCluster extends AbstractAlgorithmRunner {

    public static void main(String[] args) throws Exception {
        Problem<DoubleSolution> problem;
        Algorithm<List<DoubleSolution>> algorithm;
        CrossoverOperator<DoubleSolution> crossover;
        MutationOperator<DoubleSolution> mutation;
        SelectionOperator<List<DoubleSolution>, DoubleSolution> selection;

        int numberOfClusters = 15;
        int numberOfFeatures = 3;

        String filename = "C:\\Users\\Yasin\\git\\imagesegmentation\\resources\\ax.jpg";

        List<Point> dataset = DataInitialization.createDataset(filename);

        Map<Integer, Point> collectedDS = IntStream.range(0, dataset.size())
                .boxed()
                .collect(Collectors.toMap(i -> i, dataset::get));

        problem = new ClusterProblem(numberOfClusters , numberOfFeatures, new ArrayList<>(), collectedDS);

        double crossoverProbability = 0.9 ;
        double crossoverDistributionIndex = 20.0 ;
        crossover = new SBXCrossover(crossoverProbability, crossoverDistributionIndex) ;

        double mutationProbability = 1.0 / problem.getNumberOfVariables() ;
        double mutationDistributionIndex = 20.0 ;
        mutation = new PolynomialMutation(mutationProbability, mutationDistributionIndex) ;

        selection = new BinaryTournamentSelection<DoubleSolution>(new RankingAndCrowdingDistanceComparator<DoubleSolution>());

        algorithm = new SmartNSGAII<>(problem, 100, 100, crossover, mutation,
                selection, new SequentialSolutionListEvaluator<DoubleSolution>()) ;

        AlgorithmRunner algorithmRunner = new AlgorithmRunner.Executor(algorithm)
                .execute() ;

        long computingTime = algorithmRunner.getComputingTime() ;

        JMetalLogger.logger.info("Total execution time: " + computingTime + "ms");

        List<DoubleSolution> tempPopulation = algorithm.getResult();
        printFinalSolutionSet(tempPopulation);

        List<KMeansSolution> population = new ArrayList<>();
        for (DoubleSolution doubleSolution : tempPopulation) {
            population.add(new KMeansSolution((KMeansSolution) doubleSolution));
        }

        List<KMeansSolution> nomalizedPopulation = normalizePopulation(population);
        for (KMeansSolution kMeansSolution : nomalizedPopulation) {
            kMeansSolution.calDBMeasure(collectedDS);
        }

        System.out.println("For All Solutions");
        int populationindex = 0;
        for (KMeansSolution kMeansSolution : nomalizedPopulation) {
            System.out.println("For Solution at index : " + populationindex);
            populationindex++;
            Map<Integer, Centroid> ftempCentroids = ClusteringUtil.convertKMeansSolutionToClusters(kMeansSolution, collectedDS);
            for (Map.Entry<Integer, Centroid> entry : ftempCentroids.entrySet()) {
                System.out.println("Cluster Index : " + entry.getKey());
                System.out.println("Cluster Center : " + entry.getValue().getFeatures());
                System.out.println("Cluster Size : " + entry.getValue().getTrainingSamples().size());
            }
        }

        KMeansSolution solution = findBestSolution(nomalizedPopulation, new SimpleSolutionComparator());
        Map<Integer, Centroid> fCentroids = ClusteringUtil.convertKMeansSolutionToClusters(solution, collectedDS);

        System.out.println("For Final Solution");
        for (Map.Entry<Integer, Centroid> entry : fCentroids.entrySet()) {
            System.out.println("Cluster Index : " + entry.getKey());
            System.out.println("Cluster Center : " + entry.getValue().getFeatures());
            System.out.println("Cluster Size : " + entry.getValue().getTrainingSamples().size());
        }


        int[] groundTruth = new int[collectedDS.size()];
        for (Map.Entry<Integer, Point> entry : collectedDS.entrySet()) {
            groundTruth[entry.getKey()] = entry.getValue().getLabel();
        }

        int[] solutionDS = new int[collectedDS.size()];
        for (Map.Entry<Integer, Centroid> entry : fCentroids.entrySet()) {
            for (Integer index : entry.getValue().getTrainingSamples().keySet()) {
                solutionDS[index]=entry.getKey();
            }
        }

        double rndIndex = ClusteringUtil.randIndex(groundTruth, solutionDS);
        System.out.println("Rand Index : " + rndIndex);

       // double v = new SilhouetteCoefficient().silhouetteCoefficient(fCentroids);
        //System.out.println("Silhoutte Index : " + v);

        String writtenFilename = "C:\\Users\\Yasin\\git\\imagesegmentation\\resources\\ax_1.jpg";
        writeImage(solution, filename, writtenFilename, new ArrayList<>(fCentroids.values()));
    }

    private static void writeImage(KMeansSolution solution, String filename, String writtenFilename, List<Centroid> fCentroids) throws Exception {
        BufferedImage bi = ImageIO.read(new File(filename));

        List<KMeansSolution> bestSol = new ArrayList<>();
        bestSol.add(solution);
        new SolutionListOutput(bestSol)
                .setSeparator("\t")
                .setVarFileOutputContext(new DefaultFileOutputContext("bestVAR.tsv"))
                .setFunFileOutputContext(new DefaultFileOutputContext("bestFUN.tsv"))
                .print();


        int[][] modified_red_pixel = new int[bi.getWidth()][bi.getHeight()];
        int[][] modified_green_pixel = new int[bi.getWidth()][bi.getHeight()];
        int[][] modified_blue_pixel = new int[bi.getWidth()][bi.getHeight()];

        Color[] colors = new Color[fCentroids.size()];
        Random rand = new Random();

        for(int i = 0 ; i < colors.length ; i++){
            colors[i] = new Color(rand.nextFloat(), rand.nextFloat(), rand.nextFloat());
        }

        int index = 0;
        for (Centroid centroid : fCentroids) {
            Map<Integer, Point> trainingSamples = centroid.getTrainingSamples();
            for (Map.Entry<Integer, Point> e : trainingSamples.entrySet()) {
                int x = e.getKey() % bi.getWidth();
                int y = e.getKey() / bi.getWidth();
                modified_red_pixel[x][y] = colors[index].getRed();//new Double(features.getCoords()[0]).intValue();
                modified_green_pixel[x][y] = colors[index].getGreen();//new Double(features.getCoords()[1]).intValue();
                modified_blue_pixel[x][y] = colors[index].getBlue();//new Double(features.getCoords()[2]).intValue();
            }
            index++;
        }

        ImageUtils.writeRGB(modified_red_pixel, modified_green_pixel, modified_blue_pixel, writtenFilename);
    }

    private static List<KMeansSolution> normalizePopulation(List<KMeansSolution> population) {
        List<KMeansSolution> result = new ArrayList<>();

        double minFirstObjective = population.stream().mapToDouble(new ToDoubleFunction<DoubleSolution>() {
            @Override
            public double applyAsDouble(DoubleSolution value) {
                return value.getObjective(0);
            }
        }).min().getAsDouble();

        double maxFirstObjective = population.stream().mapToDouble(new ToDoubleFunction<DoubleSolution>() {
            @Override
            public double applyAsDouble(DoubleSolution value) {
                return value.getObjective(0);
            }
        }).max().getAsDouble();

        double minSecondObjective = population.stream().mapToDouble(new ToDoubleFunction<DoubleSolution>() {
            @Override
            public double applyAsDouble(DoubleSolution value) {
                return value.getObjective(1);
            }
        }).min().getAsDouble();

        double maxSecondObjective = population.stream().mapToDouble(new ToDoubleFunction<DoubleSolution>() {
            @Override
            public double applyAsDouble(DoubleSolution value) {
                return value.getObjective(1);
            }
        }).max().getAsDouble();

        for (KMeansSolution doubleSolution : population) {
            KMeansSolution sol = doubleSolution;

            double first = (doubleSolution.getObjective(0)-minFirstObjective) / (maxFirstObjective - minFirstObjective);
            sol.setObjective(0 , first);

            double second = (doubleSolution.getObjective(1)-minSecondObjective) / (maxSecondObjective - minSecondObjective);
            sol.setObjective(1 , second);

            result.add(sol);

        }
        return result;
    }
}
