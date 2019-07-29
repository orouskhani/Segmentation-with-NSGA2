package edu.shenzen.maysam.entities.solutions;

import edu.shenzen.maysam.entities.enums.DistanceSet;
import edu.shenzen.maysam.entities.math.Point;
import edu.shenzen.maysam.entities.ml.Centroid;
import edu.shenzen.maysam.util.ml.ClusteringUtil;
import org.uma.jmetal.problem.DoubleProblem;
import org.uma.jmetal.solution.DoubleSolution;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.solution.impl.AbstractGenericSolution;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.ToDoubleFunction;

public class KMeansSolution extends AbstractGenericSolution<Double, DoubleProblem> implements DoubleSolution {

    int numberOfClusters;
    int numberOfFeatures;
    double db;

    /**
     * Constructor
     *
     * @param problem
     */
    protected KMeansSolution(DoubleProblem problem, int numberOfClusters, int numberOfFeatures) {
        super(problem);

        List<Centroid> population = new ArrayList<>();
        initializeDoubleVariables(population);
        initializeObjectiveValues();

        setNumberOfClusters(numberOfClusters);
        setNumberOfFeatures(numberOfFeatures);

    }

    public KMeansSolution(DoubleProblem problem, List<Centroid> population, int numberOfClusters, int numberOfFeatures) {
        super(problem);

        initializeDoubleVariables(population);
        initializeObjectiveValues();

        setNumberOfClusters(numberOfClusters);
        setNumberOfFeatures(numberOfFeatures);
    }


    /** Copy constructor */
    public KMeansSolution(KMeansSolution solution) {
        super(solution.problem) ;

        for (int i = 0; i < problem.getNumberOfVariables(); i++) {
            setVariableValue(i, solution.getVariableValue(i));
        }

        for (int i = 0; i < problem.getNumberOfObjectives(); i++) {
            setObjective(i, solution.getObjective(i)) ;
        }

        attributes = new HashMap<>(solution.attributes) ;
        setNumberOfClusters(solution.getNumberOfClusters());
        setNumberOfFeatures(solution.getNumberOfFeatures());
    }

    public int getNumberOfClusters() {
        return numberOfClusters;
    }

    public void setNumberOfClusters(int numberOfClusters) {
        this.numberOfClusters = numberOfClusters;
    }

    public int getNumberOfFeatures() {
        return numberOfFeatures;
    }

    public void setNumberOfFeatures(int numberOfFeatures) {
        this.numberOfFeatures = numberOfFeatures;
    }

    @Override
    public Double getLowerBound(int index) {
        return problem.getLowerBound(index);
    }

    @Override
    public Double getUpperBound(int index) {
        return problem.getUpperBound(index);
    }

    @Override
    public String getVariableValueString(int index) {
        return getVariableValue(index).toString() ;
    }

    @Override
    public Solution<Double> copy() {
        return new KMeansSolution(this);
    }

    private void initializeDoubleVariables(List<Centroid> population) {
       /* int dimension = population.get(0).getFeatures().getDimension();
        int clusters = population.size();
        for (int i = 0 ; i < population.size(); i++) {
            setVariableValue(i, 1.0) ;
        }

        int index = 0;
        for (int i = 0; i < population.size(); i++) {
            Centroid centroid = population.get(index);
            Point features = centroid.getFeatures();

            setVariableValue(clusters + i * dimension , features.getFeature(0)) ;
            setVariableValue(clusters + i * dimension + 1, features.getFeature(1)) ;
            setVariableValue(clusters + i * dimension + 2, features.getFeature(2)) ;

            index++;
        }*/
        for (int i = 0 ; i < problem.getNumberOfVariables(); i++) {
            Double value = randomGenerator.nextDouble(getLowerBound(i), getUpperBound(i)) ;
            setVariableValue(i, value) ;
        }
    }

    public double getDb() {
        return db;
    }

    public void calDBMeasure(Map<Integer, Point> dataset) {
        Map<Integer, Centroid> centroids = ClusteringUtil.convertKMeansSolutionToClusters(this, dataset);

        Map<Integer, Double> Si = new HashMap<>();
        for (Map.Entry<Integer, Centroid> entry : centroids.entrySet()) {
            double si = 0;
            Point center = entry.getValue().getFeatures();
            if(entry.getValue().getTrainingSamples().size() == 0){
                db = Double.POSITIVE_INFINITY;
                return;
               // Si.put(entry.getKey(), Double.POSITIVE_INFINITY);
               // continue;
            }
            for (Point point : entry.getValue().getTrainingSamples().values()) {
                si += point.findDistance(center, DistanceSet.EUCLIDEAN);
            }
            si /= entry.getValue().getTrainingSamples().size();

            Si.put(entry.getKey(), si);
        }

        int max = new Double(centroids.keySet().stream().mapToDouble(new ToDoubleFunction<Integer>() {
            @Override
            public double applyAsDouble(Integer value) {
                return new Double(value);
            }
        }).max().getAsDouble()).intValue();

        List<Double> rs = new ArrayList<>();
        for(int i = 0 ; i < max ; i++){
            double maxR = 0;
            for(int j = 0 ; j < max ; j++){
                if(i != j){
                    double dij = centroids.get(i).getFeatures().findDistance(centroids.get(j).getFeatures(), DistanceSet.EUCLIDEAN);
                    double temp = (Si.get(i) + Si.get(j)) / dij;
                    if(temp > maxR){
                        maxR = temp;
                    }
                }
            }
            rs.add(maxR);
        }
        db = rs.stream().mapToDouble(f -> f.doubleValue()).sum() / centroids.size();
    }

    private boolean isValid() {
        for (int i = 0; i < this.getNumberOfClusters(); i++) {
            if(this.getVariableValue(i) > ClusteringUtil.THRESHOLD){
                return Boolean.TRUE;
            }
        }
        return Boolean.FALSE;
    }

    public KMeansSolution reset(int clusterIndex) {

        for (int i = clusterIndex * this.getNumberOfFeatures() ; i < (clusterIndex + 1) * this.getNumberOfFeatures(); i++) {
            Double value = randomGenerator.nextDouble(getLowerBound(i), getUpperBound(i)) ;
            setVariableValue(i, value) ;
        }

        initializeObjectiveValues();
        return this;
    }
}
