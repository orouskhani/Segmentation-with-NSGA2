package edu.shenzen.maysam.entities.problems;

import edu.shenzen.maysam.entities.math.Point;
import edu.shenzen.maysam.entities.ml.Centroid;
import edu.shenzen.maysam.entities.solutions.KMeansSolution;
import edu.shenzen.maysam.util.MathUtils;
import edu.shenzen.maysam.util.ml.ClusteringUtil;
import org.uma.jmetal.problem.impl.AbstractDoubleProblem;
import org.uma.jmetal.solution.DoubleSolution;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ClusterProblem extends AbstractDoubleProblem{

    private Integer numberOfClusters;
    private Integer numberOfFeatures;
    private List<Centroid> population;
    private Map<Integer, Point> dataset;
    private Point referencePoint;

    public ClusterProblem(int k , int c, List<Centroid> population, Map<Integer, Point> dataset) {
        setNumberOfClusters(k);
        setNumberOfFeatures(c);
        setNumberOfVariables(k*c);
        setName("Image Segmentation");
        setNumberOfObjectives(2);
        setDataset(dataset);
        setReferencePoint(findReferencePoint(new ArrayList<Point>(dataset.values())));

        this.population = new ArrayList<>(population);

        List<Double> lowerLimit = new ArrayList<>(getNumberOfVariables()) ;
        List<Double> upperLimit = new ArrayList<>(getNumberOfVariables()) ;

        for (int i = 0 ; i < getNumberOfVariables(); i++) {
            lowerLimit.add(0.0);
            upperLimit.add(255.0);
        }

        setLowerLimit(lowerLimit);
        setUpperLimit(upperLimit);

    }

    public Map<Integer, Point> getDataset() {
        return dataset;
    }

    public void setDataset(Map<Integer, Point> dataset) {
        this.dataset = dataset;
    }

    private Point findReferencePoint(List<Point> dataset) {

        double length = dataset.size();
        double[] referencePoint = new double[dataset.get(0).getDimension()];
        for(int i = 0 ; i < dataset.size() ; i++){
            for(int j = 0 ; j < dataset.get(i).getCoords().length ; j++){
                referencePoint[j] += dataset.get(i).getCoords()[j];
            }
        }
        for(int j = 0 ; j < referencePoint.length ; j++){
            referencePoint[j] /= length;
        }

        return new Point(referencePoint);
    }

    public Integer getNumberOfClusters() {
        return numberOfClusters;
    }

    public void setNumberOfClusters(Integer numberOfClusters) {
        this.numberOfClusters = numberOfClusters;
    }

    public Integer getNumberOfFeatures() {
        return numberOfFeatures;
    }

    public void setNumberOfFeatures(Integer numberOfFeatures) {
        this.numberOfFeatures = numberOfFeatures;
    }

    public Point getReferencePoint() {
        return referencePoint;
    }

    public void setReferencePoint(Point referencePoint) {
        this.referencePoint = referencePoint;
    }

    @Override
    public void evaluate(DoubleSolution solution) {
        double[] f = new double[getNumberOfObjectives()];

        List<Centroid> centroids = new ArrayList<>(ClusteringUtil.convertKMeansSolutionToClusters((KMeansSolution) (solution), getDataset()).values());

        int clusterIndex = ClusteringUtil.checkValiditiy(centroids);
        while(clusterIndex >= 0){
            ((KMeansSolution)solution).reset(clusterIndex);

            centroids.clear();
            centroids = new ArrayList<>(ClusteringUtil.convertKMeansSolutionToClusters((KMeansSolution) (solution), getDataset()).values());
            clusterIndex = ClusteringUtil.checkValiditiy(centroids);
        }
        double d = MathUtils.pdist2(new ArrayList<Point>(dataset.values()), centroids.stream().map(new Function<Centroid, Point>() {
            @Override
            public Point apply(Centroid centroid) {
                return centroid.getFeatures();
            }
        }).collect(Collectors.toList()));

        f[0] = d;
        solution.setObjective(0, f[0]);

        double dd = MathUtils.getDistanceOfCentroidFromReferencePoint(centroids, referencePoint);
        dd = dd / centroids.size();
        f[1] = 1 / dd ;

        solution.setObjective(1, f[1]);
    }

    @Override
    public DoubleSolution createSolution() {
        return new KMeansSolution(this, population, numberOfClusters, numberOfFeatures);
    }
}
