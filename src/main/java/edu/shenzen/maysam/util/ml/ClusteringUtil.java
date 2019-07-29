package edu.shenzen.maysam.util.ml;

import edu.shenzen.maysam.entities.enums.DistanceSet;
import edu.shenzen.maysam.entities.math.Point;
import edu.shenzen.maysam.entities.ml.Centroid;
import edu.shenzen.maysam.entities.solutions.KMeansSolution;
import smile.math.Math;


import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClusteringUtil {

    public final static Double THRESHOLD = 0.0;

    public static Map<Integer, Centroid> convertKMeansSolutionToClusters(KMeansSolution solution, Map<Integer, Point> dataset){
        KMeansSolution s1 = new KMeansSolution(solution);
//        while(!s1.isValid()){
  //          s1 = s1.reset();
    //    }


        Map<Integer, Point> map = new HashMap<>();
        Map<Integer, Centroid> result = new HashMap<>();
        int index = 0;
        for (int i = 0; i < s1.getNumberOfClusters(); i++) {
            //if(s1.getVariableValue(i) > THRESHOLD){
                double[] values = new double[s1.getNumberOfFeatures()];
                for(int j = 0 ; j < s1.getNumberOfFeatures() ; j++){
                    values[j] = s1.getVariableValue(s1.getNumberOfFeatures() * i + j);
                }
                Point point = new Point();
                point.setCoords(values);
                map.put(index, point);

                Centroid centroid = new Centroid();
                centroid.setFeatures(point);
                result.put(index, centroid);
                index += 1;
            //}
        }

        for (Map.Entry<Integer, Point> pointEntry : dataset.entrySet()) {
            Point point = pointEntry.getValue();

            double minDistance = 100000000;
            int cluster = 0;
            for (Map.Entry<Integer, Point> e : map.entrySet()) {
                double distance = point.findDistance(e.getValue(), DistanceSet.EUCLIDEAN);
                if(distance < minDistance){
                    minDistance = distance;
                    cluster = e.getKey();
                }
            }
            Centroid centroid = result.get(cluster);
            centroid.addTrainingSample(pointEntry.getKey(), pointEntry.getValue());

            result.put(cluster, centroid);
        }
        return result;
    }

    public static double randIndex(int[] y1, int[] y2) {
        if (y1.length != y2.length) {
            throw new IllegalArgumentException(String.format("The vector sizes don't match: %d != %d.", y1.length, y2.length));
        }

        // Get # of non-zero classes in each solution
        int n = y1.length;

        int[] label1 = Math.unique(y1);
        int n1 = label1.length;

        int[] label2 = Math.unique(y2);
        int n2 = label2.length;

        // Calculate N contingency matrix
        int[][] count = new int[n1][n2];
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                int match = 0;

                for (int k = 0; k < n; k++) {
                    if (y1[k] == label1[i] && y2[k] == label2[j]) {
                        match++;
                    }
                }

                count[i][j] = match;
            }
        }

        // Marginals
        int[] count1 = new int[n1];
        int[] count2 = new int[n2];

        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                count1[i] += count[i][j];
                count2[j] += count[i][j];
            }
        }

        // Calculate RAND - Adj
        double rand1 = 0.0;
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                if (count[i][j] >= 2) {
                    rand1 += Math.choose(count[i][j], 2);
                }
            }
        }

        double rand2a = 0.0;
        for (int i = 0; i < n1; i++) {
            if (count1[i] >= 2) {
                rand2a += Math.choose(count1[i], 2);
            }
        }

        double rand2b = 0;
        for (int j = 0; j < n2; j++) {
            if (count2[j] >= 2) {
                rand2b += Math.choose(count2[j], 2);
            }
        }

        double rand3 = rand2a * rand2b;
        rand3 /= Math.choose(n, 2);
        double rand_N = rand1 - rand3;

        // D
        double rand4 = (rand2a + rand2b) / 2;
        double rand_D = rand4 - rand3;

        double rand = rand_N / rand_D;
        return rand;
    }

    public static int checkValiditiy(List<Centroid> centroids){
        for (int i = 0 ; i < centroids.size() ; i++) {
            Centroid centroid = centroids.get(i);
            if(centroid.getTrainingSamples().size() < 2){
                return i;
            }
        }
        return -1;
    }
}
