package edu.shenzen.maysam.util;

import edu.shenzen.maysam.entities.enums.DistanceSet;
import edu.shenzen.maysam.entities.math.Matrix;
import edu.shenzen.maysam.entities.math.Point;
import edu.shenzen.maysam.entities.ml.Centroid;
import org.uma.jmetal.solution.DoubleSolution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MathUtils {

    public static Matrix pdist2(List<Point> xList, List<Point> yList, DistanceSet distanceDif){
        double[][] resultArray = new double[xList.size()][yList.size()];

        for (int i = 0 ; i < xList.size() ; i++) {
            Point x = xList.get(i);
            for (int j = 0 ; j < yList.size() ; j++) {
                Point y = yList.get(j);
                resultArray[i][j] = x.findDistance(y, distanceDif);
            }
        }

        return new Matrix(resultArray);
    }

    public static double pdist2(List<Point> dataset, List<Point> solution) {

        Matrix matrix1 = pdist2(dataset, solution, DistanceSet.EUCLIDEAN);
        double[][] matrix2 = matrix1.getMatrix();

        double sum = 0;
        for(int i = 0 ; i < matrix2.length ; i++){
            sum+= Arrays.stream(matrix2[i])
                    .min()
                    .getAsDouble();
        }

        return sum;
    }

    public static double getDistanceOfCentroidFromReferencePoint(List<Centroid> centroids, Point referencePoint) {
        double distance = 0;
        for (Centroid centroid : centroids) {
           distance += centroid.getTrainingSamples().size() * centroid.getFeatures().findDistance(referencePoint, DistanceSet.EUCLIDEAN);
        }
        return distance;
    }



//    public static void main(String[] args) {
//        double[] rnd = new Random().doubles(12, 0, 1).toArray();
//        List<Point> xList = new ArrayList<>();
//        List<Point> yList = new ArrayList<>();
//
//        for(int i = 0 ; i < rnd.length ; ){
//            double[] point = new double[]{rnd[i], rnd[i+1]};
//            Point p = new Point(point);
//            i = i + 2;
//            if(i < 8) {
//                xList.add(p);
//            }
//            else{
//                yList.add(p);
//            }
//        }
//
//        MathUtils.pdist2(xList, yList, DistanceSet.EUCLIDEAN);
//    }

}
