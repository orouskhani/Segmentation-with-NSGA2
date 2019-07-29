package edu.shenzen.maysam.entities.math;

import edu.shenzen.maysam.entities.enums.DistanceSet;

public class Point {

    private double[] coords;
    private int label;

    public Point(double[] coords){
        this.coords = coords;
    }

    public Point() {
    }

    public double[] getCoords() {
        return coords;
    }

    public void setCoords(double[] coords) {
        this.coords = coords;
    }

    public double findDistance(Point y, DistanceSet distanceDif) {
        if(distanceDif == DistanceSet.EUCLIDEAN){
            double sum = 0;
            double[] coords = y.getCoords();
            for(int i = 0 ; i < coords.length ; i++){
                sum += Math.pow(this.coords[i] - coords[i], 2);
            }
            return Math.sqrt(sum);
        }
        else{
            return 0;
        }
    }

    public int getDimension() {
        return coords.length;
    }

    public Double getFeature(int n) {
        return coords[n];
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString() {
        String line = "[";
        for(int i = 0 ; i < coords.length ; i++){
            line = line.concat(String.valueOf(coords[i]));
            if(i != coords.length - 1){
                line += ",";
            }
        }
        line = line.concat("]");
        return line;
    }
}
