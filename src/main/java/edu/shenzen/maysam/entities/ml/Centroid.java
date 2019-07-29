package edu.shenzen.maysam.entities.ml;

import edu.shenzen.maysam.entities.math.Point;

import java.util.HashMap;
import java.util.Map;

public class Centroid {
    Point features;
    Map<Integer, Point> trainingSamples;

    public Point getFeatures() {
        return features;
    }

    public void setFeatures(Point features) {
        this.features = features;
    }

    public Map<Integer, Point> getTrainingSamples()
    {
        return trainingSamples;
    }

    public void setTrainingSamples(Map<Integer, Point> trainingSamples)
    {
        this.trainingSamples = new HashMap<>(trainingSamples);
    }

    public void addTrainingSample(Integer index, Point trainingSample)
    {
        if (this.trainingSamples == null)
        {
            this.trainingSamples = new HashMap<>();
        }

        this.trainingSamples.put(index, trainingSample);
    }

    public Centroid(Point features, Map<Integer, Point> trainingSamples)
    {
        this.features = features;
        this.trainingSamples = new HashMap<>(trainingSamples);
    }

    public Centroid()
    {
        this.trainingSamples = new HashMap<>();
    }
}
