using System;
using System.Collections.Generic;

public class DecisionTree
{
    public int MaxDepth { get; set; }
    private TreeNode Root { get; set; }

    public DecisionTree(int maxDepth)
    {
        MaxDepth = maxDepth;
    }

    public void Train(List<double[]> features, List<int> labels)
    {
        // Build decision tree using training data
        Root = BuildTree(features, labels, 0);
    }

    public int Predict(double[] features)
    {
        // Traverse the decision tree to make a prediction
        TreeNode node = Root;
        while (!node.IsLeaf)
        {
            if (features[node.SplitFeatureIndex] <= node.SplitThreshold)
                node = node.LeftChild;
            else
                node = node.RightChild;
        }
        return node.Prediction;
    }

    private TreeNode BuildTree(List<double[]> features, List<int> labels, int depth)
    {
        // Check termination conditions
        if (depth >= MaxDepth || labels.Count <= 1)
        {
            int majorityLabel = GetMajorityLabel(labels);
            return new TreeNode(true, majorityLabel);
        }

        // Find best split
        int bestFeatureIndex = 0;
        double bestThreshold = 0.0;
        double bestInformationGain = 0.0;

        for (int featureIndex = 0; featureIndex < features[0].Length; featureIndex++)
        {
            for (int i = 0; i < features.Count; i++)
            {
                double threshold = features[i][featureIndex];
                List<int> currentLeftLabels = new List<int>();
                List<int> currentRightLabels = new List<int>();

                for (int j = 0; j < features.Count; j++)
                {
                    if (features[j][featureIndex] <= threshold)
                        currentLeftLabels.Add(labels[j]);
                    else
                        currentRightLabels.Add(labels[j]);
                }

                double informationGain = CalculateInformationGain(labels, currentLeftLabels, currentRightLabels);
                if (informationGain > bestInformationGain)
                {
                    bestInformationGain = informationGain;
                    bestFeatureIndex = featureIndex;
                    bestThreshold = threshold;
                }
            }
        }

        // Split data
        List<double[]> leftFeatures = new List<double[]>();
        List<int> leftLabels = new List<int>();
        List<double[]> rightFeatures = new List<double[]>();
        List<int> rightLabels = new List<int>();

        for (int i = 0; i < features.Count; i++)
        {
            if (features[i][bestFeatureIndex] <= bestThreshold)
            {
                leftFeatures.Add(features[i]);
                leftLabels.Add(labels[i]);
            }
            else
            {
                rightFeatures.Add(features[i]);
                rightLabels.Add(labels[i]);
            }
        }

        // Recursively build left and right subtrees
        TreeNode leftChild = BuildTree(leftFeatures, leftLabels, depth + 1);
        TreeNode rightChild = BuildTree(rightFeatures, rightLabels, depth + 1);

        // Create current node
        return new TreeNode(false, bestFeatureIndex, bestThreshold, leftChild, rightChild);
    }

    private int GetMajorityLabel(List<int> labels)
    {
        // Calculate majority label
        Dictionary<int, int> labelCounts = new Dictionary<int, int>();
        foreach (int label in labels)
        {
            if (!labelCounts.ContainsKey(label))
                labelCounts[label] = 0;
            labelCounts[label]++;
        }

        int majorityLabel = -1;
        int maxCount = 0;
        foreach (var kvp in labelCounts)
        {
            if (kvp.Value > maxCount)
            {
                maxCount = kvp.Value;
                majorityLabel = kvp.Key;
            }
        }

        return majorityLabel;
    }

    private double CalculateInformationGain(List<int> parentLabels, List<int> leftLabels, List<int> rightLabels)
    {
        // Calculate information gain
        double parentEntropy = CalculateEntropy(parentLabels);
        double leftEntropy = CalculateEntropy(leftLabels);
        double rightEntropy = CalculateEntropy(rightLabels);

        int totalSize = leftLabels.Count + rightLabels.Count;
        double leftWeight = (double)leftLabels.Count / totalSize;
        double rightWeight = (double)rightLabels.Count / totalSize;

        double weightedAverageEntropy = leftWeight * leftEntropy + rightWeight * rightEntropy;
        return parentEntropy - weightedAverageEntropy;
    }

    private double CalculateEntropy(List<int> labels)
    {
        // Calculate entropy
        Dictionary<int, int> labelCounts = new Dictionary<int, int>();
        foreach (int label in labels)
        {
            if (!labelCounts.ContainsKey(label))
                labelCounts[label] = 0;
            labelCounts[label]++;
        }

        double entropy = 0.0;
        foreach (var kvp in labelCounts)
        {
            double probability = (double)kvp.Value / labels.Count;
            entropy -= probability * Math.Log(probability, 2);
        }

        return entropy;
    }
}

public class TreeNode
{
    public bool IsLeaf { get; }
    public int Prediction { get; }
    public int SplitFeatureIndex { get; }
    public double SplitThreshold { get; }
    public TreeNode LeftChild { get; }
    public TreeNode RightChild { get; }

    public TreeNode(bool isLeaf, int prediction)
    {
        IsLeaf = isLeaf;
        Prediction = prediction;
    }

    public TreeNode(bool isLeaf, int splitFeatureIndex, double splitThreshold, TreeNode leftChild, TreeNode rightChild)
    {
        IsLeaf = isLeaf;
        SplitFeatureIndex = splitFeatureIndex;
        SplitThreshold = splitThreshold;
        LeftChild = leftChild;
        RightChild = rightChild;
    }
}

public class RandomForestHealthcare
{
    private List<DecisionTree> trees;
    public int NumTrees { get; set; }
    public int MaxFeatures { get; set; }
    public int MaxDepth { get; set; }

    public RandomForestHealthcare(int numTrees, int maxFeatures, int maxDepth)
    {
        NumTrees = numTrees;
        MaxFeatures = maxFeatures;
        MaxDepth = maxDepth;
        trees = new List<DecisionTree>();
    }

    public void Train(List<double[]> features, List<int> labels)
    {
        for (int i = 0; i < NumTrees; i++)
        {
            // Sample a subset of features with replacement
            List<double[]> sampledFeatures = new List<double[]>();
            List<int> sampledLabels = new List<int>();
            for (int j = 0; j < features.Count; j++)
            {
                int index = new Random().Next(0, features.Count);
                sampledFeatures.Add(features[index]);
                sampledLabels.Add(labels[index]);
            }

            DecisionTree tree = new DecisionTree(MaxDepth);
            tree.Train(sampledFeatures, sampledLabels);
            trees.Add(tree);
        }
    }

    public int Predict(double[] features)
    {
        Dictionary<int, int> votes = new Dictionary<int, int>();

        foreach (var tree in trees)
        {
            int prediction = tree.Predict(features);
            if (!votes.ContainsKey(prediction))
            {
                votes[prediction] = 0;
            }
            votes[prediction]++;
        }

        int maxVoteCount = 0;
        int maxVoteLabel = 0;
        foreach (var kvp in votes)
        {
            if (kvp.Value > maxVoteCount)
            {
                maxVoteCount = kvp.Value;
                maxVoteLabel = kvp.Key;
            }
        }

        return maxVoteLabel;
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Example usage:
        List<double[]> features = new List<double[]>();
        List<int> labels = new List<int>();
        // Populate features and labels with healthcare data...

        int numTrees = 10;
        int maxFeatures = 3; // Number of features to consider at each split
        int maxDepth = 5;

        RandomForestHealthcare forest = new RandomForestHealthcare(numTrees, maxFeatures, maxDepth);
        forest.Train(features, labels);

        // Example prediction:
        double[] testFeatures = new double[] { 1.0, 2.0, 3.0 }; // Example test features
        int prediction = forest.Predict(testFeatures);
        Console.WriteLine("Predicted outcome: " + prediction);
    }
}
