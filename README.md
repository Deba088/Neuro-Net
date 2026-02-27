# Neuro-Net
This project is all about Deep Learning and Neural Network

Here I have developed an Convolutional Neural Network using Object Oriented Programming.
Cost function used : Sigmoid.
Gradient Descent : Stochastic Gradient Descend.

This is tested using Iris Dataset and it is giving a prediction performance of more than 85% on previously unseen data..


```mermaid
flowchart TD
    subgraph DATA["📦 Data Layer"]
        DS[(Iris Dataset\n150 samples\n4 features\n3 classes)]
        PRE[Preprocessing\nNormalization &\nTrain/Test Split]
        DS --> PRE
    end

    subgraph PACKAGE["🧩 Package — Neural Network Architecture"]
        direction TB

        subgraph INPUT["Input Layer"]
            I1([Sepal Length])
            I2([Sepal Width])
            I3([Petal Length])
            I4([Petal Width])
        end

        subgraph HIDDEN["Hidden Layer\nConvolutional + Fully Connected"]
            H1((Neuron 1))
            H2((Neuron 2))
            H3((Neuron 3))
            H4((Neuron 4))
        end

        subgraph OUTPUT["Output Layer\nSoftmax"]
            O1([Setosa])
            O2([Versicolor])
            O3([Virginica])
        end

        I1 & I2 & I3 & I4 --> H1 & H2 & H3 & H4
        H1 & H2 & H3 & H4 --> O1 & O2 & O3
    end

    subgraph TRAINING["⚙️ Training Pipeline"]
        direction LR
        FWD["Forward Pass\nSigmoid Activation\nŷ = σ(W·x + b)"]
        LOSS["Loss Computation\nCross-Entropy\nL = -Σ y·log(ŷ)"]
        BWD["Backward Pass\nGradient Descent\n∂L/∂W via chain rule"]
        UPD["Weight Update\nSGD\nW = W - α·∇W"]
        FWD --> LOSS --> BWD --> UPD --> FWD
    end

    subgraph EVAL["📊 Evaluation"]
        PRED[Prediction\non Test Set]
        ACC[">85% Accuracy\non Unseen Data"]
        PRED --> ACC
    end

    subgraph IRIS_PY["📄 Iris.py — Entry Point"]
        LOAD[Load Data]
        INIT[Initialise NeuralNetwork\nRandom Weights]
        TRAIN[Train 10,000 iterations]
        TEST[Test & Evaluate]
        LOAD --> INIT --> TRAIN --> TEST
    end

    PRE --> IRIS_PY
    IRIS_PY --> PACKAGE
    PACKAGE --> TRAINING
    TRAINING --> EVAL

    style DATA fill:#0f2027,color:#00e5ff,stroke:#00e5ff
    style PACKAGE fill:#0f2027,color:#00e5ff,stroke:#00e5ff
    style INPUT fill:#1a3a4a,color:#7fffd4,stroke:#7fffd4
    style HIDDEN fill:#1a2a3a,color:#ff9f43,stroke:#ff9f43
    style OUTPUT fill:#1a3a4a,color:#7fffd4,stroke:#7fffd4
    style TRAINING fill:#0f2027,color:#00e5ff,stroke:#00e5ff
    style EVAL fill:#0f2027,color:#00e5ff,stroke:#00e5ff
    style IRIS_PY fill:#0f2027,color:#00e5ff,stroke:#00e5ff
```
