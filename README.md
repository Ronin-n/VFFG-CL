# VFFG-CL
VFFG-CL: Virtual Fusion Feature Generation with Curriculum Learning for Missing-Modality Emotion Recognition

# Environment
``` 
python 3.8.0
pytorch >= 1.8.0
numpy 1.22.1
scikit-learn 1.3.2
pandas 1.4.3
```

# Train

+ data

First you should change the data folder path in ```data/config``` and preprocess your data follwing the code in ```preprocess/```.

The preprocess of feature was done handcrafted in several steps, we will make it a automatical running script in the next update. You can download the preprocessed feature to run the code.

+ For Training VFFG-CL on IEMOCAP:

    First Real Fusion Feature Pretraining(RFFP).

    ```bash
    bash scripts/CAP_RFFP.sh AVL [num_of_expr] [GPU_index]
    ```

    If only train Virtual Fusion Feature Generation Network(VFFGN) 

    ```bash
    bash scripts/VFFGN_base/CAP_VFFGN_base.sh [num_of_expr] [GPU_index]
    ```
    If train Virtual Fusion Feature Generation Network(VFFGN) with Curriculum Learning Strategy

    ```bash
    bash scripts/curriculum_learning/CAP_VFFGN_CL.sh [num_of_expr] [GPU_index]
    ```

+ For Training VFFG-CL on MSP-improv: 

    First Real Fusion Feature Pretraining(RFFP).

    ```bash
    bash scripts/MSP_RFFP.sh AVL [num_of_expr] [GPU_index]
    ```

    If only train Virtual Fusion Feature Generation Network(VFFGN) 

    ```bash
    bash scripts/VFFGN_base/MSP_VFFGN_base.sh [num_of_expr] [GPU_index]
    ```
    If train Virtual Fusion Feature Generation Network(VFFGN) with Curriculum Learning Strategy

    ```bash
    bash scripts/curriculum_learning/MSP_VFFGN_CL.sh [num_of_expr] [GPU_index]
