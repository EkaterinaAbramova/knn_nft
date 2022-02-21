# Rust Smart Contract: K Nearest Neighbours Machine Learning Algo

## Algo Description

Contract uses K Nearest Neighbours ([KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)) algorithm to classify multidimensional datapoints into either class 0 or class 1. A client (a lab) may wish to interact with this NFT in order to obtain the class (0: no cancer, or 1: cancer) of a newly observed data point (e.g. different readings on a patient's biopsy). 

Decision making of the KNN algorithm is based on the Euclidean distance between the test point (one to be classified those class is unknown, e.g. cancer or not) and all of the points in the train dataset (those classes are known and available to the algorithm). 
Note that the client does not need to provide the full training dataset (the NFT contract should have access to it - in the current version only a toy such dataset is used). Hence the client only needs to provide the test data point with the required features (in the present toy data set example just 2 unnamed features). 
The class of the test data point is assigned to be the majority class of the K nearest neighbours (i.e. K nearest training data points to the test point).

The contract gives a selection of datasets to the client, in the current version it provides a choice of two toy dataset examples: `TOY_CANCER_TRAIN` and `TOY_CUSTOMER_TRAIN`, both 10x2 arrays (i.e. 2 features) and their corresponding known target classes `TOY_CANCER_TARGET` and `TOY_CUSTOMER_TARGET`. 

KNN algorithm's requirements:
- It takes one hyperparameter, K, which is a positive odd number typically in the range 1 and 15.
- K specifies how many nearest neighbours from the test data point the algorithm will look to, in order to decide test data point's class.
- The full train dataset is needed each time a new test point classification is to be made.

KNN algorithm's steps:
- Calculates Euclidean distance between the test point and all of the points in the train dataset
- Orders the distances in ascending order
- Selects points corresponding to K first distances
- Examines the most frequent class observed in the selected K points
- Classifies test data point to majority class. 
See ([KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)) for a visual. 

## Prerequisites
  * Make sure Rust is installed as per the prerequisites in [`near-sdk-rs`](https://github.com/near/near-sdk-rs).
  * Make sure [near-cli](https://github.com/near/near-cli) is installed.

## Explore Contract
The source for this contract is in `knn_ml_nft/contract/src/lib.rs`. It provides a method to classify a new unseen data point (test point) into either class 0 or class 1 using a toy dataset of your choice: _cancer_ or _customer_.

## Build Contract
Since the smart contract is technically a library (crate), the cargo run command is unavailable. Instead to build the contract run the following command:
```bash
./build.sh
```
## Test Contract
There are 6 unit tests in this contract, specifically: `test_default_k`, `test_new_k`, `test_run_analysis`, `test_calc_euclidean_dist`, `test_sort_and_argsort`, `test_classify_test_point` which are designed to verify that the contract code is working as indtended.
```bash
cargo test -- --nocapture
```

## Interact With The Contract

### Create Necessary Accounts

This smart contract is intended to be interacted with using development account. Go to [NEAR URL](https://wallet.testnet.near.org) and create a testnet account, for example called `myacc`. For help see [Dev Accounts](https://docs.near.org/docs/concepts/account#dev-accounts). Following this let the near cli generate a private key, kept in the jason file on your computer, and public key as a URL parameter to NEAR wallet by logging into your account from your Terminal (browswer opens up):
```bash
near login
```

Next, using the Terminal create a sub-account (for example `knn_nft.myacc`) to which the contract will be deployed to (best practice), using command:
```bash
near create-account knn_nft.myacc.testnet --masterAccount myacc.testnet
```

### Deploy The Contract to Blockchain
Ensure the cmd is in the dirctory containing `res` folder and run the following command which will deploy theh contract to blockchain:
```bash
near deploy knn_nft.myacc.testnet --wasmFile res/knn_supervised_learning.wasm
```
Alternatively the initialisation can be done together with deployment as a Batch Action:
```bash
near deploy knn_nft.myacc.testnet --wasmFile res/knn_supervised_learning.wasm --initFunction 'new' --initArgs '{"k": 3}'
```

### Initialise Contract
The contract has a default value for the hyperparameter k (`k=5`), or the value can be expertly set when calling the instantiation macro `[init]` with method `new()`:
```bash
near call knn_nft.myacc.testnet new '{"k": 3}' --accountId knn_nft.drkat.testnet
```

Next, specify the toy dataset you would like to work with (`cancer` or `customer`) and provide a test data point (one those class is to be established). For example cancer data set with test point [15.8, 2.0] can be classified either into class 0: no cancer, or into class 1: cancer. To obtain the classification run:
```bash
near call knn_nft.myacc.testnet run_analysis '{"data_set": "cancer", "test_point": [13.9, 1.9]}' --accountId myacc.testnet
```
The established class is provided as a result.

**Get more info at:**

* [Rust Smart Contract Quick Start](https://docs.near.org/docs/develop/contracts/rust/intro)
* [Rust SDK Book](https://www.near-sdk.io/)
